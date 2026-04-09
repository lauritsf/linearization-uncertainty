from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from tqdm import tqdm

from ..mol import (
    build_molecule_with_partial_charges,
    mol2smiles,
)


def files_exist(files: list[Path]) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    # TODO: Investigate if this behavior is desired for all datasets.
    return len(files) != 0 and all(f.exists() for f in files)


def to_list(value):
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class QM9Dataset(InMemoryDataset):
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    atom_encoder: ClassVar[dict[str, int]] = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bonds: ClassVar[dict[BondType, int]] = {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3,
    }
    atom_decoder: ClassVar[list[str]] = ["H", "C", "N", "O", "F"]

    def __init__(self, root, split="train", transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        root = Path(root) / "mol" / "QM9"
        file_idx = {"train": 0, "val": 1, "test": 2}
        self.split = split
        self.file_idx = file_idx[split]
        super().__init__(str(root), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [
            "gdb9.sdf",
            "gdb9.sdf.csv",
            "uncharacterized.txt",
            *self.split_file_name,
        ]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        """Absolute filepaths that must be present in order to skip splitting."""
        files = to_list(self.split_file_name)
        return [Path(self.raw_dir) / f for f in files]

    @property
    def num_node_types(self):
        return len(self.atom_encoder)

    @property
    def num_edge_types(self):
        return len(self.bonds)

    @property
    def processed_file_names(self):
        return [self.split + ".pt"]

    @property
    def smiles_file_name(self):
        return self.split + "_smiles.pt"

    def download(self):
        """Download raw QM9 files."""
        raw_dir = Path(self.raw_dir)
        try:
            import rdkit  # noqa

            file_path = Path(download_url(self.raw_url, str(raw_dir)))
            extract_zip(str(file_path), str(raw_dir))
            file_path.unlink()

            download_url(self.raw_url2, str(raw_dir))
            (raw_dir / "3195404").rename(raw_dir / "uncharacterized.txt")
        except ImportError:
            path = Path(download_url(self.processed_url, str(raw_dir)))
            extract_zip(str(path), str(raw_dir))
            path.unlink()

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split using DataFrame.iloc to preserve type
        shuffled = dataset.sample(frac=1, random_state=42)
        train = shuffled.iloc[:n_train]
        val = shuffled.iloc[n_train : n_train + n_val]
        test = shuffled.iloc[n_train + n_val :]

        train.to_csv(raw_dir / "train.csv")
        val.to_csv(raw_dir / "val.csv")
        test.to_csv(raw_dir / "test.csv")

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=["mol_id"], inplace=True)

        with open(self.raw_paths[2]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(suppl):
            if i in skip or i not in target_df.index:
                continue

            num_atoms = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(self.atom_encoder[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [self.bonds[bond.GetBondType()]]

            # Skip molecules with no bonds to prevent GNN message-passing errors.
            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)

            perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]

            x = torch.tensor(type_idx)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    @property
    def smiles(self):
        file_path = Path(self.processed_dir) / self.smiles_file_name
        if file_path.exists():
            return torch.load(str(file_path), weights_only=False)
        RDLogger.DisableLog("rdApp.*")
        mols_smiles = self.compute_smiles()
        torch.save(mols_smiles, str(file_path))
        return mols_smiles

    def compute_smiles(self):
        mols_smiles = []
        invalid = 0
        disconnected = 0
        for data in tqdm(self):
            mol = build_molecule_with_partial_charges(data, self.atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    disconnected += 1
            else:
                invalid += 1
        print(f"Number of invalid molecules: {invalid / len(self)}")
        print("Number of disconnected molecules:", disconnected)
        return mols_smiles
