from collections.abc import Collection

import networkx as nx

# --- Dataset Validity Functions ---
from polygraph.datasets.planar import is_planar_graph
from polygraph.metrics import VUN

# --- PolyGraph Imports (Strict Paths) ---
from polygraph.metrics.base import FrechetDistance, PolyGraphDiscrepancy
from polygraph.metrics.molecule_pgd import MoleculePGD
from polygraph.utils.descriptors import (
    ClusteringHistogram,
    DegreeHistogram,
    EigenvalueHistogram,
    OrbitCounts,
    RandomGIN,
)
from polygraph.utils.descriptors.molecule_descriptors import (
    ChemNetDescriptor,
)
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from ..mol import (
    build_molecule_with_partial_charges,
    mol2smiles,
)
from ..mol import (
    check_stability as check_valency_stability,
)


class FrechetChemNetDistance(FrechetDistance[Chem.Mol]):
    """Implementation of FCD using PolyGraph's primitives."""

    def __init__(self, reference_molecules: Collection[Chem.Mol]):
        # descriptor_fn is the correct argument name
        super().__init__(
            reference_graphs=reference_molecules,
            descriptor_fn=ChemNetDescriptor(dim=128),
        )


# --- Descriptor Registries ---

GENERIC_DESCRIPTORS = {
    "degree": DegreeHistogram(max_degree=100),
    "clustering": ClusteringHistogram(bins=100),
    "spectre": EigenvalueHistogram(n_bins=200),
    "orbit": OrbitCounts(graphlet_size=4),
    "gin": RandomGIN(node_feat_loc=None, input_dim=1, edge_feat_loc=None, edge_feat_dim=0, seed=42),
}


class SamplingMetrics:
    """Unified metric framework providing PGD summary, VUN, and Ratio."""

    def __init__(self, datamodule, metrics_list, num_ref_graphs=None):
        self.metrics_list = metrics_list

        # 1. Load All Splits
        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader(), limit=num_ref_graphs)
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader(), limit=num_ref_graphs)
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader(), limit=num_ref_graphs)

        self.num_graphs_val = len(self.val_graphs)
        self.num_graphs_test = len(self.test_graphs)

        # Ensure train_graphs and test_graphs have the same number of samples
        # for baseline calculation

        min_len = min(len(self.train_graphs), len(self.test_graphs))

        if len(self.train_graphs) > min_len:
            self.train_graphs = self.train_graphs[:min_len]

        if len(self.test_graphs) > min_len:
            self.test_graphs = self.test_graphs[:min_len]
        # 2. Configure Descriptors
        if "gin" not in metrics_list:
            metrics_list.append("gin")

        active_descriptors = {
            k: GENERIC_DESCRIPTORS[k] for k in metrics_list if k in GENERIC_DESCRIPTORS
        }

        # 3. Pre-Initialize PGD Evaluators
        print("Initializing PGD and computing baselines...")
        self.pgd_val = PolyGraphDiscrepancy(self.val_graphs, active_descriptors)
        self.pgd_test = PolyGraphDiscrepancy(self.test_graphs, active_descriptors)

        # 4. Compute Baseline for Ratio
        # Ratio = Score(Gen, Test) / Score(Train, Test)
        self.train_test_baseline = self.pgd_test.compute(self.train_graphs)

        # 5. Configure VUN Validity
        ds_name = getattr(datamodule, "dataset_names", "")
        if isinstance(ds_name, list):
            ds_name = ds_name[0]
        ds_name = ds_name.lower()

        val_fn = None
        if "planar" in metrics_list or "planar" in ds_name:
            val_fn = is_planar_graph

        self.vun_evaluator = VUN(train_graphs=self.train_graphs, validity_fn=val_fn)

        self.to_log_metrics = [
            "validity",
            "uniqueness",
            "novelty",
            "vun_score",
            "val_pgd",
            "test_pgd",
            "val_ratio",
            "test_ratio",
            "val_pgd_degree",
            "test_pgd_degree",
            "val_pgd_clustering",
            "test_pgd_clustering",
            "val_pgd_orbit",
            "test_pgd_orbit",
            "val_pgd_spectre",
            "test_pgd_spectre",
            "val_pgd_gin",
            "test_pgd_gin",
        ]

    def loader_to_nx(self, loader, limit=None) -> list[nx.Graph]:
        nx_graphs = []
        if not loader:
            return []
        iterator = loader

        for batch in iterator:
            data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else batch
            for data in data_list:
                if isinstance(data, Data):
                    nx_graphs.append(to_networkx(data, to_undirected=True, remove_self_loops=True))
                elif isinstance(data, nx.Graph):
                    nx_graphs.append(data)

                if limit is not None and limit > 0 and len(nx_graphs) >= limit:
                    return nx_graphs
        return nx_graphs

    def __call__(self, generated_graphs, split="val", fast=True):
        gen_nx = []
        for g in generated_graphs:
            if isinstance(g, Data):
                g = to_networkx(g, to_undirected=True, remove_self_loops=True)
            gen_nx.append(g)

        if len(gen_nx) < 5:
            return {"error": "Insufficient samples (need >5)"}

        # --- A. VUN ---
        vun_results = self.vun_evaluator.compute(gen_nx)

        # --- B. Format Base ---
        final_stats = {
            # Polygraph VUN keys are 'valid', 'unique', 'novel'
            "validity": vun_results.get("valid", 0.0),
            "uniqueness": vun_results["unique"],
            "novelty": vun_results["novel"],
            "vun_score": vun_results.get(
                "valid_unique_novel", vun_results.get("unique_novel", 0.0)
            ),
        }

        # --- C. Expensive Distribution Metrics (PGD & Ratio) ---
        if not fast or split == "test":
            evaluator = self.pgd_test if split == "test" else self.pgd_val
            pgd_results = evaluator.compute(gen_nx)

            baseline_score = self.train_test_baseline["pgd"]
            ratio = pgd_results["pgd"] / (baseline_score + 1e-9)

            final_stats.update(
                {
                    f"{split}_pgd": pgd_results["pgd"],
                    f"{split}_ratio": ratio,
                }
            )

            for feat, score in pgd_results["subscores"].items():
                final_stats[f"{split}_pgd_{feat}"] = score

        return final_stats


class MoleculeMetrics:
    """Evaluates generated molecules using PGD, FCD, and stability checks."""

    def __init__(self, datamodule):
        self.atom_decoder = getattr(datamodule, "atom_decoder", None)
        self.train_mols = self._to_mols(datamodule.train_smiles)

        # Robustly load test SMILES from dataset objects directly if possible
        test_smiles = []
        if hasattr(datamodule, "test_dataset") and datamodule.test_dataset is not None:
            datasets = (
                datamodule.test_dataset.datasets
                if hasattr(datamodule.test_dataset, "datasets")
                else [datamodule.test_dataset]
            )
            for ds in datasets:
                if hasattr(ds, "smiles") and ds.smiles is not None:
                    test_smiles.extend(ds.smiles)

        # Fallback to dataloader extraction if direct access failed
        if not test_smiles:
            print(
                "⚠️ MoleculeMetrics: Direct SMILES access failed, "
                "falling back to dataloader extraction."
            )
            self.test_mols = self._extract_mols(datamodule.test_dataloader())
        else:
            self.test_mols = self._to_mols(test_smiles)

        if not self.test_mols:
            raise ValueError(
                "MoleculeMetrics: No valid reference molecules found in test_dataloader."
            )

        # 1. Molecular PGD
        self.mol_pgd = MoleculePGD(self.test_mols)

        # 2. FCD Evaluator
        self.fcd = FrechetChemNetDistance(self.test_mols)

        # 3. Novelty Setup
        self.train_smiles_set = set(datamodule.train_smiles)

        # 4. Standard Interface Attributes (Required by SequenceModel)
        self.num_graphs_test = len(self.test_mols)
        # Safe fallback for validation count if not available
        if hasattr(datamodule, "val_dataset") and datamodule.val_dataset:
            self.num_graphs_val = len(datamodule.val_dataset)
        else:
            self.num_graphs_val = self.num_graphs_test

        self.to_log_metrics = [
            "validity",
            "uniqueness",
            "novelty",
            "val_mol_pgd",
            "test_mol_pgd",
            "val_fcd",
            "test_fcd",
            "mol_stable",
            "atm_stable",
            "mol_sanitize",
            "atm_sanitize",
        ]

    def _to_mols(self, smiles_list: list[str]) -> list[Chem.Mol]:
        """Converts SMILES to RDKit Mols, filtering invalids."""
        mols = []
        if not smiles_list:
            return mols
        for s in smiles_list:
            if s:
                m = Chem.MolFromSmiles(s)
                if m:
                    mols.append(m)
        return mols

    def _extract_mols(self, loader, limit=None) -> list[Chem.Mol]:
        """Extracts SMILES from loader and converts to Mols."""
        smiles = []
        if not loader:
            return []
        for batch in loader:
            if hasattr(batch, "smiles"):
                smiles.extend(batch.smiles)
            elif isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], str):
                smiles.extend(batch)

            if limit is not None and limit > 0 and len(smiles) >= limit:
                break
        return self._to_mols(smiles)

    @staticmethod
    def check_stability_sanitize(smiles_list):
        """Checks valency constraints."""
        mol_stable = 0
        atom_stable = 0
        total_atoms = 0

        for sm in smiles_list:
            if not sm:
                continue
            try:
                mol = Chem.MolFromSmiles(sm, sanitize=False)
            except Exception:
                mol = None

            if mol:
                try:
                    Chem.SanitizeMol(mol)
                    is_valid = True
                except Exception:
                    is_valid = False

                if is_valid:
                    mol_stable += 1

                n_atoms = mol.GetNumAtoms()
                total_atoms += n_atoms
                if is_valid:
                    atom_stable += n_atoms

        n = len(smiles_list) if smiles_list else 1
        return mol_stable / n, atom_stable / (total_atoms + 1e-9)

    def __call__(self, generated_samples: list[str] | list[Data], split="val", fast=True):
        # Convert Data objects to SMILES if necessary
        smiles_list: list[str] = []
        if generated_samples and isinstance(generated_samples[0], Data):
            for g in generated_samples:
                if isinstance(g, Data):
                    if self.atom_decoder is None:
                        pass

                    try:
                        mol = build_molecule_with_partial_charges(g, self.atom_decoder)
                        s = mol2smiles(mol)
                        smiles_list.append(s if s is not None else "")
                    except Exception as e:
                        print(f"Error converting graph to molecule: {e}")
                        smiles_list.append("")
                else:
                    smiles_list.append(str(g))
        else:
            # Already SMILES
            smiles_list = generated_samples  # type: ignore

        # Convert generated SMILES to Mols for VUN and stability
        generated_mols = self._to_mols(smiles_list)

        if len(generated_mols) < 2:
            return {"error": "Insufficient valid molecules"}

        # 1. Stability (Legacy/Sanitize)
        mol_stab_sanitize, atm_stab_sanitize = self.check_stability_sanitize(smiles_list)

        # 2. Valency Stability (Strict - Paper Benchmark)
        if generated_samples and isinstance(generated_samples[0], Data):
            valency_res = check_valency_stability(generated_samples, self.atom_decoder)
        else:
            valency_res = {"mol_stable": 0.0, "atm_stable": 0.0}

        # 3. VUN (Validity, Uniqueness, Novelty)
        valid_smiles = [s for s in smiles_list if s and Chem.MolFromSmiles(s, sanitize=True)]
        n_gen = len(smiles_list)
        validity = len(valid_smiles) / n_gen if n_gen > 0 else 0.0

        unique_set = set(valid_smiles)
        uniqueness = len(unique_set) / len(valid_smiles) if valid_smiles else 0.0

        novel_smiles = [s for s in unique_set if s not in self.train_smiles_set]
        novelty = len(novel_smiles) / len(unique_set) if unique_set else 0.0

        final_stats = {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            "mol_stable": valency_res["mol_stable"],
            "atm_stable": valency_res["atm_stable"],
            "mol_sanitize": mol_stab_sanitize,
            "atm_sanitize": atm_stab_sanitize,
        }

        # 3. Expensive Distribution Metrics (PGD & FCD)
        if not fast or split == "test":
            # 1. PGD
            current_pgd = self.mol_pgd

            # TabPFN (used by PolyGraph PGD) has a hard limit of 10,000 samples.
            pgd_limit = 10_000

            n_gen = len(generated_mols)
            n_ref = len(self.test_mols)
            target_n = min(n_gen, n_ref, pgd_limit)

            if n_gen != target_n or n_ref != target_n:
                # We need to resize one or both sets
                # print(f"⚠️ PGD Resizing: Gen={n_gen}, Ref={n_ref} -> Target={target_n}")

                # Resize Generated Set
                pgd_gen_mols = generated_mols[:target_n]

                # Re-init PGD only if the reference size changed from what was cached
                if target_n != n_ref:
                    # Creating new PGD instance with subset of reference molecules
                    current_pgd = MoleculePGD(self.test_mols[:target_n])
            else:
                pgd_gen_mols = generated_mols

            pgd_res = current_pgd.compute(pgd_gen_mols)

            # 2. FCD
            # FIXED: self.fcd.compute returns a float, not a dict
            fcd_score = self.fcd.compute(generated_mols)

            final_stats.update(
                {
                    f"{split}_mol_pgd": pgd_res["pgd"],
                    f"{split}_fcd": fcd_score,
                }
            )

        final_stats["smiles"] = smiles_list

        return final_stats


def get_dataset_metric(dataset_name, datamodule, **kwargs):
    if dataset_name.upper() in ["QM9", "MOSES", "GUACAMOL", "ZINC"]:
        return MoleculeMetrics(datamodule)

    return SamplingMetrics(
        datamodule,
        metrics_list=["degree", "clustering", "orbit", "spectre", "gin"],
        **kwargs,
    )
