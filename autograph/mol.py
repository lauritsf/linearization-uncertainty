import re

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric import utils
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")  # type: ignore


ATOM_ENCODER = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "I": 7,
    "P": 8,
    "S": 9,
    "Se": 10,
    "Si": 11,
}
BOND_ENCODER = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_ENCODER[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_ENCODER[bond.GetBondType()]]

    # Skip molecules with no bonds to prevent GNN message-passing errors.
    if len(row) == 0:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    perm = (edge_index[0] * num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    x = torch.tensor(type_idx)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type)
    return data


# --- From molsets.py ---


allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def data2graph(data):
    if isinstance(data, Data):
        node_types = data.x
        edge_types = utils.to_torch_coo_tensor(
            data.edge_index, data.edge_attr + 1, data.num_nodes
        ).to_dense()
        return node_types, edge_types
    return data


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(data, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    atom_types = data.x
    edge_types = data.edge_attr
    all_bonds = [tuple(i) for i in data.edge_index.t().tolist()]

    mol = Chem.RWMol()

    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    visited = set()
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        if tuple(sorted(bond)) in visited:
            continue
        if bond[0] != bond[1]:
            mol.AddBond(bond[0], bond[1], bond_dict[edge_types[i].item()])
            if verbose:
                print(
                    "bond added:",
                    bond[0],
                    bond[1],
                    edge_types[i].item(),
                    bond_dict[edge_types[i].item()],
                )
        visited.add(tuple(sorted(bond)))
    return mol


def build_molecule_with_partial_charges(data, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")
    atom_types = data.x
    edge_types = data.edge_attr
    all_bonds = [tuple(i) for i in data.edge_index.t().tolist()]

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    visited = set()
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        if tuple(sorted(bond)) in visited:
            continue
        visited.add(tuple(sorted(bond)))
        if bond[0] != bond[1]:
            mol.AddBond(bond[0], bond[1], bond_dict[int(edge_types[i])])
            if verbose:
                print(
                    "bond added:",
                    bond[0],
                    bond[1],
                    int(edge_types[i]),
                    bond_dict[int(edge_types[i])],
                )
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            _v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                bond_type = int(b.GetBondType())
                queue.append((b.GetIdx(), bond_type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if bond_type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1

                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [(s, len(s)) for s in sm.split(".")]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def check_stability_one_mol(data, atom_decoder):
    atom_types, edge_types = data2graph(data)

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds, strict=False):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if isinstance(possible_bonds, int):
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def check_stability(molecule_list, atom_decoder):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = len(molecule_list)

    for _i, mol in enumerate(molecule_list):
        validity_results = check_stability_one_mol(mol, atom_decoder)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_molecules)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        "mol_stable": fraction_mol_stable,
        "atm_stable": fraction_atm_stable,
    }
    return validity_dict
