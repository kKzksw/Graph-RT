import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, rdPartialCharges
from torch_geometric.data import Data

ATOM_FEATURES = [
    "chiral_center",
    "cip_code",
    "crippen_log_p_contrib",
    "crippen_molar_refractivity_contrib",
    "degree",
    "element",
    "formal_charge",
    "gasteiger_charge",
    "hybridization",
    "is_aromatic",
    "is_h_acceptor",
    "is_h_donor",
    "is_hetero",
    "is_in_ring_size_n",
    "labute_asa_contrib",
    "mass",
    "num_hs",
    "num_radical_electrons",
    "num_valence",
    "tpsa_contrib",
]

BOND_FEATURES = [
    "bondstereo",
    "bondtype",
    "is_conjugated",
    "is_in_ring",
    "is_rotatable",
]

ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
]


def onehot_encode(x, allowable_set):
    return list(map(lambda s: float(x == s), allowable_set))


def encode(x):
    try:
        if x is None or np.isnan(x):
            x = 0.0
    except TypeError:
        pass
    return [float(x)]


def bondtype(bond):
    return onehot_encode(
        x=bond.GetBondType(),
        allowable_set=BOND_TYPES,
    )


def is_in_ring(bond):
    return encode(bond.IsInRing())


def is_conjugated(bond):
    return encode(bond.GetIsConjugated())


def is_rotatable(bond):
    mol = bond.GetOwningMol()
    atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return encode(atom_indices in Lipinski._RotatableBonds(mol))


def bondstereo(bond):
    return onehot_encode(
        x=bond.GetStereo(),
        allowable_set=BOND_STEREO,
    )


def element(atom):
    return onehot_encode(
        x=atom.GetSymbol(),
        allowable_set=ELEMENTS,
    )


def hybridization(atom):
    return onehot_encode(
        x=atom.GetHybridization(),
        allowable_set=HYBRIDIZATIONS,
    )


def cip_code(atom):
    if atom.HasProp("_CIPCode"):
        return onehot_encode(atom.GetProp("_CIPCode"), ["R", "S"])
    return [0.0, 0.0]


def chiral_center(atom):
    return encode(atom.HasProp("_ChiralityPossible"))


def formal_charge(atom):
    return onehot_encode(
        x=min(max(atom.GetFormalCharge(), -1), 1),
        allowable_set=[-1, 1, 1],
    )


def mass(atom):
    return encode(atom.GetMass() / 100)


def num_hs(atom):
    return onehot_encode(
        x=min(atom.GetTotalNumHs(), 4),
        allowable_set=[0, 1, 2, 3, 4],
    )


def num_valence(atom):
    return onehot_encode(
        x=min(atom.GetTotalValence(), 6),
        allowable_set=[0, 1, 2, 3, 4, 5, 6],
    )


def degree(atom):
    return onehot_encode(
        x=min(atom.GetDegree(), 5),
        allowable_set=[0, 1, 2, 3, 4, 5],
    )


def is_aromatic(atom):
    return encode(atom.GetIsAromatic())


def is_hetero(atom):
    mol = atom.GetOwningMol()
    return encode(atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)])


def is_h_donor(atom):
    mol = atom.GetOwningMol()
    return encode(atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)])


def is_h_acceptor(atom):
    mol = atom.GetOwningMol()
    return encode(atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)])


def is_in_ring_size_n(atom):
    ring_size = 0
    for size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if atom.IsInRingSize(size):
            ring_size = size
            break
    return onehot_encode(
        x=ring_size,
        allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10],
    )


def num_radical_electrons(atom):
    return onehot_encode(
        x=min(atom.GetNumRadicalElectrons(), 2),
        allowable_set=[0, 1, 2],
    )


def crippen_log_p_contrib(atom):
    mol = atom.GetOwningMol()
    return encode(Crippen._GetAtomContribs(mol)[atom.GetIdx()][0])


def crippen_molar_refractivity_contrib(atom):
    mol = atom.GetOwningMol()
    return encode(Crippen._GetAtomContribs(mol)[atom.GetIdx()][1])


def tpsa_contrib(atom):
    mol = atom.GetOwningMol()
    return encode(rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()])


def labute_asa_contrib(atom):
    mol = atom.GetOwningMol()
    return encode(rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()])


def gasteiger_charge(atom):
    return encode(atom.GetDoubleProp("_GasteigerCharge"))


def atom_featurizer(atom):
    return np.concatenate(
        [globals()[feature_name](atom) for feature_name in ATOM_FEATURES],
        axis=0,
    )


def bond_featurizer(bond):
    return np.concatenate(
        [globals()[feature_name](bond) for feature_name in BOND_FEATURES],
        axis=0,
    )


def smiles_to_pyg_data(smiles, rt=None, sample_index=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    rdPartialCharges.ComputeGasteigerCharges(mol)

    node_features = np.array(
        [atom_featurizer(atom) for atom in mol.GetAtoms()],
        dtype=np.float32,
    )

    if len(mol.GetBonds()) > 0:
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            features = bond_featurizer(bond).astype(np.float32)
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_features.append(features)
            edge_features.append(features)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features, dtype=np.float32))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim()), dtype=torch.float32)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    if rt is not None:
        data.y = torch.tensor([float(rt)], dtype=torch.float32)
    if sample_index is not None:
        data.sample_index = torch.tensor([int(sample_index)], dtype=torch.long)
    return data


def node_feature_dim():
    mol = Chem.MolFromSmiles("CC")
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return len(atom_featurizer(mol.GetAtomWithIdx(0)))


def edge_feature_dim():
    mol = Chem.MolFromSmiles("CC")
    return len(bond_featurizer(mol.GetBondWithIdx(0)))
