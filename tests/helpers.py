"""Reusable synthetic inputs; no test module imports another test module."""

def ligand_input(smiles="CCCOCC"):
    """Build a deterministic, featurized ligand without external assets."""
    from rdkit import Chem
    from torch_geometric.data import HeteroData
    from tricoopdiff.diffdock.datasets.process_mols import get_lig_graph_with_matching
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
    graph = HeteroData()
    get_lig_graph_with_matching(mol, graph, popsize=None, maxiter=None,
                               matching=False, keep_original=False,
                               num_conformers=1, remove_hs=True)
    return {
        "mol": Chem.RemoveHs(mol).ToBinary(),
        "coords": graph['ligand'].pos.unsqueeze(0),
        "edge_index": graph['ligand', 'ligand'].edge_index,
        "edge_mask": graph['ligand'].edge_mask,
        "mask_rotate": graph['ligand'].mask_rotate,
    }


def flow_example(smiles='CCCOCC', seed=12):
    """Build float64 ligand/protein inputs and their SDPI coupling."""
    import numpy as np
    import torch

    from tricoopdiff.sdpi_flow import SDPIFlow

    np.random.seed(seed)
    torch.manual_seed(seed)
    data = ligand_input(smiles)
    data['coords'] = data['coords'].double()
    a = torch.randn(1, 20, 3, dtype=torch.float64) * 10
    b = torch.randn(1, 17, 3, dtype=torch.float64) * 8 + 7
    flow = SDPIFlow()
    coupling = flow.sample_coupling(a, b, data)
    return flow, data, a, b, coupling


def chemistry_example(root):
    """Write an atom-only ligand and its offline CCD reference."""
    import numpy as np
    from rdkit import Chem

    molecule = Chem.MolFromSmiles('CC(=O)[O-]')
    conformer = Chem.Conformer(4)
    coordinates = np.array([[1., 2., 3.], [2.4, 2., 3.],
                            [3., 3., 3.], [3., 1., 3.]])
    for index, (atom, name) in enumerate(zip(molecule.GetAtoms(), ('C2', 'C1', 'O1', 'O2'))):
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.rjust(4))
        info.SetResidueName('TST')
        info.SetResidueNumber(1)
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)
        conformer.SetAtomPosition(index, coordinates[index])
    molecule.AddConformer(conformer)
    pdb = root / 'ligand.pdb'
    # Match the atom-only export that caused loss of bond-order information.
    pdb.write_text('\n'.join(line for line in Chem.MolToPDBBlock(molecule).splitlines()
                              if line.startswith('HETATM')) + '\nEND\n')
    ccd = root / 'TST.cif'
    ccd.write_text('''data_TST
_chem_comp.id TST
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
O2 O -1
C1 C 0
C2 C 0
O1 O 0
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
C1 O1 DOUB
C1 C2 SING
C1 O2 SING
''')
    return pdb, ccd, coordinates


def run_tests(path):
    """Keep documented direct commands while using one fixture-aware runner."""
    import sys
    import pytest

    raise SystemExit(pytest.main([path, "-q", *sys.argv[1:]]))
