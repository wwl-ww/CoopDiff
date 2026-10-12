"""Chemical freedoms must agree between the input graph, expert and executor."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parents[1] / 'scripts')]
from tricoopdiff.initialization import validate_ligand_graph
from tricoopdiff.inputs.ligand_chemistry import legal_torsion_masks, restore_ligand


def edges_for(mol):
    return torch.tensor([(u, v) for bond in mol.GetBonds()
                         for u, v in [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                                      (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())]]).T


@pytest.mark.parametrize('smiles, expected', [('CCCC', 1), ('CC=CC', 0),
                                               ('CC#CC', 0), ('CC(=O)NC', 0),
                                               ('c1ccccc1', 0)])
def test_only_legal_independent_torsions(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    edges = edges_for(mol)
    selected, masks = legal_torsion_masks(mol, edges)
    bonds, checked = validate_ligand_graph(mol, edges, torch.tensor(selected), masks)
    assert len(bonds) == expected, 'chemically forbidden freedom reached executor'
    assert np.array_equal(masks, checked)


def test_validator_rejects_double_bond_even_with_correct_component_mask():
    mol = Chem.MolFromSmiles('CC=CC')
    edges = edges_for(mol)
    selected = torch.zeros(edges.shape[1], dtype=torch.bool)
    selected[2] = True
    with pytest.raises(ValueError, match='chemically forbidden'):
        validate_ligand_graph(mol, edges, selected, np.array([[False, False, True, True]]))


def test_torsion_row_order_follows_selected_edges_after_permutation():
    mol = Chem.MolFromSmiles('CCCCCC')
    edges = edges_for(mol)[:, torch.tensor([8, 4, 7, 0, 5, 9, 2, 1, 6, 3])]
    selected, masks = legal_torsion_masks(mol, edges)
    bonds, _ = validate_ligand_graph(mol, edges, torch.tensor(selected), masks)
    assert len(bonds) == 3
    for (u, v), mask in zip(bonds, masks):
        assert not mask[u] and mask[v], 'mask rows no longer align with directed axes'


def test_ccd_reconstruction_keeps_atom_order_and_rejects_missing_atoms(tmp_path):
    pdb = tmp_path / 'ligand.pdb'
    pdb.write_text('HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \n'
                   'HETATM    2  O1  LIG A   1       1.200   0.000   0.000  1.00  0.00           O  \nEND\n')
    ccd = tmp_path / 'LIG.cif'
    ccd.write_text('data_LIG\n_chem_comp.id LIG\nloop_\n_chem_comp_atom.atom_id\n'
                   '_chem_comp_atom.type_symbol\n_chem_comp_atom.charge\nC1 C 0\nO1 O 0\n'
                   'loop_\n_chem_comp_bond.atom_id_1\n_chem_comp_bond.atom_id_2\n'
                   '_chem_comp_bond.value_order\nC1 O1 DOUB\n')
    report = restore_ligand(pdb, ccd, tmp_path / 'corrected.sdf')
    mol = Chem.SDMolSupplier(str(tmp_path / 'corrected.sdf'), removeHs=False)[0]
    assert report['canonical_atom_names_in_order'] == ['C1', 'O1']
    assert mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE
    assert np.allclose(mol.GetConformer().GetPositions(), [[0, 0, 0], [1.2, 0, 0]])
    ccd.write_text(ccd.read_text().replace('O1 O 0', 'O1 O 0\nC2 C 0'))
    with pytest.raises(ValueError, match='incomplete or modified'):
        restore_ligand(pdb, ccd, tmp_path / 'missing.sdf')
    assert not (tmp_path / 'missing.sdf').exists()


def test_corrected_source_cannot_reuse_v9_masks(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import featurize_common as features
    from argparse import Namespace
    paths = []
    for name in ('p1.pdb', 'p2.pdb', 'corrected.sdf'):
        path = tmp_path / name
        path.write_text(name)
        paths.append(str(path))
    identity = {'model': 'verified/local', 'status': 'local_assets_hashed',
                'files': {'weights.safetensors': 'digest'}}
    args = Namespace(remove_hs=True)
    old = features._featurize_signature(*paths, args, cache_version=9, encoder_identity=identity)
    new = features._featurize_signature(*paths, args, encoder_identity=identity)
    assert old != new, 'corrected chemistry reused the old topology fingerprint'
    torch.save({'cache_version': 9, 'source_signature': old,
                'esm_encoder_identity': identity}, tmp_path / 'case.pt')
    with pytest.raises(FileNotFoundError, match='compatible'):
        features.featurize_complex('case', *paths, args, 'cpu', cache_dir=tmp_path,
                                  cache_only=True, expected_esm_encoder_identity=identity)


def test_new_pdb_cache_cannot_be_built_by_guessing_bond_orders(tmp_path):
    import featurize_common as features
    from argparse import Namespace
    paths = []
    for name in ('p1.pdb', 'p2.pdb', 'ligand.pdb'):
        path = tmp_path / name
        path.write_text(name)
        paths.append(str(path))
    with pytest.raises(ValueError, match='CCD-corrected'):
        features.featurize_complex('case', *paths, Namespace(remove_hs=True), 'cpu')


def test_quarantine_is_explicit_and_preserves_split_and_cluster_membership(tmp_path, monkeypatch):
    import json
    import prepare_chemical_index as preparation
    pdb = tmp_path / 'bad.pdb'
    pdb.write_text('HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n')
    (tmp_path / 'LIG.cif').write_text('unused by this injected incomplete-component error')
    sdf = tmp_path / 'good.sdf'
    sdf.touch()
    index = tmp_path / 'source.json'
    index.write_text(json.dumps({'entries': [{'case': 'good', 'ligand': str(sdf)},
                                            {'case': 'bad', 'ligand': str(pdb)}],
        'metadata': {'split': 'train'},
        'clusters': [{'items': ['good', 'bad'], 'representative': ['bad']},
                     {'items': ['bad'], 'representative': ['bad']}]}))
    def incomplete(*args):
        raise ValueError("incomplete or modified component: missing ['C2']; extra []")
    monkeypatch.setattr(preparation, 'restore_ligand', incomplete)
    with pytest.raises(ValueError, match='index not published'):
        preparation.prepare_index(index, [tmp_path], tmp_path / 'strict')
    assert not (tmp_path / 'strict/index.json').exists()
    out = preparation.prepare_index(index, [tmp_path], tmp_path / 'quarantine',
                                    quarantine_incomplete=True)
    result = json.loads(out.read_text())
    assert [r['case'] for r in result['entries']] == ['good']
    assert result['clusters'] == [{'items': ['good'], 'representative': []}]
    assert result['metadata']['split'] == 'train'
    assert result['metadata']['quarantined_entries'] == 1
    assert result['rejected'][0]['case'] == 'bad'
    assert result['metadata']['chemistry_protocol'] == 'complete_ccd_cohort_v1'
    monkeypatch.setattr(preparation, 'restore_ligand', lambda *args: 1 / 0)
    with pytest.raises(ValueError, match='index not published'):
        preparation.prepare_index(index, [tmp_path], tmp_path / 'technical_failure',
                                  quarantine_incomplete=True)
    assert not (tmp_path / 'technical_failure/index.json').exists()
