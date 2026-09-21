"""Chemical augmentation preserves connected pieces and never invents warheads."""

from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from test_initialization import ligand_input
from tricoopdiff.anchored_fragments import select_training_fragments
from tricoopdiff.anchored_training import prepare_augmented_anchored_sample
from tricoopdiff.anchored_torsion import chemical_rotatable_pairs, compile_anchored_torsions


def fixture(smiles='CCCCCCCCCC'):
    data = ligand_input(smiles)
    ligand = data['coords'][0]
    return data, (ligand[:3] + torch.tensor([0., 0., 2.]),
                   ligand[-3:] + torch.tensor([0., 0., -2.]))


def select(data, proteins):
    return select_training_fragments(
        data['mol'], data['edge_index'], data['edge_mask'], data['coords'][0], proteins,
        generator=torch.Generator().manual_seed(7315))


def test_cut_fragments_are_disjoint_connected_and_reproducible():
    data, proteins = fixture()
    result = select(data, proteins)
    assert result is not None
    assert result['atom_maps'] == select(data, proteins)['atom_maps']
    first, second = map(set, result['atom_maps'])
    middle = set(result['audit']['middle_atoms'])
    assert not first & second and not (first | second) & middle
    assert first | second | middle == set(range(10))
    assert all(len(Chem.GetMolFrags(fragment)) == 1 for fragment in result['fragments'])
    assert result['audit']['chemical_warhead_identity_claimed'] is False
    sample = prepare_augmented_anchored_sample(
        {'coords': proteins[0][None]}, {'coords': proteins[1][None]}, data,
        generator=torch.Generator().manual_seed(7315), evidence_id='test:chemical-cuts')
    assert sample is not None
    assert sample.conditions['ligand'][list(middle), 3].all(), 'middle must stay unknown'
    assert not sample.conditions['ligand'][:, 2].any(), 'middle is not a known linker'
    assert max(sample.audit['target'].values()) < 1e-3


def test_rigid_or_noncontacting_ligands_are_not_forced_into_augmentation():
    for smiles in ('c1ccccc1', 'CCC'):
        data, proteins = fixture(smiles)
        assert select(data, proteins) is None
    data, proteins = fixture()
    assert select(data, tuple(p + 1000 for p in proteins)) is None


def test_chemical_filter_excludes_double_triple_and_amide_bonds():
    for smiles, rejected in [('CC=CC', (1, 2)), ('CC#CC', (1, 2)),
                             ('CC(=O)NCC', (1, 3))]:
        assert frozenset(rejected) not in chemical_rotatable_pairs(Chem.MolFromSmiles(smiles))
    assert frozenset((1, 2)) in chemical_rotatable_pairs(Chem.MolFromSmiles('CCCC'))
    data = ligand_input('CCCC=CCCC')
    plan = compile_anchored_torsions(8, data['edge_index'], data['edge_mask'],
                                     (0, 1, 2), (5, 6, 7), molecule=data['mol'])
    assert any(reason == 'nonrotatable_chemical_bond' for _, reason in plan.rejected)
    assert all(set(bond) != {3, 4} for bond in plan.bonds)


if __name__ == '__main__':
    test_cut_fragments_are_disjoint_connected_and_reproducible()
    test_rigid_or_noncontacting_ligands_are_not_forced_into_augmentation()
    test_chemical_filter_excludes_double_triple_and_amide_bonds()
    print('Anchored fragment augmentation checks passed')
