"""Known binary poses assemble candidates without ternary native geometry."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

from tricoopdiff.binary_pose_constraints import (BinaryPoseSide,
                                                  BinaryPoseInitializer,
                                                  assemble_binary_pose_candidates)
from tricoopdiff.initialization import TernaryInitializer
from test_initialization import ligand_input


def transform(points, rotation, translation):
    return points @ rotation.T + translation


def fixture():
    ligand = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.],
                           [2., 1., 0.], [3., 1., 1.], [4., 0., 1.]], dtype=torch.float64)
    protein1 = torch.tensor([[0., 2., 0.], [1., 3., 1.], [-1., 2., 1.]], dtype=torch.float64)
    protein2 = torch.tensor([[3., -2., 0.], [4., -3., 1.], [5., -2., 2.]], dtype=torch.float64)
    side1 = BinaryPoseSide(protein1, ligand[:3], ((0, 1, 2),))
    side2 = BinaryPoseSide(protein2, ligand[3:], ((3, 4, 5),))
    return ligand, protein1, protein2, side1, side2


def test_assembly_preserves_both_local_binary_complexes():
    ligand, protein1, _, side1, side2 = fixture()
    angle = torch.tensor(torch.pi / 2, dtype=torch.float64)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.],
                             [torch.sin(angle), torch.cos(angle), 0.],
                             [0., 0., 1.]], dtype=torch.float64)
    current_e3 = transform(protein1, rotation, torch.tensor([7., -2., 4.]))
    result = assemble_binary_pose_candidates(ligand, current_e3, side1, side2)[0]
    assert result['side1_fit_rmsd'] < 1e-6
    assert result['side2_fit_rmsd'] < 1e-6
    assert torch.allclose(result['ligand_coords'][:3],
                          transform(ligand[:3], rotation, torch.tensor([7., -2., 4.])).float(),
                          atol=1e-5)
    source_distances = torch.cdist(torch.as_tensor(side2.fragment_coords),
                                   torch.as_tensor(side2.protein_coords))
    placed_distances = torch.cdist(result['ligand_coords'][3:], result['poi_coords'])
    assert torch.allclose(source_distances.float(), placed_distances, atol=1e-5)


def test_independent_binary_frames_do_not_change_assembled_candidates():
    ligand, protein1, _, side1, side2 = fixture()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    translation = torch.tensor([-11., 5., 3.], dtype=torch.float64)
    moved2 = BinaryPoseSide(transform(torch.as_tensor(side2.protein_coords), rotation, translation),
                            transform(torch.as_tensor(side2.fragment_coords), rotation, translation),
                            side2.atom_maps)
    first = assemble_binary_pose_candidates(ligand, protein1, side1, side2)[0]
    second = assemble_binary_pose_candidates(ligand, protein1, side1, moved2)[0]
    assert torch.allclose(first['ligand_coords'], second['ligand_coords'], atol=1e-5)
    assert torch.allclose(first['poi_coords'], second['poi_coords'], atol=1e-5)


def test_symmetry_maps_are_enumerated_and_deterministically_ranked():
    ligand, protein1, _, side1, side2 = fixture()
    symmetric = BinaryPoseSide(side1.protein_coords, side1.fragment_coords,
                               ((0, 1, 2), (1, 0, 2)))
    rows = assemble_binary_pose_candidates(ligand, protein1, symmetric, side2)
    assert len(rows) == 2
    assert [(row['map1_index'], row['map2_index']) for row in rows] == [(0, 0), (1, 0)]


def test_initializer_reuses_seeded_etkdg_not_input_native_coordinates():
    data = ligand_input('CCCCCC')
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = torch.tensor([[8., 0., 0.], [10., 0., 0.], [8., 3., 0.]])
    broad = TernaryInitializer().initialize(e3, poi, data, seed=19)[0][0]
    side1 = BinaryPoseSide(e3, broad[:3], ((0, 1, 2),))
    side2 = BinaryPoseSide(poi, broad[3:], ((3, 4, 5),))
    initializer = BinaryPoseInitializer(side1, side2)
    first = initializer.initialize(e3, poi, data, seed=19)
    changed = {**data, 'coords': torch.randn_like(data['coords']) * 1000}
    second = initializer.initialize(e3, poi, changed, seed=19)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[3]['init_mode'] == 'binary_pose'
    assert first[3]['native_information_used'] is False
    assert max(first[3]['side_fit_rmsd']) < .15
    assert 'protein_ca_pairs_below_8A' in first[3]['geometry']
