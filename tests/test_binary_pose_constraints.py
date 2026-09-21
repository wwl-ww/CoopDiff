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


def test_binary_initializer_connects_to_anchored_chart_without_native_target():
    from tricoopdiff.anchored_flow import AnchoredTorsionFlow

    data = ligand_input('CCCCCC')
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + 8
    broad = TernaryInitializer().initialize(e3, poi, data, seed=19)[0][0]
    initializer = BinaryPoseInitializer(
        BinaryPoseSide(e3, broad[:3], ((0, 1, 2),)),
        BinaryPoseSide(poi, broad[3:], ((3, 4, 5),)), max_candidates=2)
    chart, audit = AnchoredTorsionFlow.from_binary_initializer(
        initializer, e3, poi, data, seed=19, max_fit_rmsd=.15)
    assert audit['selected_atom_maps'] == [[0, 1, 2], [3, 4, 5]]
    assert len(chart.plan.bonds) == 1
    moved, receptor, _ = chart.decode(chart.ligand.new_tensor([.7]))
    assert torch.equal(moved[:3], chart.ligand[:3])
    assert torch.allclose(torch.cdist(moved[3:], receptor),
                          torch.cdist(chart.ligand[3:], chart.poi), atol=1e-5)
    changed = {**data, 'coords': data['coords'] + 1000}
    repeated, _ = AnchoredTorsionFlow.from_binary_initializer(
        initializer, e3, poi, changed, seed=19, max_fit_rmsd=.15)
    assert torch.equal(chart.ligand, repeated.ligand)
    assert torch.equal(chart.poi, repeated.poi)


def test_binary_initializer_body_maps_current_poi_frame():
    data = ligand_input('CCCCCC')
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + 8
    broad = TernaryInitializer().initialize(e3, poi, data, seed=19)[0][0]
    initializer = BinaryPoseInitializer(
        BinaryPoseSide(e3, broad[:3], ((0, 1, 2),)),
        BinaryPoseSide(poi, broad[3:], ((3, 4, 5),)), max_candidates=2)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    current = transform(poi, rotation, torch.tensor([7., -3., 9.]))
    _, placed, body, _ = initializer.initialize(e3, current, data, seed=19)
    predicted = (current - current.mean(0)) @ body[0].T
    assert torch.allclose(predicted, placed[0] - placed[0].mean(0), atol=1e-5), \
        'body rotation must start in the current POI frame, not the binary asset frame'


def test_anchored_chart_rejects_nonrigid_e3_frame_mismatch():
    from tricoopdiff.anchored_flow import AnchoredTorsionFlow

    data = ligand_input('CCCCCC')
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + 8
    broad = TernaryInitializer().initialize(e3, poi, data, seed=19)[0][0]
    initializer = BinaryPoseInitializer(
        BinaryPoseSide(e3, broad[:3], ((0, 1, 2),)),
        BinaryPoseSide(poi, broad[3:], ((3, 4, 5),)), max_candidates=1)
    changed = e3.clone()
    changed[0, 0] += 1
    try:
        AnchoredTorsionFlow.from_binary_initializer(
            initializer, changed, poi, data, seed=19, max_fit_rmsd=.15)
    except ValueError as error:
        assert 'side1 protein' in str(error)
    else:
        raise AssertionError('incompatible E3 structure must not become a hard constraint')


def test_candidate_gate_precedes_ranking_and_never_falls_back():
    from unittest.mock import patch

    data = ligand_input('CCCCCC')
    ligand, e3, poi, side1, side2 = fixture()
    initializer = BinaryPoseInitializer(side1, side2, max_candidates=2)
    row = assemble_binary_pose_candidates(ligand, e3, side1, side2)[0]

    def gate(candidate, maps):
        assert maps == (side1.atom_maps[0], side2.atom_maps[0])
        if candidate['map_pair_index'] == 0:
            raise ValueError('distorted covalent bonds')
        return {'valid': True}

    def reject_all(candidate, maps):
        raise ValueError('distorted covalent bonds')

    with patch('tricoopdiff.binary_pose_constraints.embed_with_fixed_binary_ends',
               return_value=ligand), patch(
                   'tricoopdiff.binary_pose_constraints.assemble_binary_pose_candidates',
                   side_effect=lambda *args, **kwargs: [dict(row)]):
        # Identical scores: the unfiltered legacy path selects candidate zero.
        assert initializer.initialize(e3, poi, data, seed=19)[3]['selected_map_pair_index'] == 0
        result = initializer.initialize(e3, poi, data, seed=19, candidate_validator=gate)
        audit = result[3]
        assert audit['selected_map_pair_index'] == 1
        assert audit['candidate_count'] == 1 and len(audit['constraint_rejections']) == 1
        assert audit['constraint_audit'] == {'valid': True}
        try:
            initializer.initialize(e3, poi, data, seed=19, candidate_validator=reject_all)
        except ValueError as error:
            assert 'no valid two-end candidates' in str(error)
            assert 'distorted covalent bonds' in str(error)
        else:
            raise AssertionError('all rejected candidates must fail closed')


def test_chart_filters_stretched_linker_before_selecting_reference():
    from unittest.mock import patch
    from tricoopdiff.anchored_flow import AnchoredTorsionFlow

    data = ligand_input('CCCCCC')
    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    poi = e3 + 8
    valid = TernaryInitializer().initialize(e3, poi, data, seed=19)[0][0]
    invalid = valid.clone()
    invalid[3:] += 10
    initializer = BinaryPoseInitializer(
        BinaryPoseSide(e3, valid[:3], ((0, 1, 2),)),
        BinaryPoseSide(poi, valid[3:], ((3, 4, 5),)), max_candidates=2)
    with patch('tricoopdiff.binary_pose_constraints.embed_with_fixed_binary_ends',
               side_effect=[invalid, valid]):
        chart, audit = AnchoredTorsionFlow.from_binary_initializer(
            initializer, e3, poi, data, seed=19, max_fit_rmsd=.15)
    assert audit['selected_map_pair_index'] == 1
    assert audit['candidate_count'] == 1
    assert 'distorted covalent bonds' in audit['constraint_rejections'][0]['error']
    assert audit['constraint_audit']['policy'] == 'anchored_validity_before_clash_selection_v1'
    assert torch.allclose(chart.ligand, valid, atol=1e-5)
