"""Tests for necessary external-endpoint representability checks."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tricoopdiff.online_endpoints import (
    endpoint_internal_geometry, project_endpoint_coupling,
)
from tricoopdiff.pose_anchors import apply_rigid
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.sdpi_flow import _raw_torsions


EDGE_INDEX = np.array([
    [0, 1, 1, 2, 2, 3],
    [1, 0, 2, 1, 3, 2],
])


def test_rigid_endpoint_preserves_sdpi_internal_invariants():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    endpoint = reference @ rotation.T + np.array([4.0, -3.0, 8.0])
    audit = endpoint_internal_geometry(reference, endpoint, EDGE_INDEX)
    assert audit["necessary_condition_passed"]
    assert audit["bond_lengths"]["max_abs_delta_angstrom"] < 1e-12
    assert audit["bonded_one_three_distances"]["max_abs_delta_angstrom"] < 1e-12


def test_distorted_endpoint_is_not_exactly_representable():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ])
    endpoint = reference.copy()
    endpoint[3, 2] += 0.2
    audit = endpoint_internal_geometry(reference, endpoint, EDGE_INDEX)
    assert not audit["necessary_condition_passed"]
    assert audit["bond_lengths"]["max_abs_delta_angstrom"] > 0.1


def test_projected_rigid_endpoint_roundtrips_at_flow_time_one():
    native_ligand = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0], [-2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    ligand_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    poi_rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    endpoint_ligand = apply_rigid(
        native_ligand, ligand_rotation,
        torch.tensor([3.0, -2.0, 4.0], dtype=torch.float64),
    )
    endpoint_poi = apply_rigid(
        native_poi, poi_rotation,
        torch.tensor([-4.0, 5.0, 2.0], dtype=torch.float64),
    )
    ligand_data = {
        "coords": native_ligand,
        "edge_index": torch.as_tensor(EDGE_INDEX),
        "edge_mask": torch.zeros(EDGE_INDEX.shape[1], dtype=torch.bool),
        "mask_rotate": np.zeros((0, native_ligand.shape[1]), dtype=bool),
    }
    coupling, projected, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
        no_torsion=True,
    )
    ligand, poi, body, targets = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report["endpoint_exactly_represented"]
    assert np.allclose(projected["ligand"], endpoint_ligand[0].numpy(), atol=1e-8)
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-8)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)
    assert body.shape == (1, 3, 3)
    assert all(torch.isfinite(value).all() for branch in targets.values()
               for value in (branch.values() if isinstance(branch, dict) else (branch,)))


def test_projected_torsion_endpoint_roundtrips_at_flow_time_one():
    native_ligand = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0], [3.0, 1.0, 1.0], [3.5, 2.0, 1.5],
    ]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0], [-2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    edge_mask[4] = True
    masks = np.array([[False, False, False, True, True]])
    ligand_data = {
        'coords': native_ligand, 'edge_index': edge_index,
        'edge_mask': edge_mask, 'mask_rotate': masks,
    }
    torsioned = _raw_torsions(
        native_ligand[0].numpy(), np.array([[2, 3]]), masks, np.array([0.7]),
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    endpoint_ligand = apply_rigid(
        torch.as_tensor(torsioned)[None], rotation,
        torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64),
    )
    endpoint_poi = native_poi + torch.tensor([[[4.0, 2.0, -1.0]]])
    coupling, _, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
    )
    ligand, poi, _, _ = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report['endpoint_exactly_represented']
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-6)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)


def test_single_atom_fallback_has_translation_only_roundtrip():
    native_ligand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    endpoint_ligand = torch.tensor([[[4.0, -2.0, 7.0]]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ]], dtype=torch.float64)
    endpoint_poi = native_poi + torch.tensor([[[2.0, 3.0, 4.0]]])
    ligand_data = {
        'coords': native_ligand,
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'edge_mask': torch.empty(0, dtype=torch.bool),
        'mask_rotate': np.zeros((0, 1), dtype=bool),
    }
    coupling, _, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
        no_torsion=True,
    )
    ligand, poi, _, _ = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report['ligand_orientation_underdetermined']
    assert report['endpoint_exactly_represented']
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-8)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}", flush=True)
