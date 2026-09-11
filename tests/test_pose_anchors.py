import math

import torch

from tricoopdiff.pose_anchors import (
    joint_surface_anchors, proper_rigid_fit, surface_frame,
)
from tricoopdiff.expert_anchors import select_pose_consensus


def test_surface_frame_is_right_handed_and_orthonormal():
    frame = surface_frame(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[2.0, 1.0, 1.0]]),
    )
    assert torch.allclose(frame @ frame.transpose(-1, -2),
                          torch.eye(3).unsqueeze(0), atol=1e-6)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(1), atol=1e-6)


def test_joint_anchor_maps_contact_point_and_opposes_normals():
    source_point = torch.tensor([[1.0, 2.0, 3.0]])
    source_normal = torch.tensor([[0.0, 0.0, 1.0]])
    source_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    target_point = torch.tensor([[5.0, -2.0, 1.0]])
    target_normal = torch.tensor([[0.0, 1.0, 0.0]])
    target_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        source_point, source_normal, source_tangent,
        target_point, target_normal, target_tangent,
        torch.tensor([0.0]),
    )
    assert torch.allclose(anchors.place(source_point)[0, 0], target_point[0], atol=1e-6)
    mapped_normal = source_normal @ anchors.rotations[0]
    assert torch.allclose(mapped_normal, -target_normal, atol=1e-6)


def test_twists_create_distinct_orientations_at_the_same_position():
    point = torch.zeros(1, 3)
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        point, normal, tangent, point, normal, tangent,
        torch.tensor([0.0, math.pi / 2]),
    )
    assert anchors.rotations.shape == (2, 3, 3)
    assert torch.allclose(anchors.place(point)[:, 0], point.expand(2, -1), atol=1e-6)
    assert not torch.allclose(anchors.rotations[0], anchors.rotations[1])


def test_proper_rigid_fit_recovers_exact_transform():
    source = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
    ])
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    translation = torch.tensor([3.0, -2.0, 1.0])
    target = source @ rotation.T + translation
    fitted_rotation, fitted_translation, rmsd = proper_rigid_fit(source, target)
    fitted = source @ fitted_rotation.T + fitted_translation
    assert torch.allclose(fitted, target, atol=1e-5)
    assert torch.det(fitted_rotation) > 0
    assert rmsd < 1e-5


def test_pose_consensus_selects_dense_ternary_mode():
    def candidate(offset):
        coords = torch.tensor(offset, dtype=torch.float32).reshape(1, 1, 3)
        return {
            "ligand": coords, "poi": coords, "penalty": (0, 0, 0.0, 0.0),
        }

    candidates = [
        candidate([0.0, 0.0, 0.0]), candidate([0.2, 0.0, 0.0]),
        candidate([-0.2, 0.0, 0.0]), candidate([20.0, 0.0, 0.0]),
    ]
    selected = select_pose_consensus(candidates, distance_threshold=1.0)
    assert any(selected is candidate for candidate in candidates[:3])
    assert selected["consensus_neighbors_10A"] == 3
