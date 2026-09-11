import math

import torch

from tricoopdiff.pose_anchors import joint_surface_anchors, surface_frame


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
