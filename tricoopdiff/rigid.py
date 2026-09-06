"""Exact SE(3) composition; row coordinates use x @ rotation.T + translation."""

import torch


def apply_rigid(points, rotation, translation):
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def invert_rigid(rotation, translation):
    inverse = rotation.transpose(-1, -2)
    return inverse, -(inverse @ translation.unsqueeze(-1)).squeeze(-1)


def compose_rigid(rotation_a, translation_a, rotation_b, translation_b):
    """Return A ∘ B, applying B first."""
    return (
        rotation_a @ rotation_b,
        (rotation_a @ translation_b.unsqueeze(-1)).squeeze(-1) + translation_a,
    )


def assemble_interface_pair(poi, rotation_a, translation_a, rotation_b, translation_b):
    """Keep A fixed and place B with D_A ∘ inverse(D_B)."""
    inverse_b, offset_b = invert_rigid(rotation_b, translation_b)
    rotation, translation = compose_rigid(rotation_a, translation_a, inverse_b, offset_b)
    return apply_rigid(poi, rotation, translation), rotation
