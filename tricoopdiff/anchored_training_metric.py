"""Natural local binary-pose units for a bounded decoder's velocity error.

Bounds already constrain displacement. A unit latent error near a sub-Angstrom
template must not implicitly receive the same pose weight as moving a protein
by its entire radius. This module is not enabled by legacy checkpoints.
"""

import math

import torch

from .bounded_binary_flow import BoundedBinaryTorsionFlow
from .bounded_joint_motion import _offset_derivatives
from .single_anchor_flow import SingleAnchorTorsionFlow


PARAMETER_METRIC = 'parameter_binary_offset_mse_v1'
PHYSICAL_METRIC = 'physical_binary_tangent_with_1e-4_latent_ridge_v1'


def binary_offset_metric(flow, state, velocity_error, scales, *, policy):
    count = len(flow.plan.bonds)
    if isinstance(flow, SingleAnchorTorsionFlow):
        translations, rotations = [flow.translation_bound], [flow.rotation_bound]
    elif isinstance(flow, BoundedBinaryTorsionFlow):
        translations, rotations = flow.translation_bounds, flow.rotation_bounds
    else:
        return velocity_error.sum() * 0
    size = 6 * len(translations)
    error = velocity_error[count:count + size].reshape(-1, 2, 3)
    if policy == PARAMETER_METRIC:
        return error.square().mean()
    if policy != PHYSICAL_METRIC:
        raise ValueError('unsupported bounded binary velocity metric')
    scales = torch.as_tensor(scales, dtype=state.dtype, device=state.device).reshape(2)
    if (scales <= 0).any() or not torch.isfinite(scales).all():
        raise ValueError('binary tangent metric requires finite positive receptor scales')
    offsets = state[count:count + size].detach().double().reshape(-1, 2, 3)
    losses = []
    for side, (translation, rotation) in enumerate(zip(translations, rotations)):
        derivative = getattr(flow, 'offset_derivatives', _offset_derivatives)
        dt, dw = derivative(offsets[side], translation, rotation)
        translation_error = dt.to(error) @ error[side, 0] / scales[side]
        rotation_error = dw.to(error) @ error[side, 1] / math.pi
        losses.append((translation_error.square().mean() + rotation_error.square().mean()) / 2)
    return torch.stack(losses).mean() + 1e-4 * error.square().mean()
