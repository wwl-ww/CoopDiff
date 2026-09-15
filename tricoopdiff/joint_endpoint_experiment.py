"""Training-only oracle targets for a bounded joint rigid endpoint experiment.

This module deliberately reads native ligand and POI coordinates.  It is an
oracle for training diagnostics/targets and must never be called by blind
sampling or inference code.  Torsion is intentionally unsupported: exact
recovery requires current and native ligand coordinates to differ only rigidly.

Coordinates follow the repository row-vector convention::

    x_new = x @ rotation.T + translation

``SDPIFlow.step`` applies branch A to the ligand and ``A o inverse(B)`` to the
POI.  If S maps current POI to native POI, the required branch transforms are
therefore A and ``B = inverse(S) o A``.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch

from .pose_anchors import proper_rigid_fit
from .rigid import compose_rigid, invert_rigid


Tensor = torch.Tensor
RigidTarget = Dict[str, Dict[str, Tensor]]


def _validate_points(name: str, points: Tensor) -> Tensor:
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if (points.ndim != 3 or points.shape[-1] != 3 or points.shape[-2] < 3
            or not torch.is_floating_point(points) or not torch.isfinite(points).all()):
        raise ValueError(f'{name} must be finite floating-point [B, N, 3] with N >= 3')
    return points


def stable_so3_log(rotation: Tensor, *, reject_near_pi: bool = False,
                   pi_tolerance: float = 1e-5) -> Tensor:
    """Principal SO(3) logarithm with stable identity and near-pi branches.

    The logarithm is necessarily discontinuous at pi.  ``reject_near_pi`` is
    appropriate for target generation; allowing the branch is useful for
    diagnostics and returns a finite, deterministic axis at exactly pi.
    """
    if rotation.shape[-2:] != (3, 3) or not torch.isfinite(rotation).all():
        raise ValueError('rotation must be finite [..., 3, 3]')
    if pi_tolerance < 0:
        raise ValueError('pi_tolerance must be non-negative')
    flat = rotation.reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    orthogonal_error = (flat.transpose(-1, -2) @ flat - eye).abs().amax((-2, -1))
    tolerance = 2e-5 if rotation.dtype == torch.float32 else 2e-9
    if bool((orthogonal_error > tolerance).any()) or bool((torch.det(flat) <= 0).any()):
        raise ValueError('rotation must be a proper orthogonal matrix')

    vee = torch.stack((flat[:, 2, 1] - flat[:, 1, 2],
                       flat[:, 0, 2] - flat[:, 2, 0],
                       flat[:, 1, 0] - flat[:, 0, 1]), dim=-1) * 0.5
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = ((flat.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    near_pi = math.pi - angle <= pi_tolerance
    if reject_near_pi and bool(near_pi.any()):
        raise ValueError('SO(3) endpoint is too close to the pi logarithm branch cut')

    small = angle <= (1e-4 if rotation.dtype == torch.float32 else 1e-7)
    # theta/sin(theta), with its even Taylor expansion around zero.
    scale = torch.where(small, 1 + angle.square() / 6 + 7 * angle.pow(4) / 360,
                        angle / sine.clamp_min(torch.finfo(rotation.dtype).tiny))
    result = vee * scale.unsqueeze(-1)

    # At pi, the skew part vanishes.  The principal eigenvector of (R+I)/2 is
    # the axis.  Orient it by the skew part when possible, otherwise choose a
    # deterministic representative of the unavoidable +/- ambiguity.
    use_axis = math.pi - angle < (1e-3 if rotation.dtype == torch.float32 else 1e-6)
    if bool(use_axis.any()):
        symmetric = (flat[use_axis] + eye) * 0.5
        _, vectors = torch.linalg.eigh((symmetric + symmetric.transpose(-1, -2)) * 0.5)
        axis = vectors[..., -1]
        local_vee = vee[use_axis]
        orientation = (axis * local_vee).sum(-1)
        max_index = axis.abs().argmax(-1, keepdim=True)
        canonical = axis.gather(-1, max_index).squeeze(-1)
        sign = torch.where(orientation.abs() > 1e-10, orientation, canonical).sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        result[use_axis] = axis * sign.unsqueeze(-1) * angle[use_axis].unsqueeze(-1)
    return result.reshape((*rotation.shape[:-2], 3))


def _cap_vector(vector: Tensor, cap: Optional[float]) -> Tuple[Tensor, Tensor]:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if cap is None:
        return vector, torch.zeros_like(norm, dtype=torch.bool).squeeze(-1)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('correction_cap values must be finite and positive')
    factor = torch.clamp(vector.new_tensor(cap) / norm.clamp_min(1e-12), max=1)
    return vector * factor, (norm.squeeze(-1) > cap)


def joint_rigid_endpoint_targets(
        ligand: Tensor, poi: Tensor, body: Tensor,
        native_ligand: Tensor, native_poi: Tensor, time: Tensor,
        *, teacher_targets: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
        min_time: float = 0.0, low_time: str = 'reject', blend: float = 1.0,
        correction_cap: Optional[Union[float, Mapping[str, float]]] = None,
        reject_near_pi: bool = True, pi_tolerance: float = 1e-5,
        exact_fit_tolerance: Optional[float] = 1e-6,
        body_tolerance: Optional[float] = 1e-6) -> Tuple[RigidTarget, Dict[str, Tensor]]:
    """Construct bounded no-torsion joint endpoint velocities and diagnostics.

    Exact mode uses ``min_time=0, blend=1, correction_cap=None`` and no teacher;
    a reverse step with ``dt=time`` then reaches both native point clouds when
    their internal shapes match.  For training experiments, ``min_time`` bounds
    the 1/t amplification (``low_time='clamp'``), while ``blend`` and
    ``correction_cap`` bound the change relative to supplied teacher targets.
    Those controls intentionally remove the one-step exactness guarantee.
    """
    ligand = _validate_points('ligand', ligand)
    poi = _validate_points('poi', poi)
    native_ligand = _validate_points('native_ligand', native_ligand)
    native_poi = _validate_points('native_poi', native_poi)
    if not (ligand.shape[0] == poi.shape[0] == native_ligand.shape[0] == native_poi.shape[0]):
        raise ValueError('all coordinate batches must have the same size')
    batch = ligand.shape[0]
    if body.shape != (batch, 3, 3) or not torch.isfinite(body).all():
        raise ValueError('body must be finite [B, 3, 3]')
    if time.ndim == 0:
        time = time.expand(batch)
    if time.shape != (batch,) or not torch.isfinite(time).all() or bool((time <= 0).any()):
        raise ValueError('time must be finite positive [B]')
    if min_time < 0 or not math.isfinite(min_time):
        raise ValueError('min_time must be finite and non-negative')
    if low_time not in ('reject', 'clamp'):
        raise ValueError("low_time must be 'reject' or 'clamp'")
    low_mask = time < min_time
    if low_time == 'reject' and bool(low_mask.any()):
        raise ValueError('time is below the configured endpoint minimum')
    effective_time = time.clamp_min(min_time) if low_time == 'clamp' else time
    if not 0 < blend <= 1 or not math.isfinite(blend):
        raise ValueError('blend must be finite in (0, 1]')
    if teacher_targets is None and (blend != 1 or correction_cap is not None):
        raise ValueError('blend/correction_cap require teacher_targets')

    ra, ta, ligand_fit_rmsd = proper_rigid_fit(ligand, native_ligand)
    rs, ts, poi_fit_rmsd = proper_rigid_fit(poi, native_poi)
    rs_inv, ts_inv = invert_rigid(rs, ts)
    rb, tb = compose_rigid(rs_inv, ts_inv, ra, ta)
    log_a = stable_so3_log(ra, reject_near_pi=reject_near_pi, pi_tolerance=pi_tolerance)
    log_b = stable_so3_log(rb, reject_near_pi=reject_near_pi, pi_tolerance=pi_tolerance)
    center = ligand.mean(dim=-2)

    oracle: RigidTarget = {'torsion': ligand.new_empty((batch, 0))}  # type: ignore[assignment]
    for branch, rotation, translation, log in (
            ('e3_ligand', ra, ta, log_a), ('poi_ligand', rb, tb, log_b)):
        destination = (rotation @ center.unsqueeze(-1)).squeeze(-1) + translation
        oracle[branch] = {
            'tr': (center - destination) / effective_time.unsqueeze(-1),
            'rot': -log / effective_time.unsqueeze(-1),
        }

    targets: RigidTarget = {'torsion': oracle['torsion']}  # type: ignore[dict-item]
    correction_norms, capped = {}, {}
    caps = correction_cap if isinstance(correction_cap, Mapping) else {}
    for branch in ('e3_ligand', 'poi_ligand'):
        targets[branch] = {}
        for component in ('tr', 'rot'):
            value = oracle[branch][component]
            if teacher_targets is not None:
                base = teacher_targets[branch][component]
                if base.shape != value.shape or not torch.isfinite(base).all():
                    raise ValueError('teacher rigid targets must be finite [B, 3]')
                correction = value - base
                cap = caps.get(component) if isinstance(correction_cap, Mapping) else correction_cap
                correction, was_capped = _cap_vector(correction, cap)
                correction_norms[f'{branch}_{component}'] = torch.linalg.vector_norm(
                    value - base, dim=-1)
                capped[f'{branch}_{component}'] = was_capped
                value = base + blend * correction
            targets[branch][component] = value.detach()

    body_endpoint_error = (rs @ body - torch.eye(
        3, dtype=body.dtype, device=body.device)).abs().amax((-2, -1))
    if exact_fit_tolerance is not None and bool(((ligand_fit_rmsd > exact_fit_tolerance)
                                                 | (poi_fit_rmsd > exact_fit_tolerance)).any()):
        raise ValueError('no-torsion exact mode requires rigid-equivalent native coordinates')
    if body_tolerance is not None and bool((body_endpoint_error > body_tolerance).any()):
        raise ValueError('body frame is inconsistent with current POI and its native template')
    angles = torch.stack((torch.linalg.vector_norm(log_a, dim=-1),
                          torch.linalg.vector_norm(log_b, dim=-1)), dim=-1)
    diagnostics = {
        'effective_time': effective_time.detach(),
        'low_time_clamped': low_mask.detach(),
        'ligand_fit_rmsd': ligand_fit_rmsd.detach(),
        'poi_fit_rmsd': poi_fit_rmsd.detach(),
        'body_endpoint_error': body_endpoint_error.detach(),
        'rotation_angles': angles.detach(),
        'near_pi': (math.pi - angles <= pi_tolerance).detach(),
        'bounded_target': (low_mask | torch.full_like(
            low_mask, teacher_targets is not None)).detach(),
        'transform_a_rotation': ra.detach(),
        'transform_a_translation': ta.detach(),
        'transform_b_rotation': rb.detach(),
        'transform_b_translation': tb.detach(),
        'poi_to_native_rotation': rs.detach(),
        'poi_to_native_translation': ts.detach(),
    }
    diagnostics.update({f'correction_norm_{key}': value.detach()
                        for key, value in correction_norms.items()})
    diagnostics.update({f'correction_capped_{key}': value.detach()
                        for key, value in capped.items()})
    return targets, diagnostics
