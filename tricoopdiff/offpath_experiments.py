"""Experimental off-path targets kept separate from production trainers.

The POI expert emits row-vector labels in the current global frame.  If a
teacher state has body ``B_teacher``, its body-local components are
``v_teacher @ B_teacher``.  The same components at a generated rollout body
``B_rollout`` are therefore

    v_rollout = v_teacher @ B_teacher @ B_rollout.T

This experiment transports only POI translation and rotation labels.  The
teacher E3 and torsion labels are retained.  In particular, the teacher
rotation includes the analytic torsion-gauge spin from ``SDPIFlow.path``;
rotating that label is a controlled target convention, not a derivation of the
full off-path tangent when ligand torsions changed during rollout.
"""

import numpy as np
import torch

from .consensus_experiment import ExperimentalConsensusTrainer
from .sdpi_flow import _horizontal_torsion_velocity


OFFPATH_CONTRACT_MODE = 'poi_body_frame_transported_coupling_v1'
TORSION_GAUGE_CONTRACT = 'teacher_gauge_rotated_without_offpath_rederivation_v1'
GAUGE_CORRECT_CONTRACT_MODE = 'poi_body_frame_transported_gauge_correct_v2'
GAUGE_CORRECT_TORSION_CONTRACT = 'remove_teacher_recompute_rollout_common_spin_v1'


def _validate_body(name, body):
    if not torch.is_tensor(body) or body.ndim != 3 or body.shape[-2:] != (3, 3):
        raise ValueError(f'{name} must have shape [B, 3, 3]')
    if not body.is_floating_point() or not torch.isfinite(body).all():
        raise ValueError(f'{name} must be finite floating point')
    identity = torch.eye(3, dtype=body.dtype, device=body.device).expand_as(body)
    low_precision = body.dtype in (torch.float16, torch.bfloat16, torch.float32)
    tolerance = 5e-5 if low_precision else 1e-10
    if not torch.allclose(body @ body.transpose(-1, -2), identity,
                          atol=tolerance, rtol=tolerance):
        raise ValueError(f'{name} must be orthogonal')
    if not bool((torch.linalg.det(body) > 0).all()):
        raise ValueError(f'{name} must be a proper rotation')


def transport_poi_body_frame_targets(targets, teacher_body, rollout_body):
    """Copy targets and express POI rigid labels at ``rollout_body``.

    Bodies use the ``SDPIFlow`` row-vector convention.  E3 and torsion values
    are deliberately returned unchanged, including their tensor identities.
    """
    _validate_body('teacher_body', teacher_body)
    _validate_body('rollout_body', rollout_body)
    if teacher_body.shape != rollout_body.shape:
        raise ValueError('teacher and rollout body shapes must match')
    poi = targets.get('poi_ligand')
    if not isinstance(poi, dict):
        raise ValueError('targets must contain poi_ligand rigid labels')
    transported = {}
    mapping = teacher_body @ rollout_body.transpose(-1, -2)
    for component in ('tr', 'rot'):
        value = poi.get(component)
        if (not torch.is_tensor(value) or value.shape != (*teacher_body.shape[:-2], 3)
                or value.dtype != teacher_body.dtype or value.device != teacher_body.device
                or not torch.isfinite(value).all()):
            raise ValueError(f'poi_ligand.{component} must be finite [B, 3]')
        transported[component] = torch.einsum('...i,...ij->...j', value, mapping)
    return {
        **targets,
        'poi_ligand': {**poi, **transported},
    }


def _state_torsion_gauge(coords, coupling, torsion):
    """Return the solver's common world-frame spin for one ligand state."""
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[-1] != 3:
        raise ValueError('ligand coordinates must have shape [1, N, 3]')
    if torsion.ndim != 2 or torsion.shape[0] != 1:
        raise ValueError('torsion target must have shape [1, K]')
    angles = torsion[0].detach().cpu().double().numpy()
    bonds, masks = coupling['bonds'], coupling['masks']
    if len(bonds) != len(angles) or len(masks) != len(angles):
        raise ValueError('torsion target and ligand graph sizes differ')
    _, spin = _horizontal_torsion_velocity(
        coords[0].detach().cpu().double().numpy(), bonds, masks, angles)
    return torch.as_tensor(np.array(spin, copy=True), dtype=coords.dtype,
                           device=coords.device).reshape(1, 3)


def transport_targets_with_recomputed_gauge(targets, teacher_body, rollout_body,
                                             teacher_ligand, rollout_ligand, coupling):
    """Transport rigid labels and recompute the torsion chart's common spin."""
    result = transport_poi_body_frame_targets(targets, teacher_body, rollout_body)
    teacher_gauge = _state_torsion_gauge(teacher_ligand, coupling, targets['torsion'])
    rollout_gauge = _state_torsion_gauge(rollout_ligand, coupling, targets['torsion'])
    e3_rot = targets['e3_ligand']['rot'] - teacher_gauge + rollout_gauge
    poi_global = torch.einsum('...i,...ij->...j',
                              targets['poi_ligand']['rot'], teacher_body)
    poi_local = torch.einsum('...i,...ij->...j',
                             poi_global - teacher_gauge + rollout_gauge,
                             rollout_body.transpose(-1, -2))
    return {
        **result,
        'e3_ligand': {**targets['e3_ligand'], 'rot': e3_rot},
        'poi_ligand': {**result['poi_ligand'], 'rot': poi_local},
    }


class BodyFrameTransportRolloutTrainer(ExperimentalConsensusTrainer):
    """Train on generated states with transported POI coupling labels.

    Generated-state identity is local control flow: the label mapping is
    applied in the same branch that generated the rollout.  The cumulative
    counter is observability only and cannot affect target selection.
    """

    def __init__(self, *args, rollout_probability=0.5, rollout_steps=1,
                 rollout_min_time=0.0, contract_mode=OFFPATH_CONTRACT_MODE,
                 **kwargs):
        if not 0 <= rollout_probability <= 1:
            raise ValueError('rollout_probability must be in [0, 1]')
        if (not isinstance(rollout_steps, int)
                or isinstance(rollout_steps, bool) or rollout_steps < 1):
            raise ValueError('rollout_steps must be a positive integer')
        if not 0 <= rollout_min_time < 1:
            raise ValueError('rollout_min_time must be in [0, 1)')
        if contract_mode != OFFPATH_CONTRACT_MODE:
            raise ValueError(f'unsupported off-path contract mode: {contract_mode!r}')
        self.rollout_probability = float(rollout_probability)
        self.rollout_steps = rollout_steps
        self.rollout_min_time = float(rollout_min_time)
        self.offpath_contract_mode = contract_mode
        self.rollout_augmented_states = 0
        super().__init__(*args, **kwargs)

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        teacher = self.diffusion.path(ligand_data, poi_data['coords'], coupling, t)
        should_roll_out = (
            self.rollout_probability > 0
            and float(t.item()) < 1
            and float(t.item()) >= self.rollout_min_time
            and float(torch.rand((), device=t.device)) < self.rollout_probability
        )
        if not should_roll_out:
            return teacher

        start = torch.clamp(t + dt * self.rollout_steps, max=1.0)
        step_dt = (start - t) / self.rollout_steps
        if not bool((step_dt > 0).all()):
            return teacher

        ligand, poi, body, _ = self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, start)
        with torch.no_grad():
            current = start
            for _ in range(self.rollout_steps):
                output = self._forward_state(
                    e3_data, poi_data, ligand_data,
                    ligand, poi, body, current, step_dt)
                normalized = self.diffusion.normalized_predictions(output, current)
                velocities = self.diffusion.velocities(
                    normalized, e3_data['coords'], poi_data['coords'])
                ligand, poi, body = self.diffusion.step(
                    ligand, poi, body, velocities, step_dt, ligand_data)
                current = current - step_dt

        targets = self._map_targets(teacher, ligand, body, coupling)
        self.rollout_augmented_states += 1
        return ligand.detach(), poi.detach(), body.detach(), targets

    def _map_targets(self, teacher, rollout_ligand, rollout_body, coupling):
        return transport_poi_body_frame_targets(teacher[3], teacher[2], rollout_body)

    def runtime_contract(self):
        return {
            **super().runtime_contract(),
            'rollout_probability': self.rollout_probability,
            'rollout_steps': self.rollout_steps,
            'rollout_min_time': self.rollout_min_time,
            'offpath_target': self.offpath_contract_mode,
            'offpath_contract_mode': self.offpath_contract_mode,
            'offpath_torsion_gauge': TORSION_GAUGE_CONTRACT,
        }


class GaugeCorrectBodyFrameTransportRolloutTrainer(BodyFrameTransportRolloutTrainer):
    """Transport off-path rigid labels with a solver-consistent torsion gauge."""

    def __init__(self, *args, contract_mode=GAUGE_CORRECT_CONTRACT_MODE, **kwargs):
        # The parent validates its legacy contract, so initialize with that value
        # and then install this class's distinct checkpoint identity.
        if contract_mode != GAUGE_CORRECT_CONTRACT_MODE:
            raise ValueError(f'unsupported off-path contract mode: {contract_mode!r}')
        super().__init__(*args, contract_mode=OFFPATH_CONTRACT_MODE, **kwargs)
        self.offpath_contract_mode = contract_mode

    def _map_targets(self, teacher, rollout_ligand, rollout_body, coupling):
        return transport_targets_with_recomputed_gauge(
            teacher[3], teacher[2], rollout_body, teacher[0], rollout_ligand, coupling)

    def runtime_contract(self):
        result = super().runtime_contract()
        result['offpath_target'] = self.offpath_contract_mode
        result['offpath_contract_mode'] = self.offpath_contract_mode
        result['offpath_torsion_gauge'] = GAUGE_CORRECT_TORSION_CONTRACT
        return result
