"""Differentiable short-horizon trajectory objective for experiments only."""

from __future__ import annotations

import math

import torch

from .consensus_experiment import ExperimentalConsensusTrainer
from .joint_endpoint_experiment import _proper_rigid_fit
from .rigid import apply_rigid, assemble_interface_pair
from .sdpi_flow import _rotation_exp


TRAJECTORY_CONTRACT_MODE = 'differentiable_sdpi_trajectory_v1'


def _torsion_masks(ligand_data, device):
    masks = ligand_data['mask_rotate']
    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    return torch.as_tensor(masks, dtype=torch.bool, device=device)


def differentiable_sdpi_step(ligand, poi, body, velocities, dt, ligand_data):
    """Match ``SDPIFlow.step`` while preserving autograd through torsions."""
    if ligand.ndim != 3 or ligand.shape[0] != 1 or ligand.shape[-1] != 3:
        raise ValueError('ligand must have shape [1, N, 3]')
    if poi.ndim != 3 or poi.shape[0] != 1 or poi.shape[-1] != 3:
        raise ValueError('poi must have shape [1, N, 3]')
    if body.shape != (1, 3, 3) or dt.shape != (1,) or not bool((dt > 0).all()):
        raise ValueError('body/dt shapes must be [1,3,3]/[1] with positive dt')
    old_center = ligand.mean(1)
    transforms = []
    for branch in ('e3_ligand', 'poi_ligand'):
        velocity = velocities[branch]
        rotation = _rotation_exp(-dt[:, None] * velocity['rot'])
        center = old_center - dt[:, None] * velocity['tr']
        offset = center - (rotation @ old_center[..., None]).squeeze(-1)
        transforms.append((rotation, offset))
    (ra, ta), (rb, tb) = transforms
    rigid = apply_rigid(ligand, ra, ta)
    poi, increment = assemble_interface_pair(poi, ra, ta, rb, tb)
    body = increment @ body

    torsion = velocities['torsion']
    bonds = ligand_data['edge_index'].T[ligand_data['edge_mask']].to(
        device=ligand.device, dtype=torch.long)
    masks = _torsion_masks(ligand_data, ligand.device)
    if torsion.shape != (1, len(bonds)) or masks.shape != (len(bonds), ligand.shape[1]):
        raise ValueError('torsion velocities and ligand graph sizes differ')
    raw = rigid
    for index, (u, v) in enumerate(bonds):
        axis = raw[:, u] - raw[:, v]
        norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
        if bool((norm <= 1e-10).any()):
            raise ValueError('degenerate torsion axis')
        angle = -dt * torsion[:, index]
        rotation = _rotation_exp(axis / norm * angle[:, None])
        pivot = raw[:, v:v + 1]
        moved = (raw - pivot) @ rotation.transpose(-1, -2) + pivot
        raw = torch.where(masks[index][None, :, None], moved, raw)
    if len(bonds):
        rotation, translation, _ = _proper_rigid_fit(raw, rigid)
        rigid = apply_rigid(raw, rotation, translation)
    return rigid, poi, body


class TrajectoryConsistencyTrainer(ExperimentalConsensusTrainer):
    """Add a differentiable short rollout loss against the analytic SDPI path."""

    def __init__(self, *args, trajectory_steps=3, trajectory_weight=0.1,
                 trajectory_min_time=0.25,
                 contract_mode=TRAJECTORY_CONTRACT_MODE, **kwargs):
        if contract_mode != TRAJECTORY_CONTRACT_MODE:
            raise ValueError(f'unsupported trajectory contract: {contract_mode!r}')
        if (not isinstance(trajectory_steps, int) or isinstance(trajectory_steps, bool)
                or trajectory_steps < 2):
            raise ValueError('trajectory_steps must be an integer >= 2')
        if not math.isfinite(trajectory_weight) or trajectory_weight <= 0:
            raise ValueError('trajectory_weight must be finite and positive')
        if not 0 < trajectory_min_time < 1:
            raise ValueError('trajectory_min_time must be in (0, 1)')
        self.trajectory_steps = trajectory_steps
        self.trajectory_weight = float(trajectory_weight)
        self.trajectory_min_time = float(trajectory_min_time)
        self.trajectory_contract_mode = contract_mode
        self.trajectory_augmented_states = 0
        self._trajectory_context = None
        super().__init__(*args, **kwargs)

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        state = self.diffusion.path(ligand_data, poi_data['coords'], coupling, t)
        self._trajectory_context = (
            e3_data, poi_data, ligand_data, coupling, t, dt, state[:3])
        return state

    def _trajectory_loss(self):
        e3_data, poi_data, ligand_data, coupling, time, dt, state = self._trajectory_context
        if self.trajectory_weight == 0:
            return time.new_zeros(())
        required_time = self.trajectory_steps / self.diffusion.inference_steps
        if float(time.item()) < max(self.trajectory_min_time, required_time):
            return time.new_zeros(())
        ligand, poi, body = state
        scales = self.diffusion.scales(e3_data['coords'], poi_data['coords'])
        losses = []
        current = time
        for _ in range(self.trajectory_steps):
            output = self._forward_state(
                e3_data, poi_data, ligand_data, ligand, poi, body, current, dt)
            normalized = self.diffusion.normalized_predictions(output, current)
            velocities = self.diffusion.velocities(
                normalized, e3_data['coords'], poi_data['coords'])
            ligand, poi, body = differentiable_sdpi_step(
                ligand, poi, body, velocities, dt, ligand_data)
            current = current - dt
            target_ligand, target_poi, target_body, _ = self.diffusion.path(
                ligand_data, poi_data['coords'], coupling, current)
            losses.append(
                ((ligand - target_ligand) / scales[0]).square().mean()
                + ((poi - target_poi) / scales[1]).square().mean()
                + (body - target_body).square().mean())
        self.trajectory_augmented_states += 1
        return torch.stack(losses).mean()

    def compute_loss(self, *args, diagnostics=False, **kwargs):
        self._trajectory_context = None
        result = super().compute_loss(*args, diagnostics=diagnostics, **kwargs)
        trajectory = self._trajectory_loss()
        result['loss_total'] = (
            result['loss_total'] + self.trajectory_weight * trajectory)
        if diagnostics:
            result['components']['loss_trajectory'] = trajectory
        else:
            result['loss_trajectory'] = trajectory.item()
            result['loss_score'] = result['loss_total'].item()
        return result

    def runtime_contract(self):
        return {
            **super().runtime_contract(),
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_steps': self.trajectory_steps,
            'trajectory_weight': self.trajectory_weight,
            'trajectory_min_time': self.trajectory_min_time,
        }
