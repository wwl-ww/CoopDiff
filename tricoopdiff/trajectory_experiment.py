"""Differentiable short-horizon trajectory objective for experiments only."""

from __future__ import annotations

import math

import torch

from .consensus_experiment import ExperimentalConsensusTrainer
from .joint_endpoint_experiment import _proper_rigid_fit
from .rigid import apply_rigid, assemble_interface_pair
from .sdpi_flow import _rotation_exp


TRAJECTORY_CONTRACT_MODE = 'differentiable_sdpi_trajectory_v1'
STAGED_POI_CONTRACT_MODE = 'differentiable_sdpi_staged_poi_v2'
DISTANCE_STAGED_POI_CONTRACT_MODE = 'differentiable_sdpi_staged_poi_distance_v2'


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


def staged_poi_state_loss(poi, target_poi, native_poi, body, target_body,
                          scale, *, distance_weighted=False,
                          distance_weight_cap=3.0):
    """Joint POI center/orientation loss, optionally emphasizing distant states."""
    if poi.shape != target_poi.shape or poi.shape != native_poi.shape:
        raise ValueError('POI coordinate tensors must have matching shapes')
    if poi.ndim != 3 or poi.shape[0] != 1 or poi.shape[-1] != 3:
        raise ValueError('POI coordinates must have shape [1, N, 3]')
    if body.shape != target_body.shape or body.shape != (1, 3, 3):
        raise ValueError('body tensors must have shape [1, 3, 3]')
    if (scale.numel() != 1 or not torch.isfinite(scale).all()
            or not bool((scale > 0).all())):
        raise ValueError('POI scale must be one positive value')
    if not 1 <= distance_weight_cap or not math.isfinite(distance_weight_cap):
        raise ValueError('distance_weight_cap must be finite and >= 1')
    center_error = (poi.mean(1) - target_poi.mean(1)) / scale
    center_loss = center_error.square().mean()
    orientation_loss = (body - target_body).square().mean()
    weight = poi.new_ones(())
    if distance_weighted:
        normalized_distance = torch.linalg.vector_norm(
            poi.mean(1) - native_poi.mean(1), dim=-1).mean() / scale
        weight = normalized_distance.detach().clamp(1, distance_weight_cap)
    return weight * (center_loss + orientation_loss), {
        'center': center_loss.detach(),
        'orientation': orientation_loss.detach(),
        'distance_weight': weight.detach(),
    }


class StagedPOITrajectoryTrainer(TrajectoryConsistencyTrainer):
    """Constrain coupled POI center/orientation only in a middle/late window."""

    def __init__(self, *args, trajectory_steps=5, trajectory_weight=1.0,
                 trajectory_min_time=0.25, trajectory_max_time=0.65,
                 distance_weight_cap=3.0,
                 contract_mode=STAGED_POI_CONTRACT_MODE, **kwargs):
        allowed = {STAGED_POI_CONTRACT_MODE, DISTANCE_STAGED_POI_CONTRACT_MODE}
        if contract_mode not in allowed:
            raise ValueError(f'unsupported staged POI contract: {contract_mode!r}')
        if (not math.isfinite(trajectory_max_time)
                or not trajectory_min_time < trajectory_max_time < 1):
            raise ValueError('trajectory_max_time must lie between min time and 1')
        if (not math.isfinite(distance_weight_cap)
                or distance_weight_cap < 1):
            raise ValueError('distance_weight_cap must be finite and >= 1')
        self.trajectory_max_time = float(trajectory_max_time)
        self.distance_weight_cap = float(distance_weight_cap)
        self.distance_weighted = contract_mode == DISTANCE_STAGED_POI_CONTRACT_MODE
        super().__init__(
            *args, trajectory_steps=trajectory_steps,
            trajectory_weight=trajectory_weight,
            trajectory_min_time=trajectory_min_time,
            contract_mode=TRAJECTORY_CONTRACT_MODE, **kwargs)
        self.trajectory_contract_mode = contract_mode

    def _trajectory_loss(self):
        e3_data, poi_data, ligand_data, coupling, time, dt, state = self._trajectory_context
        if self.trajectory_weight == 0:
            return time.new_zeros(())
        first_time = float((time - dt).item())
        last_time = float((time - self.trajectory_steps * dt).item())
        if first_time < self.trajectory_min_time or last_time > self.trajectory_max_time:
            return time.new_zeros(())
        ligand, poi, body = state
        poi_scale = self.diffusion.scales(e3_data['coords'], poi_data['coords'])[1]
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
            value = float(current.item())
            if value < self.trajectory_min_time:
                break
            if value <= self.trajectory_max_time:
                _, target_poi, target_body, _ = self.diffusion.path(
                    ligand_data, poi_data['coords'], coupling, current)
                loss, _ = staged_poi_state_loss(
                    poi, target_poi, poi_data['coords'], body, target_body,
                    poi_scale, distance_weighted=self.distance_weighted,
                    distance_weight_cap=self.distance_weight_cap)
                losses.append(loss)
        if not losses:
            return time.new_zeros(())
        self.trajectory_augmented_states += 1
        return torch.stack(losses).mean()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_max_time': self.trajectory_max_time,
            'trajectory_distance_weighted': self.distance_weighted,
            'trajectory_distance_weight_cap': self.distance_weight_cap,
        })
        return contract
