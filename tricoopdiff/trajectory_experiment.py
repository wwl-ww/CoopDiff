"""Differentiable short-horizon trajectory objective for experiments only."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math

import torch

from .consensus_experiment import ExperimentalConsensusTrainer
from .joint_endpoint_experiment import (
    _proper_rigid_fit,
    joint_rigid_endpoint_targets,
)
from .rigid import apply_rigid, assemble_interface_pair
from .sdpi_flow import _rotation_exp


TRAJECTORY_CONTRACT_MODE = 'differentiable_sdpi_trajectory_v1'
STAGED_POI_CONTRACT_MODE = 'differentiable_sdpi_staged_poi_v2'
DISTANCE_STAGED_POI_CONTRACT_MODE = 'differentiable_sdpi_staged_poi_distance_v2'
PRIOR_REPLAY_CONTRACT_MODE = 'differentiable_sdpi_prior_replay_v1'
PRIOR_REPLAY_EVAL_CONTRACT_MODE = 'differentiable_sdpi_prior_replay_eval_v2'
PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_v1')
PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_replace_v2')
PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_window_replace_v3')
PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_soft_time_v4')
PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_direction_v4')
PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE = (
    'differentiable_sdpi_prior_replay_joint_translation_agreement_v4')


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


def prior_replay_schedule(time, *, inference_steps, max_steps):
    """Return equal reverse steps from the prior endpoint to ``time``."""
    if time.shape != (1,) or not torch.isfinite(time).all():
        raise ValueError('time must be one finite value')
    if not 0 < float(time.item()) <= 1:
        raise ValueError('prior replay time must be in (0, 1]')
    if (not isinstance(inference_steps, int) or isinstance(inference_steps, bool)
            or inference_steps < 1):
        raise ValueError('inference_steps must be a positive integer')
    if (not isinstance(max_steps, int) or isinstance(max_steps, bool)
            or max_steps < 1):
        raise ValueError('max_steps must be a positive integer')
    if float(time.item()) == 1:
        return 0, time.new_zeros(1)
    nominal = 1 / inference_steps
    count = min(max_steps, max(1, math.ceil((1 - float(time.item())) / nominal)))
    step = (time.new_ones(1) - time) / count
    return count, step


class PriorReplayTrajectoryTrainer(TrajectoryConsistencyTrainer):
    """Constrain a short rollout after detached replay from the prior."""

    def __init__(self, *args, replay_max_steps=15,
                 contract_mode=PRIOR_REPLAY_CONTRACT_MODE, **kwargs):
        if contract_mode != PRIOR_REPLAY_CONTRACT_MODE:
            raise ValueError(f'unsupported prior replay contract: {contract_mode!r}')
        if (not isinstance(replay_max_steps, int)
                or isinstance(replay_max_steps, bool) or replay_max_steps < 1):
            raise ValueError('replay_max_steps must be a positive integer')
        self.replay_max_steps = replay_max_steps
        self.prior_replay_steps = 0
        super().__init__(*args, contract_mode=TRAJECTORY_CONTRACT_MODE, **kwargs)
        self.trajectory_contract_mode = contract_mode

    def _replay_context(self):
        return nullcontext()

    def _trajectory_loss(self):
        e3_data, poi_data, ligand_data, coupling, time, dt, _ = self._trajectory_context
        if self.trajectory_weight == 0:
            return time.new_zeros(())
        required_time = self.trajectory_steps / self.diffusion.inference_steps
        if float(time.item()) < max(self.trajectory_min_time, required_time):
            return time.new_zeros(())

        replay_count, replay_dt = prior_replay_schedule(
            time, inference_steps=self.diffusion.inference_steps,
            max_steps=self.replay_max_steps)
        current = time.new_ones(1)
        ligand, poi, body, _ = self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, current)
        with self._replay_context(), torch.no_grad():
            for _ in range(replay_count):
                output = self._forward_state(
                    e3_data, poi_data, ligand_data,
                    ligand, poi, body, current, replay_dt)
                normalized = self.diffusion.normalized_predictions(output, current)
                velocities = self.diffusion.velocities(
                    normalized, e3_data['coords'], poi_data['coords'])
                ligand, poi, body = self.diffusion.step(
                    ligand, poi, body, velocities, replay_dt, ligand_data)
                current = current - replay_dt

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
        self.prior_replay_steps += replay_count
        return torch.stack(losses).mean()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_replay_start_time': 1.0,
            'trajectory_replay_max_steps': self.replay_max_steps,
            'trajectory_replay_gradients': False,
        })
        return contract


class DeterministicPriorReplayTrajectoryTrainer(PriorReplayTrajectoryTrainer):
    """Replay production-like states with dropout disabled."""

    def __init__(self, *args, contract_mode=PRIOR_REPLAY_EVAL_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_EVAL_CONTRACT_MODE:
            raise ValueError(
                f'unsupported deterministic prior replay contract: {contract_mode!r}')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_CONTRACT_MODE, **kwargs)
        self.trajectory_contract_mode = contract_mode

    @contextmanager
    def _replay_context(self):
        training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(training)

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract['trajectory_replay_model_mode'] = 'eval'
        return contract


def joint_endpoint_translation_loss(velocities, targets, scales):
    """Scale-normalized direct loss for both coupled translation branches."""
    if len(scales) != 2:
        raise ValueError('scales must contain E3 and POI receptor scales')
    losses = []
    for branch, scale in zip(('e3_ligand', 'poi_ligand'), scales):
        prediction = velocities[branch]['tr']
        target = targets[branch]['tr']
        if prediction.shape != target.shape:
            raise ValueError('translation prediction and target shapes differ')
        if (scale.numel() != 1 or not torch.isfinite(scale).all()
                or not bool((scale > 0).all())):
            raise ValueError('translation scales must be positive scalar tensors')
        losses.append(((prediction - target) / scale).square().mean())
    return torch.stack(losses).sum()


class PriorReplayJointTranslationTrainer(PriorReplayTrajectoryTrainer):
    """Supervise coupled translation velocities on detached prior-replay states.

    Native coordinates are read only to construct training targets.  Every
    rollout transition is detached, so the auxiliary loss directly supervises
    the two translation outputs without an indirect rotation/torsion gradient.
    """

    def __init__(self, *args, trajectory_min_time=0.3,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE:
            raise ValueError(
                f'unsupported prior replay translation contract: {contract_mode!r}')
        super().__init__(
            *args, trajectory_min_time=trajectory_min_time,
            contract_mode=PRIOR_REPLAY_CONTRACT_MODE, **kwargs)
        self.trajectory_contract_mode = contract_mode

    def _translation_loss(self, velocities, endpoint_targets, scales,
                          analytic_targets):
        """Return the off-path translation objective for one replay state."""
        del analytic_targets
        return joint_endpoint_translation_loss(
            velocities, endpoint_targets, scales)

    @staticmethod
    def _analytic_translation_targets(ligand_data, coupling, body):
        """Map the sampled coupling translations into the current global frames."""
        native = ligand_data['coords'][0]
        centers = torch.as_tensor(
            coupling['centers'], device=native.device, dtype=native.dtype)
        translations = centers - native.mean(dim=0)
        return {
            'e3_ligand': {'tr': translations[0:1]},
            'poi_ligand': {'tr': translations[1:2] @ body[0].T},
        }

    def _trajectory_loss(self):
        e3_data, poi_data, ligand_data, coupling, time, dt, _ = self._trajectory_context
        if self.trajectory_weight == 0:
            return time.new_zeros(())
        if float(time.item()) < self.trajectory_min_time:
            return time.new_zeros(())

        replay_count, replay_dt = prior_replay_schedule(
            time, inference_steps=self.diffusion.inference_steps,
            max_steps=self.replay_max_steps)
        current = time.new_ones(1)
        ligand, poi, body, _ = self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, current)
        with torch.no_grad():
            for _ in range(replay_count):
                output = self._forward_state(
                    e3_data, poi_data, ligand_data,
                    ligand, poi, body, current, replay_dt)
                normalized = self.diffusion.normalized_predictions(output, current)
                velocities = self.diffusion.velocities(
                    normalized, e3_data['coords'], poi_data['coords'])
                ligand, poi, body = self.diffusion.step(
                    ligand, poi, body, velocities, replay_dt, ligand_data)
                current = current - replay_dt

        scales = self.diffusion.scales(e3_data['coords'], poi_data['coords'])
        losses = []
        current = time
        for _ in range(self.trajectory_steps):
            if float(current.item()) < self.trajectory_min_time:
                break
            output = self._forward_state(
                e3_data, poi_data, ligand_data, ligand, poi, body, current, dt)
            normalized = self.diffusion.normalized_predictions(output, current)
            velocities = self.diffusion.velocities(
                normalized, e3_data['coords'], poi_data['coords'])
            with torch.no_grad():
                targets, _ = joint_rigid_endpoint_targets(
                    ligand, poi, body, ligand_data['coords'], poi_data['coords'],
                    current, reject_near_pi=False, exact_fit_tolerance=None,
                    body_tolerance=None)
                analytic_targets = self._analytic_translation_targets(
                    ligand_data, coupling, body)
            losses.append(self._translation_loss(
                velocities, targets, scales, analytic_targets))
            with torch.no_grad():
                detached = {
                    branch: {
                        component: value.detach()
                        for component, value in velocities[branch].items()
                    }
                    for branch in ('e3_ligand', 'poi_ligand')
                }
                detached['torsion'] = velocities['torsion'].detach()
                ligand, poi, body = self.diffusion.step(
                    ligand, poi, body, detached, dt, ligand_data)
            current = current - dt
        if not losses:
            return time.new_zeros(())
        self.trajectory_augmented_states += 1
        self.prior_replay_steps += replay_count
        return torch.stack(losses).mean()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_auxiliary_target': 'native_joint_rigid_endpoint_translation',
            'trajectory_auxiliary_branches': ['e3_ligand.tr', 'poi_ligand.tr'],
            'trajectory_auxiliary_state_gradients': False,
            'trajectory_auxiliary_native_reading': 'training_only',
            'trajectory_rotation_torsion_target': 'analytic_teacher_only',
        })
        return contract


class PriorReplayJointTranslationReplaceTrainer(PriorReplayJointTranslationTrainer):
    """Replace on-path translation loss when an off-path target is available."""

    def __init__(self, *args,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE:
            raise ValueError(
                f'unsupported translation replacement contract: {contract_mode!r}')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_TRANSLATION_CONTRACT_MODE,
            **kwargs)
        self.trajectory_contract_mode = contract_mode
        self._translation_auxiliary_applied = False

    def _trajectory_loss(self):
        self._translation_auxiliary_applied = False
        loss = super()._trajectory_loss()
        self._translation_auxiliary_applied = bool(loss.requires_grad)
        return loss

    def compute_loss(self, *args, diagnostics=False, **kwargs):
        self._trajectory_context = None
        result = ExperimentalConsensusTrainer.compute_loss(
            self, *args, diagnostics=True, **kwargs)
        auxiliary = self._trajectory_loss()
        total = result['loss_total']
        mix = (self._translation_replacement_mix()
               if self._translation_auxiliary_applied else auxiliary.new_zeros(()))
        if self._translation_auxiliary_applied:
            teacher = (result['components']['loss_e3l_tr']
                       + result['components']['loss_pol_tr'])
            total = total - mix * teacher + mix * self.trajectory_weight * auxiliary
        result['loss_total'] = total
        if diagnostics:
            result['components']['loss_trajectory'] = auxiliary
            result['components']['translation_teacher_replaced'] = auxiliary.new_tensor(
                float(self._translation_auxiliary_applied))
            result['components']['translation_replacement_mix'] = mix
            return result
        return {
            'loss_total': total,
            'loss_score': total.item(),
            **{name: value.item() for name, value in result['components'].items()},
            'loss_trajectory': auxiliary.item(),
            'translation_teacher_replaced': self._translation_auxiliary_applied,
            'translation_replacement_mix': float(mix.item()),
        }

    def _translation_replacement_mix(self):
        """Weight replacing the analytic translation loss with the off-path loss."""
        return self._trajectory_context[4].new_ones(())

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_translation_teacher_policy': (
                'replace_when_prior_replay_target_available'),
        })
        return contract


class PriorReplayJointTranslationWindowReplaceTrainer(
        PriorReplayJointTranslationReplaceTrainer):
    """Keep the analytic teacher where endpoint gradients conflict near the prior."""

    def __init__(self, *args, trajectory_max_time=0.65,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_WINDOW_REPLACE_CONTRACT_MODE:
            raise ValueError(
                f'unsupported windowed translation contract: {contract_mode!r}')
        trajectory_min_time = float(kwargs.get('trajectory_min_time', 0.3))
        if not trajectory_min_time < trajectory_max_time <= 1:
            raise ValueError('translation window must satisfy min < max <= 1')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
            **kwargs)
        self.trajectory_max_time = float(trajectory_max_time)
        self.trajectory_contract_mode = contract_mode

    def _trajectory_loss(self):
        if self._trajectory_context is not None:
            time = self._trajectory_context[4]
            if float(time.item()) > self.trajectory_max_time:
                self._translation_auxiliary_applied = False
                return time.new_zeros(())
        return super()._trajectory_loss()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_max_time': self.trajectory_max_time,
            'trajectory_translation_teacher_policy': (
                'replace_only_inside_prior_replay_time_window'),
        })
        return contract


class PriorReplayJointTranslationSoftTimeTrainer(
        PriorReplayJointTranslationReplaceTrainer):
    """Continuously introduce endpoint supervision as states become identifiable."""

    def __init__(self, *args, endpoint_zero_time=0.8, endpoint_full_time=0.3,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_SOFT_TIME_CONTRACT_MODE:
            raise ValueError(f'unsupported soft-time contract: {contract_mode!r}')
        if not 0 < endpoint_full_time < endpoint_zero_time <= 1:
            raise ValueError('soft-time bounds must satisfy 0 < full < zero <= 1')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
            **kwargs)
        self.endpoint_zero_time = float(endpoint_zero_time)
        self.endpoint_full_time = float(endpoint_full_time)
        self.trajectory_contract_mode = contract_mode

    def _trajectory_loss(self):
        if (self._trajectory_context is not None
                and float(self._trajectory_context[4].item()) >= self.endpoint_zero_time):
            self._translation_auxiliary_applied = False
            return self._trajectory_context[4].new_zeros(())
        return super()._trajectory_loss()

    def _translation_replacement_mix(self):
        time = self._trajectory_context[4]
        return ((self.endpoint_zero_time - time)
                / (self.endpoint_zero_time - self.endpoint_full_time)).clamp(0, 1).squeeze()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_translation_teacher_policy': 'linear_time_mixture',
            'trajectory_endpoint_zero_time': self.endpoint_zero_time,
            'trajectory_endpoint_full_time': self.endpoint_full_time,
        })
        return contract


class PriorReplayJointTranslationDirectionTrainer(
        PriorReplayJointTranslationReplaceTrainer):
    """Use endpoint direction while retaining analytic-teacher speed magnitude."""

    def __init__(self, *args,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_DIRECTION_CONTRACT_MODE:
            raise ValueError(f'unsupported direction contract: {contract_mode!r}')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
            **kwargs)
        self.trajectory_contract_mode = contract_mode

    def _translation_loss(self, velocities, endpoint_targets, scales,
                          analytic_targets):
        losses = []
        for branch, scale in zip(('e3_ligand', 'poi_ligand'), scales):
            endpoint = endpoint_targets[branch]['tr']
            magnitude = torch.linalg.vector_norm(
                analytic_targets[branch]['tr'], dim=-1, keepdim=True)
            target = torch.nn.functional.normalize(endpoint, dim=-1) * magnitude
            losses.append(((velocities[branch]['tr'] - target) / scale).square().mean())
        return torch.stack(losses).sum()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_translation_target_direction': 'native_endpoint',
            'trajectory_translation_target_magnitude': 'analytic_coupling_teacher',
        })
        return contract


class PriorReplayJointTranslationAgreementTrainer(
        PriorReplayJointTranslationReplaceTrainer):
    """Trust endpoint supervision only to the extent it agrees with the teacher."""

    def __init__(self, *args,
                 contract_mode=PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE,
                 **kwargs):
        if contract_mode != PRIOR_REPLAY_TRANSLATION_AGREEMENT_CONTRACT_MODE:
            raise ValueError(f'unsupported agreement contract: {contract_mode!r}')
        super().__init__(
            *args, contract_mode=PRIOR_REPLAY_TRANSLATION_REPLACE_CONTRACT_MODE,
            **kwargs)
        self.trajectory_contract_mode = contract_mode

    def _translation_loss(self, velocities, endpoint_targets, scales,
                          analytic_targets):
        losses = []
        for branch, scale in zip(('e3_ligand', 'poi_ligand'), scales):
            endpoint = endpoint_targets[branch]['tr']
            analytic = analytic_targets[branch]['tr']
            agreement = torch.nn.functional.cosine_similarity(
                endpoint, analytic, dim=-1).clamp(0, 1).detach()[:, None]
            target = agreement * endpoint + (1 - agreement) * analytic
            losses.append(((velocities[branch]['tr'] - target) / scale).square().mean())
        return torch.stack(losses).sum()

    def runtime_contract(self):
        contract = super().runtime_contract()
        contract.update({
            'trajectory_objective': self.trajectory_contract_mode,
            'trajectory_translation_teacher_policy': 'positive_cosine_agreement_gate',
        })
        return contract


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
