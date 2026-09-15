"""Explicit factorial experiments; checkpoints are separate from production v2."""
import os

import torch

from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .consensus import ConsensusCoopDiff
from .consensus_train import ConsensusTrainer

EXPERIMENT_ARCHITECTURE = 'residual_consensus_factorial_v1'


def clone_state(state):
    return {name: value.detach().clone() for name, value in state.items()}


class ExperimentalConsensus(ConsensusCoopDiff):
    """Factorial consensus model with an immutable pretrained expert."""
    architecture = EXPERIMENT_ARCHITECTURE

    def __init__(self, expert, trifuse):
        super().__init__(expert, trifuse)


class ExperimentalConsensusTrainer(ConsensusTrainer):
    """Train and serialize a factorial frozen-expert consensus model."""
    def __init__(self, *args, experiment_contract, **kwargs):
        self.experiment_contract = dict(experiment_contract)
        super().__init__(*args, **kwargs)

    def runtime_contract(self):
        return dict(activation_clamp=self.activation_clamp,
                    activation_clamp_mode=self.activation_clamp_mode,
                    residual_weight=self.residual_weight, ema_decay=self.ema_decay,
                    optimizer_groups=self.optimizer_config)

    def save_checkpoint(self, path, epoch=0, best_val_loss=float('inf')):
        payload = dict(
            checkpoint_version=1, architecture=EXPERIMENT_ARCHITECTURE,
            experiment_contract=self.experiment_contract,
            runtime_config=self.runtime_contract(),
            expert_base_identity=expert_base_identity(self.model.expert),
            model_config=self.model.trifuse.config,
            trifuse_state_dict=self.model.trifuse.state_dict(),
            ema_trifuse_state_dict=self.ema,
            global_step=self.global_step,
            training_state=dict(epoch=epoch, best_val_loss=best_val_loss),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict())
        temporary = str(path) + '.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        expected = dict(checkpoint_version=1, architecture=EXPERIMENT_ARCHITECTURE,
                        experiment_contract=self.experiment_contract,
                        runtime_config=self.runtime_contract(),
                        model_config=self.model.trifuse.config)
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f'experimental checkpoint mismatch: {name}')
        validate_expert_base_identity(payload, self.model.expert)
        self.model.trifuse.load_state_dict(payload['trifuse_state_dict'])
        self.ema = clone_state(payload['ema_trifuse_state_dict'])
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.scheduler.load_state_dict(payload['scheduler_state_dict'])
        self.global_step = payload['global_step']
        self.resume_epoch = payload['training_state']['epoch']
        self.resume_best_val_loss = payload['training_state']['best_val_loss']


class RolloutAugmentedTrainer(ExperimentalConsensusTrainer):
    """Mix teacher states with one-step states produced by the current model."""

    def __init__(self, *args, rollout_probability=0.5, rollout_steps=1, **kwargs):
        if not 0 <= rollout_probability <= 1:
            raise ValueError('rollout_probability must be in [0, 1]')
        if not isinstance(rollout_steps, int) or rollout_steps < 1:
            raise ValueError('rollout_steps must be a positive integer')
        self.rollout_probability = float(rollout_probability)
        self.rollout_steps = rollout_steps
        self.rollout_augmented_states = 0
        super().__init__(*args, **kwargs)

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        teacher = self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, t)
        if (self.rollout_probability == 0 or float(t.item()) >= 1
                or float(t.item()) < getattr(self, 'rollout_min_time', 0.0)
                or float(torch.rand((), device=t.device)) >= self.rollout_probability):
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
        self.rollout_augmented_states += 1
        return ligand.detach(), poi.detach(), body.detach(), teacher[3]

    def runtime_contract(self):
        return {
            **super().runtime_contract(),
            'rollout_probability': self.rollout_probability,
            'rollout_steps': self.rollout_steps,
        }


class EndpointTranslationRolloutTrainer(RolloutAugmentedTrainer):
    """Use the exact clean-endpoint E3 translation target off teacher paths."""

    def __init__(self, *args, rollout_min_time=0.25, endpoint_blend=0.25,
                 endpoint_correction_ratio=1.0, **kwargs):
        if not 0 < rollout_min_time < 1:
            raise ValueError('rollout_min_time must be in (0, 1)')
        if not 0 < endpoint_blend <= 1:
            raise ValueError('endpoint_blend must be in (0, 1]')
        if endpoint_correction_ratio <= 0:
            raise ValueError('endpoint_correction_ratio must be positive')
        self.rollout_min_time = float(rollout_min_time)
        self.endpoint_blend = float(endpoint_blend)
        self.endpoint_correction_ratio = float(endpoint_correction_ratio)
        self._endpoint_applied_states = 0
        super().__init__(*args, **kwargs)

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        state = super()._training_state(
            e3_data, poi_data, ligand_data, coupling, t, dt)
        ligand, poi, body, targets = state
        # The parent increments this counter only when it returns a generated
        # off-path state. Teacher states retain their analytic coupling target.
        if not ligand.requires_grad and self.rollout_augmented_states > getattr(
                self, '_endpoint_applied_states', 0):
            target_center = ligand_data['coords'].mean(dim=1)
            endpoint_tr = (ligand.mean(dim=1) - target_center) / t[:, None]
            original_tr = targets['e3_ligand']['tr']
            correction = endpoint_tr - original_tr
            limit = (self.diffusion.scales(
                e3_data['coords'], poi_data['coords'])[0]
                * self.endpoint_correction_ratio)
            norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            correction = correction * torch.clamp(limit / norm.clamp(min=1e-8), max=1)
            targets = {
                **targets,
                'e3_ligand': {
                    **targets['e3_ligand'],
                    'tr': (original_tr + self.endpoint_blend * correction).detach(),
                },
            }
            self._endpoint_applied_states = self.rollout_augmented_states
        return ligand, poi, body, targets

    def runtime_contract(self):
        return {
            **super().runtime_contract(),
            'offpath_target': 'e3_ligand_endpoint_translation_v1',
            'rollout_min_time': self.rollout_min_time,
            'endpoint_blend': self.endpoint_blend,
            'endpoint_correction_ratio': self.endpoint_correction_ratio,
        }
