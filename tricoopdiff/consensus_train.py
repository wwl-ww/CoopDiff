"""Stage 1 training: frozen binary priors, residual consensus and EMA validation."""

from contextlib import contextmanager
import math
import os

import torch

from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .consensus import (
    ARCHITECTURE, CONSENSUS_CHECKPOINT_VERSION,
    LEGACY_CONSENSUS_CHECKPOINT_VERSION, SOURCE_CONTRACT_VERSION,
)
from .diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from .sdpi_flow import FLOW_VERSION
from .equivariant_clamp import validate_clamp_mode
from .train import TriCoopDiffTrainer


def validate_consensus_checkpoint(payload):
    version = payload.get('checkpoint_version')
    if version not in (LEGACY_CONSENSUS_CHECKPOINT_VERSION,
                       CONSENSUS_CHECKPOINT_VERSION):
        raise ValueError('model-v2 requires checkpoint_version=18 or 19')
    if payload.get('architecture') != ARCHITECTURE or payload.get('generative_process') != FLOW_VERSION:
        raise ValueError('checkpoint does not describe residual consensus with SDPI flow')
    if payload.get('expert_lora_config') or payload.get('expert_lora_state_dict'):
        raise ValueError('Stage 1 consensus checkpoint cannot contain expert adapters')
    for key in ('trifuse_state_dict', 'ema_trifuse_state_dict', 'model_config', 'expert_base_identity'):
        if not payload.get(key):
            raise ValueError(f'consensus checkpoint missing {key}')
    config = payload.get('runtime_config', {})
    validate_clamp_mode(config.get('activation_clamp_mode', 'componentwise'))
    for key in ('activation_clamp', 'residual_weight', 'ema_decay'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid consensus runtime {key}')
    if config['activation_clamp'] <= 0 or config['residual_weight'] < 0 or not 0 <= config['ema_decay'] < 1:
        raise ValueError('invalid consensus runtime range')
    validate_score_norm_version(config)
    if version == CONSENSUS_CHECKPOINT_VERSION:
        if config.get('flow_time_contract') != 'shared_flow_expert_clock_v1':
            raise ValueError('checkpoint has an unknown flow/expert time contract')
        if config.get('candidate_source_contract') != SOURCE_CONTRACT_VERSION:
            raise ValueError('checkpoint has an unknown candidate source contract')
        if config.get('training_source_mix') not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('checkpoint has an invalid training source mix')


class ConsensusTrainer(TriCoopDiffTrainer):
    def __init__(self, *args, residual_weight=1e-3, ema_decay=0.999,
                 activation_clamp_mode='irrep_norm',
                 training_source_mix='sdpi_only', **kwargs):
        if not math.isfinite(residual_weight) or residual_weight < 0 or not 0 <= ema_decay < 1:
            raise ValueError('invalid residual regularization or EMA decay')
        kwargs['torsion_expert_aux_weight'] = 0.0
        kwargs['lora_warmup_steps'] = 0
        super().__init__(*args, **kwargs)
        self.activation_clamp_mode = validate_clamp_mode(activation_clamp_mode)
        os.environ['NAN_CLAMP_MODE'] = self.activation_clamp_mode
        self.residual_weight = float(residual_weight)
        self.ema_decay = float(ema_decay)
        if training_source_mix not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('training_source_mix must be sdpi_only or sdpi_bitseed_1_1')
        self.training_source_mix = training_source_mix
        self.set_phase(1)
        self.ema = {name: value.detach().clone() for name, value in self.model.trifuse.state_dict().items()}

    def train_step(self, **batch):
        result = super().train_step(**batch)
        with torch.no_grad():
            for name, value in self.model.trifuse.state_dict().items():
                if value.is_floating_point():
                    self.ema[name].lerp_(value.detach(), 1 - self.ema_decay)
                else:
                    self.ema[name].copy_(value)
        return result

    @contextmanager
    def ema_parameters(self):
        raw = {name: value.detach().clone() for name, value in self.model.trifuse.state_dict().items()}
        self.model.trifuse.load_state_dict(self.ema)
        try:
            yield
        finally:
            self.model.trifuse.load_state_dict(raw)

    def validate_step(self, **batch):
        with self.ema_parameters():
            return super().validate_step(**batch)

    def save_checkpoint(self, path, epoch=0, best_val_loss=float('inf')):
        payload = {
            'checkpoint_version': CONSENSUS_CHECKPOINT_VERSION,
            'architecture': ARCHITECTURE, 'generative_process': FLOW_VERSION,
            'expert_base_identity': expert_base_identity(self.model.expert),
            'expert_lora_config': None, 'expert_lora_state_dict': {},
            'model_config': self.model.trifuse.config,
            'runtime_config': {
                'torus_score_norm_version': SCORE_NORM_VERSION,
                'activation_clamp': self.activation_clamp,
                'activation_clamp_mode': self.activation_clamp_mode,
                'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
                'optimizer_groups': self.optimizer_config,
                'flow_time_contract': 'shared_flow_expert_clock_v1',
                'candidate_source_contract': SOURCE_CONTRACT_VERSION,
                'training_source_mix': self.training_source_mix,
            },
            'trifuse_state_dict': self.model.trifuse.state_dict(),
            'ema_trifuse_state_dict': self.ema,
            'global_step': self.global_step,
            'training_state': {'epoch': epoch, 'best_val_loss': best_val_loss},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        validate_consensus_checkpoint(payload)
        temporary = f'{path}.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_consensus_checkpoint(payload)
        validate_expert_base_identity(payload, self.model.expert)
        saved_model_config = dict(payload['model_config'])
        if payload['checkpoint_version'] == LEGACY_CONSENSUS_CHECKPOINT_VERSION:
            saved_model_config.setdefault('source_conditioning', False)
        if saved_model_config != self.model.trifuse.config:
            raise ValueError('consensus model configuration differs from checkpoint')
        saved_mode = payload['runtime_config'].get('activation_clamp_mode', 'componentwise')
        if saved_mode != self.activation_clamp_mode:
            raise ValueError(f'checkpoint activation_clamp_mode={saved_mode!r}; '
                             'construct the trainer with that explicit mode to resume')
        expected = {'activation_clamp': self.activation_clamp,
                    'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
                    'optimizer_groups': self.optimizer_config}
        if payload['checkpoint_version'] == CONSENSUS_CHECKPOINT_VERSION:
            expected.update({
                'flow_time_contract': 'shared_flow_expert_clock_v1',
                'candidate_source_contract': SOURCE_CONTRACT_VERSION,
                'training_source_mix': self.training_source_mix,
            })
        for key, value in expected.items():
            if payload['runtime_config'].get(key) != value:
                raise ValueError(f'consensus checkpoint runtime mismatch: {key}')
        self.model.trifuse.load_state_dict(payload['trifuse_state_dict'])
        self.ema = {name: value.clone() for name, value in payload['ema_trifuse_state_dict'].items()}
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.scheduler.load_state_dict(payload['scheduler_state_dict'])
        self.global_step = payload['global_step']
        self.resume_epoch = payload['training_state']['epoch']
        self.resume_best_val_loss = payload['training_state']['best_val_loss']
