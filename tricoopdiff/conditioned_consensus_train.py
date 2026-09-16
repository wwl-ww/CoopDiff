"""Experimental condition-aware training contract, isolated from v20 production weights."""

import math
import os
from pathlib import Path
import hashlib

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem

from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .condition_augmentation import ConditionFeatureDropout, dropout_condition_features
from .condition_features import build_supervised_contact_features
from .conditional_consensus import ARCHITECTURE, CONDITION_CONTRACT, ConditionedConsensusFuse
from .conditional_consensus import ConditionedConsensusCoopDiff
from .consensus import ConsensusFuse
from .consensus_train import ConsensusTrainer
from .consensus_train import validate_consensus_checkpoint
from .diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from .sdpi_flow import FLOW_VERSION


CHECKPOINT_FORMAT = 'conditioned_consensus_training_v1'
TRAINING_CONDITION_CONTRACT = 'native_contact_masks_local_distances_6A_dropout_v3_no_provenance'


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def checkpoint_sha256(path):
    """Hash a parent checkpoint without loading a second in-memory copy."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_conditioned_model(parent_payload, frozen_expert):
    """Migrate the exact EMA C checkpoint into the isolated architecture."""
    validate_consensus_checkpoint(parent_payload)
    validate_expert_base_identity(parent_payload, frozen_expert)
    config = dict(parent_payload['model_config'])
    if not config.get('source_conditioning'):
        raise ValueError('conditioned protocol requires the C parent with source conditioning')
    base = ConsensusFuse(**config)
    base.load_state_dict(parent_payload['ema_trifuse_state_dict'], strict=True)
    conditioned = ConditionedConsensusFuse.from_base(base)
    return ConditionedConsensusCoopDiff(
        frozen_expert, conditioned, num_conv_layers=frozen_expert.num_conv_layers)


def supervised_condition_batch(batch, *, seed, dropout=ConditionFeatureDropout(), cutoff=6.):
    """Create training-only masks in current node order and discard native coordinates."""
    if type(seed) is not int or seed < 0:
        raise ValueError('condition batch seed must be a nonnegative integer')
    molecule = Chem.Mol(batch['ligand_data']['mol'])
    if molecule is None:
        raise ValueError('cached ligand chemical graph cannot be restored')
    features = build_supervised_contact_features(
        molecule,
        (batch['e3_data']['coords'][0], batch['poi_data']['coords'][0]),
        batch['ligand_data']['coords'][0], cutoff=cutoff)
    features, audit = dropout_condition_features(features, seed=seed, config=dropout)
    audit.update(condition_contract=TRAINING_CONDITION_CONTRACT, pocket_cutoff_angstrom=float(cutoff),
                 case=batch.get('case', 'unknown'), source='native_training_geometry')
    return features, audit


def validate_conditioned_checkpoint(payload):
    if (payload.get('format') != CHECKPOINT_FORMAT or payload.get('architecture') != ARCHITECTURE
            or payload.get('condition_contract') != CONDITION_CONTRACT
            or payload.get('training_condition_contract') != TRAINING_CONDITION_CONTRACT
            or payload.get('generative_process') != FLOW_VERSION):
        raise ValueError('unsupported conditioned consensus training checkpoint')
    if not _sha256(payload.get('parent_checkpoint_sha256')):
        raise ValueError('conditioned checkpoint requires the exact parent checkpoint identity')
    for key in ('trifuse_state_dict', 'ema_trifuse_state_dict', 'model_config',
                'expert_base_identity', 'optimizer_state_dict', 'scheduler_state_dict'):
        if not payload.get(key):
            raise ValueError(f'conditioned checkpoint missing {key}')
    runtime = payload.get('runtime_config', {})
    for key in ('activation_clamp', 'residual_weight', 'ema_decay', 'pocket_cutoff_angstrom',
                'condition_lr'):
        value = runtime.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid conditioned runtime value: {key}')
    if (runtime['activation_clamp'] <= 0 or runtime['residual_weight'] < 0
            or not 0 <= runtime['ema_decay'] < 1 or runtime['pocket_cutoff_angstrom'] != 6.
            or runtime['condition_lr'] <= 0):
        raise ValueError('invalid conditioned runtime range/contract')
    dropout = runtime.get('condition_dropout')
    if not isinstance(dropout, dict):
        raise ValueError('condition dropout policy must be explicit')
    ConditionFeatureDropout(**dropout).validate()
    validate_score_norm_version(runtime)
    if runtime.get('training_source_mix') != 'sdpi_bitseed_1_1':
        raise ValueError('first conditioned model requires the established dual-init source mix')
    if runtime.get('training_endpoint_protocol') != 'online_bitseed_training_c1_m1_v1':
        raise ValueError('conditioned mixed training requires online C=1/M=1 endpoints')
    if runtime.get('parent_weight_source') != 'ema_trifuse_state_dict':
        raise ValueError('conditioned training must initialize from the parent EMA weights')
    if runtime.get('native_coordinates_stored_in_conditions') is not False:
        raise ValueError('conditioned checkpoint cannot store native condition coordinates')


class ConditionedConsensusTrainer(ConsensusTrainer):
    """EMA trainer whose checkpoint cannot be loaded as an old C40/v20 model."""

    def __init__(self, *args, parent_checkpoint_sha256, condition_dropout=ConditionFeatureDropout(),
                 pocket_cutoff_angstrom=6., condition_lr=1e-3, **kwargs):
        if not _sha256(parent_checkpoint_sha256):
            raise ValueError('record the C40 parent checkpoint SHA256')
        condition_dropout.validate()
        if pocket_cutoff_angstrom != 6.:
            raise ValueError('the first protocol is fixed to the DeepTernary 6 Å pocket cutoff')
        super().__init__(*args, **kwargs)
        if not isinstance(self.model.trifuse, ConditionedConsensusFuse):
            raise ValueError('conditioned trainer requires the isolated conditioned architecture')
        self.parent_checkpoint_sha256 = parent_checkpoint_sha256
        self.condition_dropout = condition_dropout
        self.pocket_cutoff_angstrom = float(pocket_cutoff_angstrom)
        if not isinstance(condition_lr, (int, float)) or not math.isfinite(condition_lr) or condition_lr <= 0:
            raise ValueError('condition learning rate must be finite and positive')
        self.condition_lr = float(condition_lr)
        base, condition = [], []
        for name, parameter in self.model.trifuse.named_parameters():
            (condition if name.startswith('condition_') else base).append(parameter)
        if not base or not condition or set(map(id, base)) & set(map(id, condition)):
            raise AssertionError('condition optimizer partition is incomplete')
        previous = self.optimizer.param_groups[0]
        self.optimizer = AdamW([
            {'params': base, 'lr': float(previous['lr']), 'name': 'base_trifuse'},
            {'params': condition, 'lr': self.condition_lr, 'name': 'condition'},
        ], weight_decay=float(previous['weight_decay']))
        self.optimizer_config = {
            group['name']: {'lr': float(group['lr']), 'weight_decay': float(group['weight_decay'])}
            for group in self.optimizer.param_groups}
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=.5, patience=10, min_lr=1e-7)

    def condition_batch(self, batch, *, seed):
        return supervised_condition_batch(
            batch, seed=seed, dropout=self.condition_dropout, cutoff=self.pocket_cutoff_angstrom)

    def save_checkpoint(self, path, epoch=0, best_val_loss=float('inf')):
        payload = {
            'format': CHECKPOINT_FORMAT, 'architecture': ARCHITECTURE,
            'condition_contract': CONDITION_CONTRACT,
            'training_condition_contract': TRAINING_CONDITION_CONTRACT,
            'generative_process': FLOW_VERSION,
            'parent_checkpoint_sha256': self.parent_checkpoint_sha256,
            'expert_base_identity': expert_base_identity(self.model.expert),
            'model_config': dict(self.model.trifuse.config),
            'runtime_config': {
                'torus_score_norm_version': SCORE_NORM_VERSION,
                'activation_clamp': self.activation_clamp,
                'activation_clamp_mode': self.activation_clamp_mode,
                'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
                'optimizer_groups': self.optimizer_config,
                'training_source_mix': self.training_source_mix,
                'training_endpoint_protocol': self.training_endpoint_protocol,
                'parent_weight_source': 'ema_trifuse_state_dict',
                'pocket_cutoff_angstrom': self.pocket_cutoff_angstrom,
                'condition_dropout': vars(self.condition_dropout),
                'condition_lr': self.condition_lr,
                'native_coordinates_stored_in_conditions': False,
            },
            'trifuse_state_dict': self.model.trifuse.state_dict(),
            'ema_trifuse_state_dict': self.ema,
            'global_step': self.global_step,
            'training_state': {'epoch': epoch, 'best_val_loss': best_val_loss},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        validate_conditioned_checkpoint(payload)
        temporary = f'{path}.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path, *, allow_dropout_change=False):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_conditioned_checkpoint(payload)
        if payload['parent_checkpoint_sha256'] != self.parent_checkpoint_sha256:
            raise ValueError('conditioned parent checkpoint mismatch')
        validate_expert_base_identity(payload, self.model.expert)
        if payload['model_config'] != self.model.trifuse.config:
            raise ValueError('conditioned model configuration mismatch')
        expected = {
            'activation_clamp': self.activation_clamp, 'activation_clamp_mode': self.activation_clamp_mode,
            'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
            'optimizer_groups': self.optimizer_config, 'training_source_mix': self.training_source_mix,
            'training_endpoint_protocol': self.training_endpoint_protocol,
            'parent_weight_source': 'ema_trifuse_state_dict',
            'pocket_cutoff_angstrom': self.pocket_cutoff_angstrom,
            'condition_dropout': vars(self.condition_dropout), 'native_coordinates_stored_in_conditions': False,
            'condition_lr': self.condition_lr,
        }
        for key, value in expected.items():
            if key == 'condition_dropout' and allow_dropout_change:
                continue
            if payload['runtime_config'].get(key) != value:
                raise ValueError(f'conditioned checkpoint runtime mismatch: {key}')
        self.model.trifuse.load_state_dict(payload['trifuse_state_dict'], strict=True)
        self.ema = {name: value.clone() for name, value in payload['ema_trifuse_state_dict'].items()}
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.scheduler.load_state_dict(payload['scheduler_state_dict'])
        self.global_step = int(payload['global_step'])
        self.resume_epoch = int(payload['training_state']['epoch'])
        self.resume_best_val_loss = float(payload['training_state']['best_val_loss'])
        return payload
