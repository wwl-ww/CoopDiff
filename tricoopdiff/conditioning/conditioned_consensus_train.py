"""Experimental condition-aware training contract, isolated from v20 production weights."""

import math

from rdkit import Chem
import torch
from torch import nn

from tricoopdiff.conditioning.condition_augmentation import ConditionFeatureDropout, dropout_condition_features
from tricoopdiff.conditioning.condition_features import (
    ProtacConditionEncoder, condition_feature_shapes, build_supervised_contact_features,
)
from tricoopdiff.diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from tricoopdiff.models.consensus import ConsensusCoopDiff, ConsensusFuse
from tricoopdiff.models.expert import validate_expert_base_identity
from tricoopdiff.sdpi_flow import FLOW_VERSION
from tricoopdiff.runtime import atomic_torch_save, sha256_file as checkpoint_sha256
from tricoopdiff.train import ConsensusTrainer, validate_consensus_checkpoint


ARCHITECTURE = 'conditioned_residual_consensus_v3_prototype'


CONDITION_CONTRACT = 'protac_condition_features_v2_no_learned_provenance'


class ConditionedConsensusFuse(ConsensusFuse):
    """Condition residuals affect consensus only, never frozen expert proposals."""

    supports_conditions = True

    def __init__(self, **base_config):
        super().__init__(**base_config)
        # Keep common initialization and the caller RNG stream identical to v2.
        with torch.random.fork_rng(devices=[]):
            # Only CPU parameters are initialized here. manual_seed() would also
            # reset CUDA streams that this CPU-only fork does not restore.
            torch.random.default_generator.manual_seed(1701)
            self.condition_encoder = ProtacConditionEncoder(self.config['node_dim'])
            self.condition_node_projection = nn.Linear(self.config['node_dim'], self.config['node_dim'], bias=False)
            self.condition_gate_head = nn.Linear(self.config['node_dim'], 4, bias=False)
            # Exact function-preserving migration matters more than avoiding a
            # one-update delay in the new encoders. Each injection path first
            # learns whether to open, then propagates gradients into its body.
            self.condition_scale = nn.Parameter(torch.zeros(()))
            nn.init.normal_(self.condition_node_projection.weight, std=.01)
            nn.init.normal_(self.condition_gate_head.weight, std=.01)

    @classmethod
    def from_base(cls, base):
        if type(base) is not ConsensusFuse:
            raise ValueError('migration requires an unmodified base ConsensusFuse')
        with torch.random.fork_rng(devices=[]):
            model = cls(**base.config).to(next(base.parameters()).device)
        state = model.state_dict()
        state.update(base.state_dict())
        model.load_state_dict(state, strict=True)
        model.train(base.training)
        return model

    def prepare_conditions(self, graph, conditions):
        reference, types = graph['node_features'], graph['node_types']
        shapes = condition_feature_shapes(
            [int((types == 0).sum()), int((types == 1).sum())], int((types == 2).sum()))
        if set(conditions) != set(shapes):
            raise ValueError('condition feature fields do not match the versioned contract')
        values = {}
        for name, shape in shapes.items():
            value = torch.as_tensor(conditions[name], device=reference.device, dtype=reference.dtype).detach()
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f'condition/node alignment mismatch: {name}')
            values[name] = value
        encoded = self.condition_encoder(values)
        delta = reference.new_zeros(reference.shape[0], self.config['node_dim'])
        summaries = [encoded['sides'].mean(0, keepdim=True)]
        for index, name in enumerate(('protein1', 'protein2', 'ligand')):
            if encoded[name].shape[0] == 0:
                raise ValueError('conditioned consensus requires nonempty node groups')
            delta[types == index] = self.condition_node_projection(encoded[name]) * self.condition_scale
            summaries.append(encoded[name].mean(0, keepdim=True))
        raw_context = torch.stack(summaries).mean(0)
        context = raw_context * self.condition_scale
        gate_bias = self.condition_gate_head(raw_context) * self.condition_scale
        return {**graph, 'condition_node_residual': delta}, context, gate_bias

    def prototype_checkpoint(self, parent_sha256):
        if len(parent_sha256) != 64 or any(c not in '0123456789abcdef' for c in parent_sha256):
            raise ValueError('record the parent checkpoint SHA256')
        return {'format': 'conditioned_fuse_prototype_v1', 'architecture': ARCHITECTURE,
                'condition_contract': CONDITION_CONTRACT, 'base_model_config': dict(self.config),
                'parent_checkpoint_sha256': parent_sha256,
                'fuse_state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                'scope': 'experimental fusion module only; not a trained production refinement checkpoint'}


def load_prototype_checkpoint(payload):
    if (payload.get('format') != 'conditioned_fuse_prototype_v1'
            or payload.get('architecture') != ARCHITECTURE
            or payload.get('condition_contract') != CONDITION_CONTRACT):
        raise ValueError('unsupported condition prototype checkpoint')
    parent = payload.get('parent_checkpoint_sha256', '')
    if len(parent) != 64 or any(c not in '0123456789abcdef' for c in parent):
        raise ValueError('missing parent checkpoint identity')
    model = ConditionedConsensusFuse(**payload['base_model_config'])
    model.load_state_dict(payload['fuse_state_dict'], strict=True)
    return model


class ConditionedConsensusCoopDiff(ConsensusCoopDiff):
    """Explicit experimental wrapper; legacy evaluators cannot silently enable it."""

    architecture = ARCHITECTURE

    def __init__(self, frozen_expert, trifuse, **kwargs):
        if not isinstance(trifuse, ConditionedConsensusFuse):
            raise ValueError('conditional model requires ConditionedConsensusFuse')
        super().__init__(frozen_expert, trifuse, **kwargs)


CHECKPOINT_FORMAT = 'conditioned_consensus_training_v1'


TRAINING_CONDITION_CONTRACT = 'native_contact_masks_local_distances_6A_dropout_v3_no_provenance'


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


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
    return ConditionedConsensusCoopDiff(frozen_expert, conditioned)


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
        self._configure_optimizer([
            {'params': base, 'lr': float(previous['lr']), 'name': 'base_trifuse'},
            {'params': condition, 'lr': self.condition_lr, 'name': 'condition'},
        ], float(previous['weight_decay']))

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
            **self._optimization_payload({'epoch': epoch, 'best_val_loss': best_val_loss}),
        }
        validate_conditioned_checkpoint(payload)
        atomic_torch_save(payload, path)

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
        self._restore_optimization(payload)
        return payload
