"""Isolated optimizer/EMA/checkpoint contract for complete two-sided charts."""

import copy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile

import torch

from . import diffdock_compat  # noqa: F401
from .anchored_consensus import anchored_target_loss
from .anchored_flow import FLOW_VERSION
from .anchored_joint_motion import ACTION_POLICY
from .bounded_binary_flow import BOUNDED_FLOW_VERSION
from .bounded_binary_training import BOUNDED_TRAINING_CONTRACT, validate_binary_bounds
from .bounded_joint_motion import BOUNDED_ACTION_POLICY
from .anchored_fragments import FRAGMENT_POLICY
from .anchored_training import TRAINING_CONTRACT
from .anchored_tasks import task_contract, validate_task_sample
from .chemical_prior_contract import validate_prior_config
from .anchored_data_schedule import INDEX_ORDER, CLUSTER_ORDER
from .anchored_training_metric import PARAMETER_METRIC, PHYSICAL_METRIC
from .anchored_action_adapter import ADAPTER_POLICY, ADAPTER_ARCHITECTURE, install_action_adapter
from .coupled_linker_action import COUPLED_ADAPTER_POLICY, COUPLED_ADAPTER_ARCHITECTURE
from .prior_joint_action import PRIOR_ACTION_POLICY, PRIOR_ACTION_ARCHITECTURE
from .anchored_validation import VALIDATION_POLICY, VALIDATION_TIMES
from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .conditional_consensus import (ARCHITECTURE, CONDITION_CONTRACT,
                                    ConditionedConsensusCoopDiff, ConditionedConsensusFuse)
from .diffdock.utils.torus import SCORE_NORM_VERSION
from .equivariant_clamp import validate_clamp_mode


FORMAT = 'anchored_consensus_training_v3'
TASK_PRIOR_FORMAT = 'anchored_consensus_training_v4'
METRIC_DATA_FORMAT = 'anchored_consensus_training_v5'
ADAPTER_FORMAT = 'anchored_consensus_training_v6'
COUPLED_ADAPTER_FORMAT = 'anchored_consensus_training_v8'
LEGACY_COUPLED_POLICY = 'anchored_body_aware_coupled_linker_residual_v2'
SOURCE_POLICY = 'anchored_only_reuses_source_embedding_zero_v1'


def expert_environment():
    return {'activation_clamp': os.environ.get('NAN_CLAMP'),
            'activation_clamp_mode': os.environ.get('NAN_CLAMP_MODE', 'componentwise'),
            'torus_score_norm_version': SCORE_NORM_VERSION}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_validation_contract(contract):
    if contract is None:
        return
    if (set(contract) != {'policy', 'index_sha256', 'seed', 'times'}
            or contract['policy'] != VALIDATION_POLICY or not _sha(contract['index_sha256'])
            or type(contract['seed']) is not int or contract['seed'] < 0
            or contract['times'] != list(VALIDATION_TIMES)):
        raise ValueError('invalid anchored validation contract')


def validate_anchored_checkpoint(payload):
    bounds = payload.get('binary_bounds')
    validate_binary_bounds(bounds)
    contract = task_contract(payload.get('conditioning_task', 'dual_binary'), bounds)
    validate_prior_config(payload.get('prior_config'))
    assembly_prior = (payload.get('prior_config') or {}).get('binary_offset_prior') == 'assembly_pose_mixture_v1'
    if assembly_prior != (payload.get('conditioning_task') == 'dual_assembly'):
        raise ValueError('assembly task and pose corruption prior must be versioned together')
    metric = payload.get('binary_metric', PARAMETER_METRIC)
    schedule = payload.get('data_schedule', INDEX_ORDER)
    if metric not in (PARAMETER_METRIC, PHYSICAL_METRIC) or schedule not in (INDEX_ORDER, CLUSTER_ORDER):
        raise ValueError('unsupported binary metric or training data schedule')
    revised = payload.get('prior_config') is not None or payload.get('conditioning_task', 'dual_binary') != 'dual_binary'
    format_version = METRIC_DATA_FORMAT if metric != PARAMETER_METRIC or schedule != INDEX_ORDER else (
        TASK_PRIOR_FORMAT if revised else FORMAT)
    adapter = payload.get('action_adapter')
    if adapter not in (None, ADAPTER_POLICY, COUPLED_ADAPTER_POLICY, LEGACY_COUPLED_POLICY, PRIOR_ACTION_POLICY):
        raise ValueError('unsupported anchored action adapter')
    coupled = adapter == COUPLED_ADAPTER_POLICY
    expected = {'format': COUPLED_ADAPTER_FORMAT if coupled else ADAPTER_FORMAT if adapter else format_version,
                'architecture': COUPLED_ADAPTER_ARCHITECTURE if coupled else ADAPTER_ARCHITECTURE if adapter else ARCHITECTURE,
                'condition_contract': CONDITION_CONTRACT,
                'source_policy': SOURCE_POLICY, **contract}
    if adapter == LEGACY_COUPLED_POLICY:
        expected.update(format='anchored_consensus_training_v7',
                        architecture='conditioned_consensus_v8_with_coupled_linker_action_v2')
    if adapter == PRIOR_ACTION_POLICY:
        expected.update(format='anchored_consensus_training_v9', architecture=PRIOR_ACTION_ARCHITECTURE)
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f'anchored checkpoint contract mismatch: {name}')
    for name in ('parent_checkpoint_sha256', 'training_index_sha256'):
        if not _sha(payload.get(name)):
            raise ValueError(f'anchored checkpoint missing identity: {name}')
    if type(payload.get('global_step')) is not int or payload['global_step'] < 0:
        raise ValueError('invalid anchored global step')
    if payload.get('scope') not in ('training', 'implementation_smoke'):
        raise ValueError('checkpoint scope must distinguish training from implementation smoke')
    libraries = payload.get('prior_library_identity')
    if libraries is not None:
        if (payload.get('prior_config') is None or not isinstance(libraries, dict)
                or set(libraries) != {'train', 'validation'} or not _sha(libraries['train'])
                or (libraries['validation'] is not None and not _sha(libraries['validation']))):
            raise ValueError('invalid chemical prior library identities')
    if payload.get('prior_config') is not None and payload['scope'] == 'training':
        if libraries is None or not _sha(libraries['validation']):
            raise ValueError('chemical-prior training requires immutable train/validation library hashes')
    contract = payload.get('validation_contract')
    validate_validation_contract(contract)
    if payload['scope'] == 'training' and contract is None:
        raise ValueError('training checkpoints require a validation contract')
    selection = payload.get('validation_state', {})
    if set(selection) != {'best_loss', 'best_step', 'cohort_sha256'}:
        raise ValueError('missing validation selection state')
    if selection['best_loss'] is None:
        if selection['best_step'] is not None or selection['cohort_sha256'] is not None:
            raise ValueError('inconsistent empty validation state')
        if payload['scope'] == 'training':
            raise ValueError('training checkpoint has no completed validation')
    elif (contract is None or not isinstance(selection['best_loss'], (int, float))
          or not math.isfinite(selection['best_loss']) or selection['best_loss'] < 0
          or type(selection['best_step']) is not int or not 0 <= selection['best_step'] <= payload['global_step']
          or not _sha(selection['cohort_sha256'])):
        raise ValueError('invalid validation selection state')
    for name in ('model_config', 'expert_base_identity', 'trifuse_state_dict',
                 'ema_trifuse_state_dict', 'optimizer_state_dict'):
        if not isinstance(payload.get(name), dict) or not payload[name]:
            raise ValueError(f'missing anchored checkpoint field: {name}')
    if payload['model_config'].get('source_conditioning') is not True:
        raise ValueError('anchored source policy requires the source-conditioned network')
    runtime = payload.get('runtime_config', {})
    for name in ('lr', 'weight_decay', 'ema_decay', 'cartesian_weight', 'grad_clip'):
        if (not isinstance(runtime.get(name), (int, float))
                or not math.isfinite(runtime[name])):
            raise ValueError(f'invalid anchored runtime value: {name}')
    if (runtime['lr'] <= 0 or runtime['weight_decay'] < 0 or not 0 <= runtime['ema_decay'] < 1
            or runtime['cartesian_weight'] < 0 or runtime['grad_clip'] <= 0):
        raise ValueError('invalid anchored optimizer/loss settings')
    progress = payload.get('progress')
    if (not isinstance(progress, dict) or set(progress) != {'epoch', 'next_case', 'seed'}
            or any(type(value) is not int or value < 0 for value in progress.values())):
        raise ValueError('checkpoint must record the deterministic data cursor')
    environment = payload.get('expert_environment', {})
    if environment.get('torus_score_norm_version') != SCORE_NORM_VERSION:
        raise ValueError('invalid expert runtime contract')
    validate_clamp_mode(environment.get('activation_clamp_mode'))
    clamp = environment.get('activation_clamp')
    if clamp is not None:
        try:
            valid = math.isfinite(float(clamp)) and float(clamp) > 0
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError('invalid expert activation clamp')
    rng = payload.get('rng_state', {})
    if not torch.is_tensor(rng.get('cpu')) or rng['cpu'].dtype != torch.uint8:
        raise ValueError('missing torch CPU random state')
    if rng.get('cuda') is not None and (not torch.is_tensor(rng['cuda']) or rng['cuda'].dtype != torch.uint8):
        raise ValueError('invalid torch CUDA random state')
    for name in ('trifuse_state_dict', 'ema_trifuse_state_dict'):
        if any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in payload[name].values()):
            raise ValueError('nonfinite anchored model state')


def build_anchored_model(payload, expert, *, use_ema=True, allow_smoke=False):
    """Load only this process, with explicit allowance needed for smoke artifacts."""
    validate_anchored_checkpoint(payload)
    if payload['scope'] == 'implementation_smoke' and not allow_smoke:
        raise ValueError('implementation smoke is not a trained prediction checkpoint')
    validate_expert_base_identity(payload, expert)
    if payload['expert_environment'] != expert_environment():
        raise ValueError('set the saved expert runtime environment explicitly before loading')
    fuse = ConditionedConsensusFuse(**payload['model_config'])
    model = ConditionedConsensusCoopDiff(expert, fuse, num_conv_layers=expert.num_conv_layers)
    if payload.get('action_adapter'):
        install_action_adapter(model, policy=payload['action_adapter'])
    fuse.load_state_dict(payload['ema_trifuse_state_dict' if use_ema else 'trifuse_state_dict'], strict=True)
    return model


def warm_start_anchored_weights(model, payload, *, reset_action_adapter=False):
    """Explicit weight transfer across tasks; never restore optimizer/validation."""
    validate_anchored_checkpoint(payload)
    if payload['scope'] != 'training':
        raise ValueError('weight transfer requires a trained checkpoint, not a smoke artifact')
    validate_expert_base_identity(payload, model.expert)
    if (payload['model_config'] != model.trifuse.config
            or payload['expert_environment'] != expert_environment()):
        raise ValueError('warm-start network or expert runtime mismatch')
    target_adapter = getattr(getattr(model.trifuse, 'anchored_action_adapter', None), 'policy', None)
    source_adapter = payload.get('action_adapter')
    if reset_action_adapter:
        if target_adapter not in (COUPLED_ADAPTER_POLICY, PRIOR_ACTION_POLICY):
            raise ValueError('explicit action replacement requires the unified action module')
        prefix = 'anchored_action_adapter.'
        state = {key: value for key, value in payload['ema_trifuse_state_dict'].items()
                 if not key.startswith(prefix)}
        target = model.trifuse.state_dict()
        if set(state) != {key for key in target if not key.startswith(prefix)}:
            raise ValueError('action replacement requires exact shared backbone coverage')
        model.trifuse.load_state_dict({**target, **state}, strict=True)
        return
    if source_adapter == LEGACY_COUPLED_POLICY or (source_adapter == ADAPTER_POLICY
                                                  and target_adapter == COUPLED_ADAPTER_POLICY):
        raise ValueError('replacing stacked heads requires explicit reset_action_adapter')
    if ((source_adapter and not target_adapter)
            or source_adapter == COUPLED_ADAPTER_POLICY and target_adapter != COUPLED_ADAPTER_POLICY):
        raise ValueError('warm-start cannot silently drop a trained body-aware adapter')
    state = payload['ema_trifuse_state_dict']
    if target_adapter != source_adapter:
        prefix = 'anchored_action_adapter.coupled_' if source_adapter == ADAPTER_POLICY else 'anchored_action_adapter.'
        expected = {name for name in model.trifuse.state_dict() if not name.startswith(prefix)}
        if set(state) != expected:
            raise ValueError('body-aware migration requires exact pretrained backbone coverage')
        state = {**model.trifuse.state_dict(), **state}
    model.trifuse.load_state_dict(state, strict=True)


class AnchoredTrainer:
    """Complete mapped binary conditions only; no feature-only condition dropout."""

    def __init__(self, model, *, parent_checkpoint_sha256, training_index_sha256,
                 lr=1e-4, weight_decay=1e-4, ema_decay=.999, cartesian_weight=.1, grad_clip=10.,
                 validation_contract=None, binary_bounds=None, conditioning_task='dual_binary', prior_config=None,
                 prior_library_identity=None, binary_metric=PARAMETER_METRIC, data_schedule=INDEX_ORDER):
        if not isinstance(model, ConditionedConsensusCoopDiff):
            raise ValueError('anchored trainer requires the conditioned consensus network')
        if model.trifuse.config.get('source_conditioning') is not True:
            raise ValueError('anchored training requires a source-conditioned parent network')
        if not _sha(parent_checkpoint_sha256) or not _sha(training_index_sha256):
            raise ValueError('record exact parent and training-index hashes')
        if any(p.requires_grad for p in model.expert.parameters()):
            raise ValueError('anchored training requires a frozen expert')
        expert_base_identity(model.expert)
        self.model = model
        self.action_adapter = getattr(getattr(model.trifuse, 'anchored_action_adapter', None), 'policy', None)
        self.parent_sha = parent_checkpoint_sha256
        self.index_sha = training_index_sha256
        validate_validation_contract(validation_contract)
        self.validation_contract = copy.deepcopy(validation_contract)
        validate_binary_bounds(binary_bounds)
        self.binary_bounds = copy.deepcopy(binary_bounds)
        self.task_contract = task_contract(conditioning_task, binary_bounds)
        self.conditioning_task = conditioning_task
        validate_prior_config(prior_config)
        self.prior_config = copy.deepcopy(prior_config)
        self.prior_library_identity = copy.deepcopy(prior_library_identity)
        if binary_metric not in (PARAMETER_METRIC, PHYSICAL_METRIC) or data_schedule not in (INDEX_ORDER, CLUSTER_ORDER):
            raise ValueError('unsupported binary metric or training data schedule')
        self.binary_metric, self.data_schedule = binary_metric, data_schedule
        self.validation_state = dict(best_loss=None, best_step=None, cohort_sha256=None)
        self.environment = expert_environment()
        self.runtime = dict(lr=lr, weight_decay=weight_decay, ema_decay=ema_decay,
                            cartesian_weight=cartesian_weight, grad_clip=grad_clip)
        if (not all(math.isfinite(v) for v in self.runtime.values()) or lr <= 0
                or weight_decay < 0 or not 0 <= ema_decay < 1 or cartesian_weight < 0 or grad_clip <= 0):
            raise ValueError('invalid anchored training settings')
        self.optimizer = torch.optim.AdamW(model.trifuse.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema = {key: value.detach().clone() for key, value in model.trifuse.state_dict().items()}
        self.global_step = 0

    @contextmanager
    def ema_mode(self):
        """Validation must not alter online weights, optimizer or training RNG."""
        state = {key: value.detach().clone() for key, value in self.model.trifuse.state_dict().items()}
        was_training = self.model.training
        device = next(self.model.trifuse.parameters()).device
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            try:
                self.model.trifuse.load_state_dict(self.ema, strict=True)
                self.model.eval()
                yield
            finally:
                self.model.trifuse.load_state_dict(state, strict=True)
                self.model.train(was_training)
                self.model.expert.eval()

    @torch.no_grad()
    def evaluate(self, sample, e3, poi, ligand, *, time, dt):
        if self.model.training or self.environment != expert_environment():
            raise ValueError('validation requires eval mode and unchanged expert settings')
        validate_task_sample(sample, self.conditioning_task, self.binary_bounds, self.prior_config)
        losses = anchored_target_loss(
            self.model, sample.flow, ligand['coords'][0], poi['coords'][0], sample.prior_angles,
            e3, poi, ligand, time, dt, conditions=sample.conditions, source=0,
            cartesian_weight=self.runtime['cartesian_weight'], binary_metric=self.binary_metric)
        return {key: float(value) for key, value in losses.items() if key.startswith('loss_')}

    def record_validation(self, report):
        """Select only a fixed cohort under the declared development protocol."""
        if (report.get('binary_metric', PARAMETER_METRIC) != self.binary_metric
                or report.get('conditioning_task', 'dual_binary') != self.conditioning_task):
            raise ValueError('validation task or binary metric mismatch')
        if self.validation_contract is None or any(report.get(key) != value
                                                   for key, value in self.validation_contract.items()):
            raise ValueError('validation report does not match checkpoint contract')
        loss, cohort = report.get('mean_loss'), report.get('cohort_sha256')
        if (report.get('global_step') != self.global_step or not _sha(cohort)
                or type(report.get('evaluated')) is not int or report['evaluated'] < 1
                or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0):
            raise ValueError('invalid validation report')
        prior = self.validation_state
        if prior['cohort_sha256'] is not None and prior['cohort_sha256'] != cohort:
            raise ValueError('validation cohort changed; comparisons are invalid')
        improved = prior['best_loss'] is None or loss < prior['best_loss']
        if improved:
            self.validation_state = dict(best_loss=loss, best_step=self.global_step, cohort_sha256=cohort)
        return improved

    def step(self, sample, e3, poi, ligand, *, time, dt):
        if any(p.requires_grad for p in self.model.expert.parameters()):
            raise ValueError('expert was unfrozen during anchored training')
        if self.environment != expert_environment():
            raise ValueError('expert runtime environment changed during anchored training')
        validate_task_sample(sample, self.conditioning_task, self.binary_bounds, self.prior_config)
        self.model.train()
        self.model.expert.eval()
        self.optimizer.zero_grad(set_to_none=True)
        losses = anchored_target_loss(
            self.model, sample.flow, ligand['coords'][0], poi['coords'][0], sample.prior_angles,
            e3, poi, ligand, time, dt, conditions=sample.conditions, source=0,
            cartesian_weight=self.runtime['cartesian_weight'], binary_metric=self.binary_metric)
        losses['loss_total'].backward()
        if any(p.grad is not None for p in self.model.expert.parameters()):
            raise RuntimeError('frozen expert received a gradient')
        norm = torch.nn.utils.clip_grad_norm_(self.model.trifuse.parameters(),
                                              self.runtime['grad_clip'], error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for key, value in self.model.trifuse.state_dict().items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError('nonfinite anchored parameters after update')
                if value.is_floating_point():
                    self.ema[key].lerp_(value, 1 - self.runtime['ema_decay'])
                else:
                    self.ema[key].copy_(value)
        self.global_step += 1
        joint = losses.get('model_output', {}).get('anchored_joint_motion', {})
        return {**{key: float(value.detach()) for key, value in losses.items() if key.startswith('loss_')},
                **{f'joint_{key}': float(value) for key, value in joint.items()
                   if key.endswith('_rms')},
                'gradient_norm_before_clipping': float(norm), 'global_step': self.global_step}

    def checkpoint(self, *, epoch, next_case, seed, scope='training'):
        device = next(self.model.trifuse.parameters()).device
        revised = self.prior_config is not None or self.conditioning_task != 'dual_binary'
        format_version = METRIC_DATA_FORMAT if self.binary_metric != PARAMETER_METRIC or self.data_schedule != INDEX_ORDER else (
            TASK_PRIOR_FORMAT if revised else FORMAT)
        coupled = self.action_adapter == COUPLED_ADAPTER_POLICY
        payload = dict(format=COUPLED_ADAPTER_FORMAT if coupled else ADAPTER_FORMAT if self.action_adapter else format_version,
                       architecture=COUPLED_ADAPTER_ARCHITECTURE if coupled else ADAPTER_ARCHITECTURE if self.action_adapter else ARCHITECTURE,
                       action_adapter=self.action_adapter, condition_contract=CONDITION_CONTRACT,
                       **self.task_contract, source_policy=SOURCE_POLICY, scope=scope,
                       conditioning_task=self.conditioning_task, prior_config=copy.deepcopy(self.prior_config),
                       prior_library_identity=copy.deepcopy(self.prior_library_identity),
                       binary_metric=self.binary_metric, data_schedule=self.data_schedule,
                       binary_bounds=copy.deepcopy(self.binary_bounds),
                       parent_checkpoint_sha256=self.parent_sha, training_index_sha256=self.index_sha,
                       expert_base_identity=expert_base_identity(self.model.expert),
                       model_config=dict(self.model.trifuse.config), runtime_config=dict(self.runtime),
                       global_step=self.global_step, progress=dict(epoch=epoch, next_case=next_case, seed=seed),
                       trifuse_state_dict=self.model.trifuse.state_dict(), ema_trifuse_state_dict=self.ema,
                       optimizer_state_dict=self.optimizer.state_dict())
        payload['expert_environment'] = dict(self.environment)
        if self.action_adapter == PRIOR_ACTION_POLICY:
            payload.update(format='anchored_consensus_training_v9', architecture=PRIOR_ACTION_ARCHITECTURE)
        payload['validation_contract'] = copy.deepcopy(self.validation_contract)
        payload['validation_state'] = dict(self.validation_state)
        payload['rng_state'] = {'cpu': torch.get_rng_state(),
                                'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
        validate_anchored_checkpoint(payload)
        return copy.deepcopy(payload)

    def restore(self, payload):
        """Restore optimizer and EMA only when data, parent, network and runtime agree."""
        validate_anchored_checkpoint(payload)
        validate_expert_base_identity(payload, self.model.expert)
        if payload.get('action_adapter') != self.action_adapter:
            raise ValueError('resume cannot add or remove the body-aware action adapter')
        if payload.get('binary_bounds') != self.binary_bounds:
            raise ValueError('resume cannot change binary uncertainty bounds')
        if (payload.get('conditioning_task', 'dual_binary') != self.conditioning_task
                or payload.get('prior_config') != self.prior_config
                or payload.get('prior_library_identity') != self.prior_library_identity):
            raise ValueError('resume cannot change conditioning task or chemical prior')
        if (payload.get('binary_metric', PARAMETER_METRIC) != self.binary_metric
                or payload.get('data_schedule', INDEX_ORDER) != self.data_schedule):
            raise ValueError('resume cannot change binary metric or data schedule')
        device = next(self.model.trifuse.parameters()).device
        if (payload['expert_environment'] != self.environment or self.environment != expert_environment()
                or (payload['rng_state']['cuda'] is None) != (device.type != 'cuda')):
            raise ValueError('anchored resume expert environment/device mismatch')
        for key, expected in (('parent_checkpoint_sha256', self.parent_sha),
                              ('validation_contract', self.validation_contract),
                              ('training_index_sha256', self.index_sha), ('runtime_config', self.runtime),
                              ('model_config', self.model.trifuse.config)):
            if payload[key] != expected:
                raise ValueError(f'anchored resume mismatch: {key}')
        self.model.trifuse.load_state_dict(payload['trifuse_state_dict'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        reference = self.model.trifuse.state_dict()
        self.ema = {key: value.detach().to(reference[key]).clone()
                    for key, value in payload['ema_trifuse_state_dict'].items()}
        self.global_step = payload['global_step']
        self.validation_state = dict(payload['validation_state'])
        torch.set_rng_state(payload['rng_state']['cpu'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(payload['rng_state']['cuda'].cpu(), device)
        return dict(payload['progress'])

    def save(self, path, **progress):
        """Publish a new immutable checkpoint; never replace an existing file."""
        payload = self.checkpoint(**progress)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.anchored-', suffix='.pt', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as handle:
                torch.save(payload, handle)
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
