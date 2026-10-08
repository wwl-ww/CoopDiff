"""Training, executed endpoint supervision, EMA and checkpoint contracts.

Variable-size complexes accumulate gradients before one optimizer update.
Production executes real-initialization endpoint supervision. Historical
velocity experiments explicitly use the base trainer.
"""

from contextlib import contextmanager, nullcontext
import math
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tricoopdiff.diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from tricoopdiff.geometry.equivariant_clamp import validate_clamp_mode
from tricoopdiff.models.assembly import (
    ARCHITECTURE, DIRECT_ARCHITECTURE, PREVIOUS_DIRECT_ARCHITECTURE,
    ASSEMBLY_CHECKPOINT_VERSION, EXPERT_CROP_POLICY,
    ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
    OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
    INTERFACE_OBJECTIVE_CHECKPOINT_VERSION, PRODUCTION_MODEL_PROFILE, VECTOR_EXPERT_MODEL_PROFILE,
    production_model_config,
    LEGACY_CONSENSUS_CHECKPOINT_VERSION, PREVIOUS_CONSENSUS_CHECKPOINT_VERSION,
    SOURCE_CONTRACT_VERSION,
    candidate_source_tensor,
)
from tricoopdiff.models.expert import (expert_base_identity, validate_expert_base_identity,
                                    configure_expert_policy, expert_runtime_identity, FrozenMatcha, FrozenFABind)
from tricoopdiff.runtime import (atomic_torch_save, capture_rng_state, evaluation_context,
                                preserve_rng, restore_rng_state)
from tricoopdiff.sdpi_flow import (
    CONTINUOUS_TORSION_MODE, FLOW_VERSION, INDEPENDENT_RIGID_MODE, LEGACY_RIGID_MODE, SDPIFlow,
)


class NonFiniteTrainingError(FloatingPointError):
    """Raised when a training sample cannot produce a valid optimization step."""


LIGAND_GEOMETRY_SCALE_ANGSTROM = 1.0
PROTEIN_GEOMETRY_SCALE_ANGSTROM = 1.0


def clip_grad_norm_finite(parameters, max_norm=1.0):
    """Clip in float64 so large *finite* float32 gradients do not overflow."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0., dtype=torch.float64)
    if any(not torch.isfinite(g).all() for g in gradients):
        raise NonFiniteTrainingError('non-finite trainable gradients')
    norm = torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients]).norm()
    if not torch.isfinite(norm):
        raise NonFiniteTrainingError('non-finite trainable gradient norm')
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return norm


class TriCoopDiffTrainer:
    """
    Trainer for TriCoopDiff model.

    Handles conditional velocity matching and gradient isolation of frozen experts.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: SDPIFlow,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda',
        activation_clamp: float = 100.0,
        torsion_expert_aux_weight: float = 0.25,
        residual_weight: float = 0.0,
    ):
        if not isinstance(diffusion, SDPIFlow):
            raise ValueError('training requires SDPIFlow, not the legacy DSM process')
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        if activation_clamp <= 0:
            raise ValueError("activation_clamp must be positive")
        self.activation_clamp = float(activation_clamp)
        # Numerical settings belong to this expert, not the process or another trainer.
        configure_expert_policy(self.model.expert, self.activation_clamp, 'componentwise')
        if torsion_expert_aux_weight < 0:
            raise ValueError("torsion_expert_aux_weight must be non-negative")
        self.torsion_expert_aux_weight = float(torsion_expert_aux_weight)
        if not math.isfinite(residual_weight) or residual_weight < 0:
            raise ValueError('residual weight must be finite and nonnegative')
        self.residual_weight = float(residual_weight)
        # Only TriFuse is trainable; experts are frozen
        fuse_parameters = list(model.trifuse.parameters())
        if getattr(model.trifuse, 'direct_ternary_motion', False):
            for name in ('judge_head', 'state_head'):
                head = getattr(model.trifuse, name, None)
                if head is not None:
                    head.requires_grad_(False)
            fuse_parameters = [parameter for parameter in fuse_parameters if parameter.requires_grad]
        self._configure_optimizer([{
            'params': fuse_parameters,
            'lr': lr,
            'name': 'trifuse',
        }], weight_decay)

        self.global_step = 0
        self.samples_seen = 0
        self.resume_epoch = 0
        self.resume_best_val_loss = float('inf')

    def _configure_optimizer(self, parameter_groups, weight_decay):
        """Keep optimization and its serialized contract identical across trainers."""
        self.optimizer = AdamW(parameter_groups, weight_decay=weight_decay)
        self.optimizer_config = {
            group['name']: {
                'lr': float(group['lr']),
                'weight_decay': float(group['weight_decay']),
            }
            for group in self.optimizer.param_groups
        }
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7,
        )

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        """Return the state and velocity target used by one loss evaluation."""
        return self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, t,
        )

    def _forward_state(self, e3_data, poi_data, ligand_data,
                       ligand, poi, body, t, dt, source=0, conditions=None):
        """Run the model at an explicitly supplied flow state."""
        return self.diffusion.forward_state(
            self.model, e3_data, poi_data, ligand_data, ligand, poi, body,
            t, dt, source, conditions)

    def compute_loss(
        self, e3_data: dict, poi_data: dict, ligand_data: dict,
        case: str = "unknown", t_override: Optional[torch.Tensor] = None,
        t_pol_override: Optional[torch.Tensor] = None,
        coupling_override=None, diagnostics: bool = False, source=0, conditions=None,
    ) -> dict:
        """Conditional velocity matching; t=0 crystal, t=1 SDPI movable DOFs."""
        if e3_data['coords'].shape[0] != 1:
            raise ValueError("SDPI flow training requires B=1")
        device = self.device
        e3_data = {**e3_data, 'coords': e3_data['coords'].to(device)}
        poi_data = {**poi_data, 'coords': poi_data['coords'].to(device)}
        ligand_data = {**ligand_data, 'coords': ligand_data['coords'].to(device)}
        t = (torch.rand(1, device=device) if t_override is None
             else torch.as_tensor(t_override, device=device, dtype=torch.float32).reshape(-1))
        if t.shape != (1,) or not torch.isfinite(t).all() or not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError("flow time must be one finite value in [0, 1]")
        if t_pol_override is not None:
            other = torch.as_tensor(t_pol_override, device=device, dtype=t.dtype).reshape(-1)
            if other.shape != t.shape or not torch.equal(t, other):
                raise ValueError("SDPI flow requires one shared interface clock")
        coupling = coupling_override if coupling_override is not None else self.diffusion.sample_coupling(
            e3_data['coords'], poi_data['coords'], ligand_data,
            no_torsion=bool(getattr(self.model.expert.model, 'no_torsion', False)),
        )
        dt = t.new_full((1,), 1 / self.diffusion.inference_steps)
        ligand, poi, body, targets = self._training_state(
            e3_data, poi_data, ligand_data, coupling, t, dt)
        output = self._forward_state(
            e3_data, poi_data, ligand_data, ligand, poi, body, t, dt, source, conditions,
        )
        predictions = self.diffusion.normalized_predictions(output, t)
        output_mode = predictions.get('rigid_output_mode', LEGACY_RIGID_MODE)
        if output_mode == INDEPENDENT_RIGID_MODE:
            targets = self.diffusion.independent_targets(targets, ligand, poi)
        elif output_mode != LEGACY_RIGID_MODE:
            raise ValueError(f'unknown rigid training output mode: {output_mode!r}')
        scales = self.diffusion.scales(e3_data['coords'], poi_data['coords'])
        losses = {}
        rms = {}
        def record_rms(name, predicted, target):
            if diagnostics:
                for label, value in (('prediction', predicted), ('target', target)):
                    rms[f'{name}_{label}_rms'] = (
                        value.detach().square().mean().sqrt().item()
                        if value.numel() else 0.0)
        for index, (branch, prefix) in enumerate((('e3_ligand', 'e3l'), ('poi_ligand', 'pol'))):
            for component, scale in (('tr', scales[index]), ('rot', np.pi)):
                predicted = predictions[branch][component]
                target = targets[branch][component] / scale
                if predicted.shape != target.shape:
                    raise ValueError(f"{branch}.{component} shape mismatch: {predicted.shape} vs {target.shape}")
                losses[f'loss_{prefix}_{component}'] = (predicted - target).square().mean()
                record_rms(f'{prefix}_{component}', predicted, target)
        tor_target = targets['torsion'] / np.pi
        if bool(getattr(self.model.expert.model, 'no_torsion', False)):
            tor_target = tor_target[:, :0]
        for name, predicted in (
            ('loss_score_tor', predictions['torsion']),
            ('loss_e3l_tor', predictions['e3_ligand']['tor']),
            ('loss_pol_tor', predictions['poi_ligand']['tor']),
        ):
            if predicted.shape != tor_target.shape:
                raise ValueError(f"{name} shape mismatch: {predicted.shape} vs {tor_target.shape}")
            losses[name] = ((predicted - tor_target).square().mean()
                            if predicted.numel() else t.new_zeros(()))
            record_rms(name.removeprefix('loss_'), predicted, tor_target)
        if any(not torch.isfinite(value) for value in losses.values()):
            raise NonFiniteTrainingError(f"case={case}: non-finite SDPI velocity loss")
        total = (losses['loss_e3l_tr'] + losses['loss_e3l_rot']
                 + losses['loss_pol_tr'] + losses['loss_pol_rot'] + losses['loss_score_tor']
                 + self.torsion_expert_aux_weight * 0.5 * (losses['loss_e3l_tor'] + losses['loss_pol_tor']))
        if 'residual_penalty' in output:
            losses['loss_residual'] = output['residual_penalty']
            total = total + self.residual_weight * losses['loss_residual']
        if diagnostics:
            return {'loss_total': total, 'components': losses, 'rms': rms,
                    'rigid_output_mode': output_mode}
        return {'loss_total': total, 'loss_score': total.item(),
                **{name: value.item() for name, value in losses.items()}}

    def train_step(self, **batch) -> dict:
        """Forward + backward + optimizer step."""
        return self.train_batch([batch])[0]

    def train_batch(self, batches) -> list:
        """Mean microbatch gradient, one optimizer step, bounded graph memory."""
        if getattr(self, '_ema_active', False):
            raise RuntimeError('cannot train while temporary EMA weights are active')
        if not batches:
            raise ValueError('training batch must contain at least one complex')
        self.model.train()  # dropout / attention dropout active during training
        self.optimizer.zero_grad(set_to_none=True)
        rng_before = capture_rng_state()
        results = []
        try:
            for batch in batches:
                loss_dict = self.compute_loss(**batch)
                total = loss_dict['loss_total']
                if not torch.isfinite(total):
                    raise NonFiniteTrainingError('non-finite total loss')
                (total / len(batches)).backward()
                results.append({k: v.item() if torch.is_tensor(v) else v
                                for k, v in loss_dict.items()})
                del loss_dict, total
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state(rng_before)
            raise
        trainable = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group['params']
            if parameter.requires_grad
        ]
        try:
            grad_norm = clip_grad_norm_finite(trainable, max_norm=1.0)
        except NonFiniteTrainingError:
            self.optimizer.zero_grad(set_to_none=True)
            restore_rng_state(rng_before)
            raise
        self.optimizer.step()
        self.global_step += 1
        self.samples_seen += len(batches)
        for result in results:
            result['lr'] = self.optimizer.param_groups[0]['lr']
            result['grad_norm'] = float(grad_norm)
            result['optimizer_batch_size'] = len(batches)
        return results

    def validate_step(self, **batch) -> dict:
        """Forward + loss only, no backward."""
        with evaluation_context(self.model):
            loss_dict = self.compute_loss(**batch)
            return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}


    def scheduler_step(self, validation_loss: float):
        """Update the plateau scheduler from the completed validation pass."""
        self.scheduler.step(validation_loss)

    def set_phase(self, phase: int):
        """
        Switch training phase.

        Only Phase 1 is currently defined. The auxiliary judge/state heads have
        no targets in ``compute_loss`` and must not be advertised as trainable.
        """
        if phase == 1:
            # Freeze judge and state heads
            for name in ('judge_head', 'state_head'):
                head = getattr(self.model.trifuse, name, None)
                if head is not None:
                    head.requires_grad_(False)
        else:
            raise ValueError(
                "only phase=1 is implemented; phase=2 requires explicit "
                "confidence/state labels and losses"
            )


VELOCITY_OBJECTIVE = 'sdpi_velocity_mse_v1'


ENDPOINT_OBJECTIVE = 'executed_endpoint_geometry_v1'
REAL_ENDPOINT_OBJECTIVE = 'real_initialization_reachable_endpoint_v2'
POSE_OUTPUT_CONTRACT = {'version': 'topological_joint_legal_pose_v5',
                        'translation_unit_angstrom': 10.,
                        'rotation': 'bounded_axis_angle', 'torsion': 'bounded_periodic_increment',
                        'execution': 'remaining_time_reverse_velocity'}
ENDPOINT_OBJECTIVES = (ENDPOINT_OBJECTIVE, REAL_ENDPOINT_OBJECTIVE)


REAL_VALIDATION_DOCKQ_PROTOCOL = 'case_fixed_real_initialization_protein_fullatom_v4'
NEAR_TARGET_SUPERVISION = {
    'weight': .25, 'start_times': [1., .2], 'translation_std_angstrom': .5,
    'rotation_std_radians': .05, 'torsion_std_radians': .05,
    'entry_selection': 'source_zero_clock_orthogonal_cycle8_v1',
}
LEGACY_NEAR_TARGET_SUPERVISION = {
    **{key: value for key, value in NEAR_TARGET_SUPERVISION.items()
       if key not in ('start_times', 'entry_selection')},
    'start_t': .2, 'entry_selection': 'sample_parity_zero_or_perturbed_v1',
}


def terminal_interface_losses(e3, native_poi, predicted_poi, cutoff):
    """Native C-alpha interface coordinate/distance surrogates, not DockQ."""
    with torch.no_grad():
        target_distances = torch.cdist(e3[0], native_poi[0], compute_mode='donot_use_mm_for_euclid_dist')
        contacts = target_distances <= cutoff
        poi_interface = contacts.any(dim=0)
    count = int(contacts.sum())
    if not count:
        zero = predicted_poi.sum() * 0
        return {'loss_interface_geometry': zero, 'loss_contact_geometry': zero}, 0
    e3_indices, poi_indices = contacts.nonzero(as_tuple=True)
    predicted_distances = torch.linalg.vector_norm(
        e3[0, e3_indices] - predicted_poi[0, poi_indices], dim=-1)
    target_distances = torch.linalg.vector_norm(
        e3[0, e3_indices] - native_poi[0, poi_indices], dim=-1)
    return {
        'loss_interface_geometry': ((predicted_poi[0, poi_interface]
                                    - native_poi[0, poi_interface]) / cutoff).square().mean(),
        'loss_contact_geometry': ((predicted_distances - target_distances) / cutoff).square().mean(),
    }, count


def coarse_intercomponent_repulsion(e3, native_poi, native_ligand, poi, ligand, cutoff=2.):
    """Conservative short-range CA/ligand penalty, not all-atom sterics.

    Native distances cap each exclusion threshold, preserving zero loss at the
    target even when a coarse representation legitimately has close nodes.
    Worst penetration per node avoids dilution by many distant node pairs.
    """
    penalties = []
    for fixed, target, left, right in ((e3, native_poi, e3, poi),
                                       (e3, native_ligand, e3, ligand),
                                       (native_poi, native_ligand, poi, ligand)):
        with torch.no_grad():
            threshold = torch.cdist(fixed[0], target[0], compute_mode='donot_use_mm_for_euclid_dist').clamp_max(cutoff)
        distances = torch.cdist(left[0], right[0], compute_mode='donot_use_mm_for_euclid_dist')
        penetration = ((threshold - distances).clamp_min(0) / cutoff).square()
        penalties.append((penetration.amax(0).mean() + penetration.amax(1).mean()) / 2)
    return torch.stack(penalties).mean()


def validate_assembly_checkpoint(payload):
    version = payload.get('checkpoint_version')
    if not isinstance(version, int) or not LEGACY_CONSENSUS_CHECKPOINT_VERSION <= version <= ASSEMBLY_CHECKPOINT_VERSION:
        raise ValueError(f'consensus requires checkpoint_version=18 through {ASSEMBLY_CHECKPOINT_VERSION}')
    if payload.get('architecture') not in (ARCHITECTURE, DIRECT_ARCHITECTURE,
                                           PREVIOUS_DIRECT_ARCHITECTURE) or payload.get('generative_process') != FLOW_VERSION:
        raise ValueError('checkpoint does not describe a supported consensus SDPI flow')
    model_config = payload.get('model_config')
    if not isinstance(model_config, dict):
        raise ValueError('checkpoint model_config must be a mapping')
    direct = model_config.get('direct_ternary_motion', False)
    for key in ('global_node_context', 'spatial_geometry', 'motion_attention',
                'global_node_attention', 'rotation_scale_normalization'):
        if model_config.get(key, False) and not direct:
            raise ValueError(f'{key} requires direct ternary motion')
    if model_config.get('model_profile') is not None:
        for key in ('node_dim', 'input_dim'):
            value = model_config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'production checkpoint requires a positive integer {key}')
        if model_config['model_profile'] not in ('ternary_endpoint_v1', 'ternary_endpoint_v2',
                                               'ternary_endpoint_v3', PRODUCTION_MODEL_PROFILE,
                                               VECTOR_EXPERT_MODEL_PROFILE):
            raise ValueError('unknown checkpoint model profile')
        expected = production_model_config(model_config['node_dim'], model_config['input_dim'],
                                           model_config.get('source_conditioning', False),
                                           model_profile=model_config['model_profile'],
                                           expert_scalar_dim=model_config.get('expert_scalar_dim', 58),
                                           expert_vector_dim=model_config.get('expert_vector_dim', 20),
                                           expert_backend=model_config.get('expert_backend', 'diffdock'),
                                           **({'conditioning_contract': model_config['conditioning_contract']}
                                              if 'conditioning_contract' in model_config else {}))
        if model_config != expected:
            raise ValueError('checkpoint violates its production model profile')
    expected_architecture = (DIRECT_ARCHITECTURE if model_config.get('model_profile') == VECTOR_EXPERT_MODEL_PROFILE
                             else PREVIOUS_DIRECT_ARCHITECTURE if direct else ARCHITECTURE)
    if payload.get('architecture') != expected_architecture:
        raise ValueError('checkpoint architecture and rigid output configuration disagree')
    if direct and (version < ENDPOINT_PROTOCOL_CHECKPOINT_VERSION or
                   payload.get('model_config', {}).get('protein_pair_motion', False)):
        raise ValueError('direct ternary motion requires version 20+ without the legacy side head')
    if payload.get('expert_lora_config') or payload.get('expert_lora_state_dict'):
        raise ValueError('Stage 1 consensus checkpoint cannot contain expert adapters')
    for key in ('trifuse_state_dict', 'ema_trifuse_state_dict', 'model_config', 'expert_base_identity'):
        if not payload.get(key):
            raise ValueError(f'consensus checkpoint missing {key}')
    config = payload.get('runtime_config', {})
    if ('ligand_geometry_scale_angstrom' in config
            and config['ligand_geometry_scale_angstrom'] != LIGAND_GEOMETRY_SCALE_ANGSTROM):
        raise ValueError('unsupported ligand endpoint supervision units')
    if ('protein_geometry_scale_angstrom' in config
            and config['protein_geometry_scale_angstrom'] != PROTEIN_GEOMETRY_SCALE_ANGSTROM):
        raise ValueError('unsupported protein endpoint supervision units')
    if model_config.get('model_profile') == 'ternary_endpoint_v2' and version < 25:
        raise ValueError('ternary endpoint v2 requires checkpoint version 25')
    if model_config.get('model_profile') == 'ternary_endpoint_v3' and version < 26:
        raise ValueError('ternary endpoint v3 requires checkpoint version 26')
    if model_config.get('model_profile') == 'ternary_endpoint_v4' and version < 27:
        raise ValueError('ternary endpoint v4 requires checkpoint version 27')
    if model_config.get('model_profile') == VECTOR_EXPERT_MODEL_PROFILE and version < ASSEMBLY_CHECKPOINT_VERSION:
        raise ValueError(f'ternary endpoint v14 legal pose requires checkpoint version {ASSEMBLY_CHECKPOINT_VERSION}')
    if model_config.get('model_profile') == VECTOR_EXPERT_MODEL_PROFILE:
        if config.get('pose_output_contract') != POSE_OUTPUT_CONTRACT:
            raise ValueError('legal pose output contract differs')
        expert_identity = config.get('expert_runtime_identity', {})
        if model_config.get('expert_backend', 'diffdock') == 'matcha':
            if (expert_identity.get('backend') != 'matcha'
                    or expert_identity != payload['expert_base_identity']
                    or config.get('expert_crop_policy') != FrozenMatcha.crop_policy
                    or expert_identity.get('torsion_contract') != 'canonical_executor_bonds_and_fragments_v2'):
                raise ValueError('Matcha checkpoint requires its own canonical input and frozen-prior identity')
        elif model_config.get('expert_backend') == 'fabind':
            if (expert_identity.get('backend') != 'fabind'
                    or expert_identity != payload['expert_base_identity']
                    or config.get('expert_crop_policy') != FrozenFABind.crop_policy
                    or expert_identity.get('advice_contract') != 'ridge_horizontal_rigid_torsion_tangent_reverse_clock_v1'):
                raise ValueError('FABind checkpoint requires its current-pose frozen-prior identity')
        elif (config.get('expert_crop_policy') != EXPERT_CROP_POLICY
                or expert_identity.get('smooth_cross_edges') is not True
                or expert_identity.get('smooth_edges', False)):
            raise ValueError('current assembly requires cross-edge tapering, unchanged internal expert edges, and a fixed full-receptor node set')
    if version >= 25:
        profile = model_config.get('model_profile')
        new_recipe = profile in ('ternary_endpoint_v2', 'ternary_endpoint_v3', 'ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE)
        near_config = NEAR_TARGET_SUPERVISION if profile in ('ternary_endpoint_v3', 'ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE) else LEGACY_NEAR_TARGET_SUPERVISION
        expected = {'training_recipe': ('endpoint_near_target_v2' if profile in ('ternary_endpoint_v3', 'ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE) else 'endpoint_near_target_v1')
                                       if new_recipe else 'endpoint_only_v1',
                    'ema_update_contract': 'optimizer_step_warmup_v1' if new_recipe else 'optimizer_step_fixed_v1',
                    'near_target_supervision': near_config if new_recipe else None,
                    'validation_weights': 'ema'}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('checkpoint has an incompatible endpoint/EMA training recipe')
        if payload.get('checkpoint_kind') != 'inference_only':
            updates = payload.get('training_state', {}).get('ema_updates')
            if isinstance(updates, bool) or not isinstance(updates, int) or updates != payload.get('global_step'):
                raise ValueError('checkpoint EMA updates must equal successful optimizer updates')
    if version >= INTERFACE_OBJECTIVE_CHECKPOINT_VERSION:
        batch_size = config.get('effective_batch_size')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('version 22+ requires a positive effective batch size')
    objective = config.get('training_objective', VELOCITY_OBJECTIVE)
    if version >= OBJECTIVE_CONTRACT_CHECKPOINT_VERSION and 'training_objective' not in config:
        raise ValueError('version 21+ requires an explicit training objective')
    if payload.get('checkpoint_kind') != 'inference_only' and objective not in (VELOCITY_OBJECTIVE, *ENDPOINT_OBJECTIVES):
        raise ValueError('checkpoint has an unknown training objective')
    if objective in ENDPOINT_OBJECTIVES:
        steps = config.get('inference_steps')
        weight = config.get('endpoint_loss_weight')
        if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
                or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0):
            raise ValueError('endpoint objective requires explicit rollout steps and positive loss weight')
        if version >= INTERFACE_OBJECTIVE_CHECKPOINT_VERSION:
            for key in ('interface_loss_weight', 'interface_cutoff'):
                value = config.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'endpoint training requires valid {key}')
            if config['interface_cutoff'] <= 0:
                raise ValueError('interface cutoff must be positive')
        if version >= 23:
            weight, cutoff = config.get('steric_loss_weight'), config.get('steric_cutoff')
            if (not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0
                    or not isinstance(cutoff, (float, int)) or not math.isfinite(cutoff) or cutoff <= 0):
                raise ValueError('version 23+ requires explicit valid coarse steric settings')
    if 'expert_input_grad' in config and not isinstance(config['expert_input_grad'], bool):
        raise ValueError('expert_input_grad must be an explicit boolean')
    validate_clamp_mode(config.get('activation_clamp_mode', 'componentwise'))
    for key in ('activation_clamp', 'residual_weight', 'ema_decay'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid consensus runtime {key}')
    if config['activation_clamp'] <= 0 or config['residual_weight'] < 0 or not 0 <= config['ema_decay'] < 1:
        raise ValueError('invalid consensus runtime range')
    validate_score_norm_version(config)
    if version >= PREVIOUS_CONSENSUS_CHECKPOINT_VERSION:
        if config.get('flow_time_contract') != 'shared_flow_expert_clock_v1':
            raise ValueError('checkpoint has an unknown flow/expert time contract')
        if config.get('candidate_source_contract') != SOURCE_CONTRACT_VERSION:
            raise ValueError('checkpoint has an unknown candidate source contract')
        if config.get('training_source_mix') not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('checkpoint has an invalid training source mix')
    if version >= ENDPOINT_PROTOCOL_CHECKPOINT_VERSION:
        endpoint_protocol = config.get('training_endpoint_protocol')
        expected = ('none' if config.get('training_source_mix') == 'sdpi_only'
                    else ('online_bitseed_raw_resampled_v2' if objective == REAL_ENDPOINT_OBJECTIVE
                          else 'online_bitseed_training_c1_m1_v1'))
        if endpoint_protocol != expected:
            raise ValueError('checkpoint has an invalid training endpoint protocol')
    if objective == REAL_ENDPOINT_OBJECTIVE and version < 24:
        raise ValueError('real initialization supervision requires checkpoint version 24')
    if version >= 24 and payload.get('checkpoint_kind') != 'inference_only':
        if not payload.get('rng_state'):
            raise ValueError('version 24+ training checkpoint requires complete RNG state')
        if not config.get('expert_runtime_identity'):
            raise ValueError('version 24+ requires the expert runtime identity')
    if version >= 28 and payload.get('checkpoint_kind') != 'inference_only':
        if (payload.get('optimizer_mapping_contract') != 'named_parameters_v1'
                or not isinstance(payload.get('optimizer_parameter_manifest'), dict)):
            raise ValueError('version 28+ requires an optimizer named parameter manifest')


class AssemblyTrainer(TriCoopDiffTrainer):
    """Endpoint trainer without historical objective switches."""

    training_objective = REAL_ENDPOINT_OBJECTIVE
    mixed_endpoint_protocol = 'online_bitseed_raw_resampled_v2'

    def __init__(self, *args, residual_weight=1e-3, ema_decay=0.999,
                 activation_clamp_mode='irrep_norm',
                 training_source_mix='sdpi_only',
                 training_endpoint_protocol='none',
                 endpoint_loss_weight=0.1, interface_loss_weight=1.0,
                 interface_cutoff=10.0, effective_batch_size=1,
                 steric_loss_weight=0.0, steric_cutoff=2.0, **kwargs):
        if not math.isfinite(residual_weight) or residual_weight < 0 or not 0 <= ema_decay < 1:
            raise ValueError('invalid residual regularization or EMA decay')
        kwargs['torsion_expert_aux_weight'] = 0.0
        kwargs['residual_weight'] = residual_weight
        super().__init__(*args, **kwargs)
        if not math.isfinite(endpoint_loss_weight) or endpoint_loss_weight <= 0:
            raise ValueError('endpoint loss weight must be finite and positive')
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        if (not math.isfinite(interface_loss_weight) or interface_loss_weight < 0
                or not math.isfinite(interface_cutoff) or interface_cutoff <= 0):
            raise ValueError('invalid interface supervision configuration')
        if not isinstance(effective_batch_size, int) or effective_batch_size < 1:
            raise ValueError('effective batch size must be a positive integer')
        self.interface_loss_weight = float(interface_loss_weight)
        self.interface_cutoff = float(interface_cutoff)
        if (not math.isfinite(steric_loss_weight) or steric_loss_weight < 0
                or not math.isfinite(steric_cutoff) or steric_cutoff <= 0):
            raise ValueError('invalid coarse steric configuration')
        self.steric_loss_weight = float(steric_loss_weight)
        self.steric_cutoff = float(steric_cutoff)
        self.effective_batch_size = effective_batch_size
        self.resume_best_val_dockq = float('-inf')
        self.activation_clamp_mode = validate_clamp_mode(activation_clamp_mode)
        configure_expert_policy(self.model.expert, self.activation_clamp, self.activation_clamp_mode)
        self.ema_decay = float(ema_decay)
        profile = self.model.trifuse.config.get('model_profile')
        self.training_recipe = ('endpoint_near_target_v2' if profile in ('ternary_endpoint_v3', 'ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE)
                                else 'endpoint_near_target_v1' if profile == 'ternary_endpoint_v2'
                                else 'endpoint_only_v1')
        self.ema_update_contract = ('optimizer_step_warmup_v1'
                                    if self.training_recipe != 'endpoint_only_v1'
                                    else 'optimizer_step_fixed_v1')
        self.near_target_supervision = (dict(NEAR_TARGET_SUPERVISION) if profile in ('ternary_endpoint_v3', 'ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE)
                                       else dict(LEGACY_NEAR_TARGET_SUPERVISION) if profile == 'ternary_endpoint_v2'
                                       else None)
        self.near_target_weight = (self.near_target_supervision['weight']
                                   if self.near_target_supervision is not None else 0.0)
        if training_source_mix not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('training_source_mix must be sdpi_only or sdpi_bitseed_1_1')
        self.training_source_mix = training_source_mix
        expected_protocol = ('none' if training_source_mix == 'sdpi_only'
                             else self.mixed_endpoint_protocol)
        if training_endpoint_protocol != expected_protocol:
            raise ValueError(
                f'{training_source_mix} requires training endpoint protocol '
                f'{expected_protocol}'
            )
        self.training_endpoint_protocol = training_endpoint_protocol
        self.set_phase(1)
        self.ema = {name: value.detach().clone() for name, value in self.model.trifuse.state_dict().items()}

    def compute_loss(self, e3_data, poi_data, ligand_data, case='unknown',
                     t_override=None, t_pol_override=None, coupling_override=None,
                     diagnostics=False, source=0, conditions=None, near_target_zero=None,
                     near_target_start_t=None):
        if near_target_start_t is not None and near_target_zero is None:
            raise ValueError('near-target start time requires an auxiliary entry')
        if near_target_zero is not None:
            if not isinstance(near_target_zero, bool) or not self.near_target_weight:
                raise ValueError('near-target entry requires the new recipe and an explicit zero/perturbed flag')
            start_times = (self.near_target_supervision['start_times']
                           if 'start_times' in self.near_target_supervision else [self.near_target_supervision['start_t']])
            start_t = (start_times[0]
                       if near_target_start_t is None else float(near_target_start_t))
            if not math.isfinite(start_t) or start_t not in start_times:
                raise ValueError('near-target start time must match the recorded training clocks')
        for override in (t_override, t_pol_override):
            if override is not None:
                value = torch.as_tensor(override).reshape(-1)
                if value.shape != (1,) or not torch.equal(value, torch.ones_like(value)):
                    raise ValueError('executed endpoint training starts at t=1, not a random teacher time')
        ed = {**e3_data, 'coords': e3_data['coords'].to(self.device)}
        pd = {**poi_data, 'coords': poi_data['coords'].to(self.device)}
        ld = {**ligand_data, 'coords': ligand_data['coords'].to(self.device)}
        no_torsion = bool(getattr(self.model.expert.model, 'no_torsion', False))
        dt = ld['coords'].new_full((1,), 1 / self.diffusion.inference_steps)
        projection = None
        source = int(candidate_source_tensor(source, dt).item())
        from tricoopdiff.initialization import TernaryInitializer
        from tricoopdiff.initialization.endpoints import reachable_ligand_target
        if coupling_override is None:
            if int(source) != 0:
                raise ValueError('BiTSeed training requires its actual sampled state')
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            ligand, poi, body, _ = TernaryInitializer(no_torsion=no_torsion).initialize(
                ed['coords'][0], pd['coords'][0], ld, seed=seed)
            target, projection = reachable_ligand_target(
                ld, ligand, poi, pd['coords'], no_torsion=no_torsion)
        else:
            if 'initial_state' not in coupling_override:
                raise ValueError('real initialization training rejects crystal-projected couplings')
            state = coupling_override['initial_state']
            ligand, poi, body = [torch.as_tensor(state[key], device=self.device,
                                                dtype=ld['coords'].dtype).reshape(1, -1, 3)
                                 for key in ('ligand', 'poi', 'body_frame')]
            if 'target_ligand' in coupling_override:
                target = coupling_override['target_ligand']
            else:
                target, projection = reachable_ligand_target(
                    ld, ligand, poi, pd['coords'], no_torsion=no_torsion)
        target_ligand = torch.as_tensor(target, device=self.device,
                                        dtype=ld['coords'].dtype).reshape_as(ld['coords'])
        if ligand.shape != ld['coords'].shape or poi.shape != pd['coords'].shape:
            raise ValueError('actual initialization atom counts must match the indexed inputs')
        result = self._endpoint_rollout_loss(
            ed, pd, ld, ligand, poi, body, target_ligand, source, conditions)
        result['loss_endpoint_main'] = result['loss_total']
        if near_target_zero is not None:
            near_ligand, near_poi = target_ligand.detach(), pd['coords'].detach()
            near_body = torch.eye(3, device=self.device, dtype=near_ligand.dtype).unsqueeze(0)
            if not near_target_zero:
                velocities = {
                    'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                    **({'torsion_update_mode': CONTINUOUS_TORSION_MODE}
                       if self.model.trifuse.config.get('model_profile') in ('ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE) else {}),
                    'torsion': near_ligand.new_empty((1, 0 if no_torsion else int(ld['edge_mask'].sum()))).normal_(
                        0, self.near_target_supervision['torsion_std_radians']),
                    **{branch: {'tr': near_ligand.new_empty((1, 3)).normal_(
                                    0, self.near_target_supervision['translation_std_angstrom']),
                                'rot': near_ligand.new_empty((1, 3)).normal_(
                                    0, self.near_target_supervision['rotation_std_radians'])}
                       for branch in ('e3_ligand', 'poi_ligand')},
                }
                with torch.no_grad():
                    near_ligand, near_poi, near_body = self.diffusion.differentiable_step(
                        near_ligand, near_poi, near_body, velocities, dt.new_ones((1,)), ld)
            near = self._endpoint_rollout_loss(
                ed, pd, ld, near_ligand, near_poi, near_body, target_ligand, source,
                conditions, t_start=start_t, steps=max(1, math.ceil(self.diffusion.inference_steps * start_t)))
            result['components']['loss_near_target'] = near['loss_total']
            result['loss_total'] = result['loss_total'] + self.near_target_weight * near['loss_total']
        if not torch.isfinite(result['loss_total']):
            raise NonFiniteTrainingError(f'case={case}: non-finite executed endpoint loss')
        if diagnostics:
            return {**result, 'rms': {}, 'target_ligand': target_ligand,
                    'target_projection': projection, 'training_objective': self.training_objective}
        return {'loss_total': result['loss_total'], 'loss_score': result['loss_total'].detach().item(),
                'loss_endpoint_main': result['loss_endpoint_main'].detach().item(),
                'native_interface_pairs': result['native_interface_pairs'],
                **{name: value.detach().item() for name, value in result['components'].items()}}

    def _endpoint_rollout_loss(self, ed, pd, ld, ligand, poi, body, target_ligand,
                               source, conditions, t_start=1., steps=None):
        """One geometric objective for both full starts and reachable short starts."""
        for t, dt in self.diffusion.reverse_times(ld['coords'], steps=steps, t_start=t_start):
            output = self._forward_state(ed, pd, ld, ligand, poi, body, t, dt, source, conditions)
            normalized = self.diffusion.normalized_predictions(output, t)
            velocities = self.diffusion.velocities(normalized, ed['coords'], pd['coords'])
            ligand, poi, body = self.diffusion.differentiable_step(
                ligand, poi, body, velocities, dt, ld)
        # A protein-radius denominator can hide a >10 Å ligand error behind a
        # tiny total loss. Ligand accuracy has a fixed physical scale, independent
        # of receptor size; target projection already ensures it is reachable.
        losses = {'loss_ligand_geometry': ((ligand - target_ligand) / LIGAND_GEOMETRY_SCALE_ANGSTROM).square().mean(),
                  'loss_protein_geometry': ((poi - pd['coords']) / PROTEIN_GEOMETRY_SCALE_ANGSTROM).square().mean()}
        total = sum(losses.values())
        interface_pairs = 0
        if self.interface_loss_weight:
            interface, interface_pairs = terminal_interface_losses(
                ed['coords'], pd['coords'], poi, self.interface_cutoff)
            losses.update(interface)
            total = total + self.interface_loss_weight * sum(interface.values())
        if self.steric_loss_weight:
            steric = coarse_intercomponent_repulsion(ed['coords'], pd['coords'], target_ligand,
                                                     poi, ligand, self.steric_cutoff)
            losses['loss_clash_geometry'] = steric
            total = total + self.steric_loss_weight * steric
        total = self.endpoint_loss_weight * total
        ligand_rmsd = (ligand.detach() - target_ligand.detach()).square().sum(-1).mean().sqrt()
        return {'loss_total': total, 'components': losses, 'native_interface_pairs': interface_pairs,
                'ligand_target_rmsd_angstrom': ligand_rmsd,
                'predicted_ligand': ligand, 'predicted_poi': poi}

    def train_batch(self, batches):
        if len(batches) > self.effective_batch_size:
            raise ValueError('microbatch group exceeds the recorded effective batch size')
        if self.near_target_weight:
            if self.training_recipe == 'endpoint_near_target_v2':
                batches = [{**batch, 'near_target_zero': ((self.samples_seen + offset) // 2) % 2 == 0,
                            'near_target_start_t': self.near_target_supervision['start_times'][
                                ((self.samples_seen + offset) // 4) % 2]}
                           for offset, batch in enumerate(batches)]
            else:
                batches = [{**batch, 'near_target_zero': (self.samples_seen + offset) % 2 == 0,
                            'near_target_start_t': self.near_target_supervision['start_t']}
                           for offset, batch in enumerate(batches)]
        result = super().train_batch(batches)
        decay = self.effective_ema_decay
        with torch.no_grad():
            for name, value in self.model.trifuse.state_dict().items():
                if value.is_floating_point():
                    self.ema[name].lerp_(value.detach(), 1 - decay)
                else:
                    self.ema[name].copy_(value)
        return result

    @property
    def effective_ema_decay(self):
        """EMA advances once per optimizer update, never once per microbatch."""
        return (min(self.ema_decay, (1 + self.global_step) / (10 + self.global_step))
                if self.ema_update_contract == 'optimizer_step_warmup_v1' else self.ema_decay)

    @contextmanager
    def ema_parameters(self):
        if getattr(self, '_ema_active', False):
            yield
            return
        raw = {name: value.detach().clone() for name, value in self.model.trifuse.state_dict().items()}
        self._ema_active = True
        try:
            self.model.trifuse.load_state_dict(self.ema)
            yield
        finally:
            self._ema_active = False
            self.model.trifuse.load_state_dict(raw)

    def validate_step(self, dockq_context=None, weights='ema', **batch):
        if any(batch.get(key) is not None for key in ('near_target_zero', 'near_target_start_t')):
            raise ValueError('validation evaluates the main endpoint only, not training-only auxiliary entries')
        if weights not in ('raw', 'ema'):
            raise ValueError('validation weights must be raw or ema')
        if weights == 'raw' and getattr(self, '_ema_active', False):
            raise RuntimeError('raw validation cannot run while temporary EMA weights are active')
        with (self.ema_parameters() if weights == 'ema' else nullcontext()), evaluation_context(self.model):
            output = self.compute_loss(**batch, diagnostics=dockq_context is not None)
            if dockq_context is not None:
                from tricoopdiff.metrics.dockq import compute_fullatom_dockq

                metric = compute_fullatom_dockq(
                    output['predicted_ligand'][0].cpu().numpy(),
                    output['predicted_poi'][0].cpu().numpy(),
                    batch['e3_data']['coords'][0].detach().cpu().numpy(),
                    **dockq_context)
                if not math.isfinite(metric['dockq']) or not 0 <= metric['dockq'] <= 1:
                    raise NonFiniteTrainingError('invalid full-atom validation DockQ')
                output = {'loss_total': output['loss_total'],
                          'ligand_target_rmsd_angstrom': output['ligand_target_rmsd_angstrom'],
                          'loss_endpoint_main': output['loss_endpoint_main'], **output['components'], **metric}
            return {key: value.item() if torch.is_tensor(value) else value for key, value in output.items()}

    def _optimization_payload(self, training_state):
        """Shared trainable state; subclasses own their supervision contracts."""
        if getattr(self, '_ema_active', False):
            raise RuntimeError('cannot save training state while temporary EMA weights are active')
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        return {
            'expert_base_identity': expert_base_identity(self.model.expert),
            'model_config': dict(self.model.trifuse.config),
            'trifuse_state_dict': self.model.trifuse.state_dict(),
            'ema_trifuse_state_dict': self.ema,
            'global_step': self.global_step,
            'training_state': training_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_mapping_contract': 'named_parameters_v1',
            'optimizer_parameter_manifest': {index: names[id(parameter)]
                                             for index, parameter in enumerate(parameters)},
            'scheduler_state_dict': self.scheduler.state_dict(),
            'rng_state': capture_rng_state(),
        }

    def _restore_optimization(self, payload):
        """Validate a staged optimizer/RNG before committing any live state."""
        if getattr(self, '_ema_active', False):
            raise RuntimeError('cannot restore training state while temporary EMA weights are active')
        training_state = payload['training_state']
        counters = (payload['global_step'], training_state['epoch'],
                    training_state.get('samples_seen', payload['global_step']))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in counters):
            raise ValueError('checkpoint training counters must be nonnegative integers')
        if counters[2] < counters[0]:
            raise ValueError('checkpoint samples_seen cannot be less than successful optimizer updates')
        saved_optimizer = payload['optimizer_state_dict']
        parameters = [parameter for group in self.optimizer.param_groups for parameter in group['params']]
        indices = [index for group in saved_optimizer['param_groups'] for index in group['params']]
        if (any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
                or indices != list(range(len(parameters))) or not set(saved_optimizer['state']).issubset(indices)):
            raise ValueError('checkpoint optimizer parameter IDs must match the canonical unique parameter order')
        mapping = payload.get('optimizer_mapping_contract')
        if mapping is not None or payload.get('checkpoint_version', 0) >= 28:
            names = {id(parameter): name for name, parameter in self.model.named_parameters()}
            expected = {index: names[id(parameter)] for index, parameter in enumerate(parameters)}
            if mapping != 'named_parameters_v1' or payload.get('optimizer_parameter_manifest') != expected:
                raise ValueError('checkpoint optimizer named parameter mapping differs from the active model')
        else:
            import warnings
            warnings.warn('legacy optimizer checkpoint has no parameter names; resume trusts registration order',
                          RuntimeWarning, stacklevel=2)
        active = getattr(self.model.trifuse, 'active_state_dict', dict)
        raw, ema = [active(payload[key]) for key in ('trifuse_state_dict', 'ema_trifuse_state_dict')]
        current = self.model.trifuse.state_dict()
        for label, state in (('raw', raw), ('EMA', ema)):
            if set(state) != set(current) or any(
                    not torch.is_tensor(state[name]) or state[name].shape != current[name].shape
                    or not torch.isfinite(state[name]).all() for name in current):
                raise ValueError(f'checkpoint {label} state does not match the finite active model')
            for name in current:
                state[name] = state[name].to(current[name])
                if not torch.isfinite(state[name]).all():
                    raise ValueError(f'checkpoint {label} state overflows the active model dtype')
        optimizer = AdamW([{**group, 'params': list(group['params'])}
                           for group in self.optimizer.param_groups])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        for expected, group in zip(self.optimizer.param_groups, optimizer.param_groups):
            if any(group[key] != expected[key] for key in expected if key not in ('params', 'lr')):
                raise ValueError('checkpoint optimizer hyperparameters do not match the trainer')
            if not math.isfinite(group['lr']) or group['lr'] < 0:
                raise ValueError('checkpoint optimizer learning rate is invalid')
            for parameter in group['params']:
                moment = optimizer.state.get(parameter, {})
                if not moment:
                    continue
                keys = ('exp_avg', 'exp_avg_sq') + (('max_exp_avg_sq',) if group['amsgrad'] else ())
                for key in keys:
                    value = moment.get(key)
                    if (not torch.is_tensor(value) or value.shape != parameter.shape
                            or not torch.isfinite(value).all()
                            or (key != 'exp_avg' and (value < 0).any())):
                        raise ValueError(f'checkpoint optimizer {key} is invalid')
                step = moment.get('step')
                if (not torch.is_tensor(step) or step.numel() != 1
                        or step.dtype == torch.bool or not torch.isfinite(step).all()
                        or step.item() < 0 or step.item() != int(step.item())
                        or step.item() > counters[0]):
                    raise ValueError('checkpoint optimizer step is invalid')
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                     patience=10, min_lr=1e-7)
        saved_scheduler = payload['scheduler_state_dict']
        fresh_scheduler = scheduler.state_dict()
        history = {'best', 'num_bad_epochs', 'cooldown_counter', 'last_epoch', '_last_lr'}
        if (set(saved_scheduler) != set(fresh_scheduler)
                or any(saved_scheduler[key] != value for key, value in fresh_scheduler.items()
                       if key not in history)
                or math.isnan(float(saved_scheduler['best']))
                or any(isinstance(saved_scheduler[key], bool)
                       or not isinstance(saved_scheduler[key], int) or saved_scheduler[key] < 0
                       for key in ('num_bad_epochs', 'cooldown_counter', 'last_epoch'))
                or saved_scheduler['_last_lr'] != [group['lr'] for group in optimizer.param_groups]):
            raise ValueError('checkpoint scheduler state is invalid or inconsistent with optimizer')
        scheduler.load_state_dict(saved_scheduler)
        best_loss = float(training_state['best_val_loss'])
        if math.isnan(best_loss) or best_loss < 0:
            raise ValueError('checkpoint best validation loss is invalid')
        best_dockq = float(training_state.get('best_val_dockq', float('-inf')))
        if best_dockq != float('-inf') and not 0 <= best_dockq <= 1:
            raise ValueError('checkpoint best validation DockQ is invalid')
        if 'rng_state' in payload:
            preserve_rng(restore_rng_state)(payload['rng_state'])
        else:
            import warnings
            warnings.warn('legacy checkpoint has no RNG state; resume is not trajectory-exact',
                          RuntimeWarning, stacklevel=2)
        self.model.trifuse.load_state_dict(raw, strict=True)
        self.ema = {name: value.detach().clone() for name, value in ema.items()}
        self.optimizer, self.scheduler = optimizer, scheduler
        self.global_step, self.resume_epoch, self.samples_seen = counters
        self.resume_best_val_loss = best_loss
        if 'rng_state' in payload:
            restore_rng_state(payload['rng_state'])

    def checkpoint_payload(self, epoch=0, best_val_loss=float('inf'), best_val_dockq=None):
        """Build the shared training/inference payload without writing files."""
        payload = {
            'checkpoint_version': ASSEMBLY_CHECKPOINT_VERSION,
            'architecture': self.model.architecture, 'generative_process': FLOW_VERSION,
            'expert_lora_config': None, 'expert_lora_state_dict': {},
            'runtime_config': {
                'torus_score_norm_version': SCORE_NORM_VERSION,
                'activation_clamp': self.activation_clamp,
                'activation_clamp_mode': self.activation_clamp_mode,
                'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
                'optimizer_groups': self.optimizer_config,
                'flow_time_contract': 'shared_flow_expert_clock_v1',
                'candidate_source_contract': SOURCE_CONTRACT_VERSION,
                'training_source_mix': self.training_source_mix,
                'training_endpoint_protocol': self.training_endpoint_protocol,
                'expert_input_grad': getattr(self.model, 'expert_input_grad', True),
                'training_objective': self.training_objective,
                'endpoint_loss_weight': self.endpoint_loss_weight,
                'ligand_geometry_scale_angstrom': LIGAND_GEOMETRY_SCALE_ANGSTROM,
                'protein_geometry_scale_angstrom': PROTEIN_GEOMETRY_SCALE_ANGSTROM,
                'inference_steps': self.diffusion.inference_steps,
                'interface_loss_weight': self.interface_loss_weight,
                'interface_cutoff': self.interface_cutoff,
                'steric_loss_weight': self.steric_loss_weight,
                'steric_cutoff': self.steric_cutoff,
                'effective_batch_size': self.effective_batch_size,
                'training_recipe': self.training_recipe,
                'pose_output_contract': (dict(POSE_OUTPUT_CONTRACT)
                                         if self.model.trifuse.config.get('model_profile') == VECTOR_EXPERT_MODEL_PROFILE
                                         else None),
                'ema_update_contract': self.ema_update_contract,
                'near_target_supervision': self.near_target_supervision,
                'validation_weights': 'ema',
                'validation_dockq_protocol': REAL_VALIDATION_DOCKQ_PROTOCOL,
                'expert_runtime_identity': expert_runtime_identity(self.model.expert),
                'expert_crop_policy': (getattr(self.model.expert, 'crop_policy', EXPERT_CROP_POLICY)
                                       if self.model.trifuse.config.get('model_profile') == VECTOR_EXPERT_MODEL_PROFILE
                                       else None),
            },
            **self._optimization_payload({'epoch': epoch, 'best_val_loss': best_val_loss,
                               'best_val_dockq': (self.resume_best_val_dockq if best_val_dockq is None
                                                  else best_val_dockq),
                               'samples_seen': self.samples_seen, 'ema_updates': self.global_step}),
        }
        validate_assembly_checkpoint(payload)
        return payload

    def save_checkpoint(self, path, epoch=0, best_val_loss=float('inf'), best_val_dockq=None):
        atomic_torch_save(self.checkpoint_payload(epoch, best_val_loss, best_val_dockq), path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_assembly_checkpoint(payload)
        if payload.get('checkpoint_kind') == 'inference_only':
            raise ValueError('inference-only weights cannot resume the production training objective')
        if payload['runtime_config'].get('ligand_geometry_scale_angstrom') != LIGAND_GEOMETRY_SCALE_ANGSTROM:
            raise ValueError('historical ligand loss used protein-radius units; load weights only, not resume')
        if payload['runtime_config'].get('protein_geometry_scale_angstrom') != PROTEIN_GEOMETRY_SCALE_ANGSTROM:
            raise ValueError('historical protein loss used protein-radius units; load weights only, not resume')
        if (payload['checkpoint_version'] not in (24, 25, 26, 27, 39, ASSEMBLY_CHECKPOINT_VERSION)
                or payload['runtime_config'].get('training_objective') != REAL_ENDPOINT_OBJECTIVE):
            raise ValueError('training objective differs or checkpoint is historical; load weights only')
        for key in ('inference_steps', 'endpoint_loss_weight', 'interface_loss_weight',
                    'interface_cutoff', 'steric_loss_weight', 'steric_cutoff',
                    'effective_batch_size'):
            expected = self.diffusion.inference_steps if key == 'inference_steps' else getattr(self, key)
            if payload['runtime_config'].get(key) != expected:
                raise ValueError(f'checkpoint {key} differs from the trainer')
        validate_expert_base_identity(payload, self.model.expert)
        if payload['runtime_config']['expert_runtime_identity'] != expert_runtime_identity(self.model.expert):
            raise ValueError('checkpoint expert runtime configuration differs')
        saved_grad = payload['runtime_config'].get('expert_input_grad', False)
        if saved_grad != getattr(self.model, 'expert_input_grad', True):
            raise ValueError('expert input-gradient contract differs; use matching '
                             'expert_input_grad for resume or load weights only for a new run')
        saved_model_config = dict(payload['model_config'])
        if saved_model_config != self.model.trifuse.config:
            raise ValueError('consensus model configuration differs from checkpoint')
        saved_mode = payload['runtime_config'].get('activation_clamp_mode', 'componentwise')
        if saved_mode != self.activation_clamp_mode:
            raise ValueError(f'checkpoint activation_clamp_mode={saved_mode!r}; '
                             'construct the trainer with that explicit mode to resume')
        expected = {'activation_clamp': self.activation_clamp,
                    'residual_weight': self.residual_weight, 'ema_decay': self.ema_decay,
                    'optimizer_groups': self.optimizer_config}
        expected.update({
            'flow_time_contract': 'shared_flow_expert_clock_v1',
            'candidate_source_contract': SOURCE_CONTRACT_VERSION,
            'training_source_mix': self.training_source_mix,
            'training_endpoint_protocol': self.training_endpoint_protocol,
        })
        if payload['checkpoint_version'] >= 25:
            expected.update(training_recipe=self.training_recipe,
                            ema_update_contract=self.ema_update_contract,
                            near_target_supervision=self.near_target_supervision)
        for key, value in expected.items():
            if payload['runtime_config'].get(key) != value:
                raise ValueError(f'consensus checkpoint runtime mismatch: {key}')
        self._restore_optimization(payload)
        self.resume_best_val_dockq = payload['training_state'].get('best_val_dockq', float('-inf'))
        if payload['runtime_config'].get('validation_dockq_protocol') != REAL_VALIDATION_DOCKQ_PROTOCOL:
            # Keep optimizer/LR, but do not compare different validation protocols.
            self.resume_best_val_dockq = float('-inf')
            self.resume_best_val_loss = float('inf')
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                              patience=10, min_lr=1e-7)
