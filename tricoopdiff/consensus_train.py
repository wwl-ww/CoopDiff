"""Stage 1 training: frozen binary priors, residual consensus and EMA validation."""

from contextlib import contextmanager
import math
import os

import torch

from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .consensus import (
    ARCHITECTURE, DIRECT_ARCHITECTURE, CONSENSUS_CHECKPOINT_VERSION,
    ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
    OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
    INTERFACE_OBJECTIVE_CHECKPOINT_VERSION, PRODUCTION_MODEL_PROFILE,
    production_model_config,
    LEGACY_CONSENSUS_CHECKPOINT_VERSION, PREVIOUS_CONSENSUS_CHECKPOINT_VERSION,
    SOURCE_CONTRACT_VERSION,
)
from .diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from .sdpi_flow import FLOW_VERSION, INDEPENDENT_RIGID_MODE
from .equivariant_clamp import validate_clamp_mode
from .train import NonFiniteTrainingError, TriCoopDiffTrainer


VELOCITY_OBJECTIVE = 'sdpi_velocity_mse_v1'
ENDPOINT_OBJECTIVE = 'executed_endpoint_geometry_v1'


def terminal_interface_losses(e3, native_poi, predicted_poi, cutoff):
    """Native C-alpha interface coordinate/distance surrogates, not DockQ."""
    with torch.no_grad():
        target_distances = torch.cdist(e3[0], native_poi[0])
        contacts = target_distances <= cutoff
        poi_interface = contacts.any(dim=0)
    count = int(contacts.sum())
    if not count:
        zero = predicted_poi.sum() * 0
        return {'loss_interface_geometry': zero, 'loss_contact_geometry': zero}, 0
    predicted_distances = torch.cdist(e3[0], predicted_poi[0])
    return {
        'loss_interface_geometry': ((predicted_poi[0, poi_interface]
                                    - native_poi[0, poi_interface]) / cutoff).square().mean(),
        'loss_contact_geometry': ((predicted_distances[contacts]
                                  - target_distances[contacts]) / cutoff).square().mean(),
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
            threshold = torch.cdist(fixed[0], target[0]).clamp_max(cutoff)
        distances = torch.cdist(left[0], right[0])
        penetration = ((threshold - distances).clamp_min(0) / cutoff).square()
        penalties.append((penetration.amax(0).mean() + penetration.amax(1).mean()) / 2)
    return torch.stack(penalties).mean()


def validate_consensus_checkpoint(payload):
    version = payload.get('checkpoint_version')
    if version not in (LEGACY_CONSENSUS_CHECKPOINT_VERSION,
                       PREVIOUS_CONSENSUS_CHECKPOINT_VERSION,
                       ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
                       OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                       INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                       CONSENSUS_CHECKPOINT_VERSION):
        raise ValueError('consensus requires checkpoint_version=18 through 23')
    if payload.get('architecture') not in (ARCHITECTURE, DIRECT_ARCHITECTURE) or payload.get('generative_process') != FLOW_VERSION:
        raise ValueError('checkpoint does not describe a supported consensus SDPI flow')
    direct = payload.get('model_config', {}).get('direct_ternary_motion', False)
    if payload.get('model_config', {}).get('global_node_context', False) and not direct:
        raise ValueError('global node context requires direct ternary motion')
    if payload.get('model_config', {}).get('spatial_geometry', False) and not direct:
        raise ValueError('spatial geometry requires direct ternary motion')
    for key in ('motion_attention', 'global_node_attention', 'rotation_scale_normalization'):
        if payload.get('model_config', {}).get(key, False) and not direct:
            raise ValueError(f'{key} requires direct ternary motion')
    model_config = payload.get('model_config', {})
    if model_config.get('model_profile') is not None:
        if model_config['model_profile'] != PRODUCTION_MODEL_PROFILE:
            raise ValueError('unknown checkpoint model profile')
        expected = production_model_config(model_config['node_dim'], model_config['input_dim'],
                                           model_config['num_transformer_layers'],
                                           model_config.get('source_conditioning', False))
        if model_config != expected:
            raise ValueError('checkpoint violates its production model profile')
    if bool(direct) != (payload.get('architecture') == DIRECT_ARCHITECTURE):
        raise ValueError('checkpoint architecture and rigid output configuration disagree')
    if direct and (version not in (ENDPOINT_PROTOCOL_CHECKPOINT_VERSION, OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                                  INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                                  CONSENSUS_CHECKPOINT_VERSION) or
                   payload.get('model_config', {}).get('protein_pair_motion', False)):
        raise ValueError('direct ternary motion requires version 20 through 23 without the legacy side head')
    if payload.get('expert_lora_config') or payload.get('expert_lora_state_dict'):
        raise ValueError('Stage 1 consensus checkpoint cannot contain expert adapters')
    for key in ('trifuse_state_dict', 'ema_trifuse_state_dict', 'model_config', 'expert_base_identity'):
        if not payload.get(key):
            raise ValueError(f'consensus checkpoint missing {key}')
    config = payload.get('runtime_config', {})
    if version in (INTERFACE_OBJECTIVE_CHECKPOINT_VERSION, CONSENSUS_CHECKPOINT_VERSION):
        batch_size = config.get('effective_batch_size')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('version 22/23 requires a positive effective batch size')
    objective = config.get('training_objective', VELOCITY_OBJECTIVE)
    if version in (OBJECTIVE_CONTRACT_CHECKPOINT_VERSION, INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                   CONSENSUS_CHECKPOINT_VERSION) and 'training_objective' not in config:
        raise ValueError('version 21 through 23 requires an explicit training objective')
    if payload.get('checkpoint_kind') != 'inference_only' and objective not in (VELOCITY_OBJECTIVE, ENDPOINT_OBJECTIVE):
        raise ValueError('checkpoint has an unknown training objective')
    if objective == ENDPOINT_OBJECTIVE:
        steps = config.get('inference_steps')
        weight = config.get('endpoint_loss_weight')
        if (not isinstance(steps, int) or isinstance(steps, bool) or steps < 1
                or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0):
            raise ValueError('endpoint objective requires explicit rollout steps and positive loss weight')
        if version in (INTERFACE_OBJECTIVE_CHECKPOINT_VERSION, CONSENSUS_CHECKPOINT_VERSION):
            for key in ('interface_loss_weight', 'interface_cutoff'):
                value = config.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'endpoint training requires valid {key}')
            if config['interface_cutoff'] <= 0:
                raise ValueError('interface cutoff must be positive')
        if version == CONSENSUS_CHECKPOINT_VERSION:
            weight, cutoff = config.get('steric_loss_weight'), config.get('steric_cutoff')
            if (not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0
                    or not isinstance(cutoff, (float, int)) or not math.isfinite(cutoff) or cutoff <= 0):
                raise ValueError('version 23 requires explicit valid coarse steric settings')
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
    if version in (PREVIOUS_CONSENSUS_CHECKPOINT_VERSION,
                   ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
                   OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                   INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                   CONSENSUS_CHECKPOINT_VERSION):
        if config.get('flow_time_contract') != 'shared_flow_expert_clock_v1':
            raise ValueError('checkpoint has an unknown flow/expert time contract')
        if config.get('candidate_source_contract') != SOURCE_CONTRACT_VERSION:
            raise ValueError('checkpoint has an unknown candidate source contract')
        if config.get('training_source_mix') not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('checkpoint has an invalid training source mix')
    if version in (ENDPOINT_PROTOCOL_CHECKPOINT_VERSION, OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                   INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                   CONSENSUS_CHECKPOINT_VERSION):
        endpoint_protocol = config.get('training_endpoint_protocol')
        expected = ('none' if config.get('training_source_mix') == 'sdpi_only'
                    else 'online_bitseed_training_c1_m1_v1')
        if endpoint_protocol != expected:
            raise ValueError('checkpoint has an invalid training endpoint protocol')


class ConsensusTrainer(TriCoopDiffTrainer):
    def __init__(self, *args, residual_weight=1e-3, ema_decay=0.999,
                 activation_clamp_mode='irrep_norm',
                 training_source_mix='sdpi_only',
                 training_endpoint_protocol='none', training_objective=None,
                 endpoint_loss_weight=0.1, interface_loss_weight=1.0,
                 interface_cutoff=10.0, effective_batch_size=1,
                 steric_loss_weight=0.0, steric_cutoff=2.0, **kwargs):
        if not math.isfinite(residual_weight) or residual_weight < 0 or not 0 <= ema_decay < 1:
            raise ValueError('invalid residual regularization or EMA decay')
        kwargs['torsion_expert_aux_weight'] = 0.0
        super().__init__(*args, **kwargs)
        self.training_objective = training_objective or (
            ENDPOINT_OBJECTIVE if getattr(self.model.trifuse, 'direct_ternary_motion', False)
            else VELOCITY_OBJECTIVE)
        if self.training_objective not in (VELOCITY_OBJECTIVE, ENDPOINT_OBJECTIVE):
            raise ValueError('unsupported training objective')
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
        os.environ['NAN_CLAMP_MODE'] = self.activation_clamp_mode
        self.residual_weight = float(residual_weight)
        self.ema_decay = float(ema_decay)
        if training_source_mix not in ('sdpi_only', 'sdpi_bitseed_1_1'):
            raise ValueError('training_source_mix must be sdpi_only or sdpi_bitseed_1_1')
        self.training_source_mix = training_source_mix
        expected_protocol = ('none' if training_source_mix == 'sdpi_only'
                             else 'online_bitseed_training_c1_m1_v1')
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
                     diagnostics=False, source=0, conditions=None):
        if self.training_objective == VELOCITY_OBJECTIVE:
            return super().compute_loss(
                e3_data, poi_data, ligand_data, case=case, t_override=t_override,
                t_pol_override=t_pol_override, coupling_override=coupling_override,
                diagnostics=diagnostics, source=source, conditions=conditions)
        for override in (t_override, t_pol_override):
            if override is not None:
                value = torch.as_tensor(override).reshape(-1)
                if value.shape != (1,) or not torch.equal(value, torch.ones_like(value)):
                    raise ValueError('executed endpoint training starts at t=1, not a random teacher time')
        ed = {**e3_data, 'coords': e3_data['coords'].to(self.device)}
        pd = {**poi_data, 'coords': poi_data['coords'].to(self.device)}
        ld = {**ligand_data, 'coords': ligand_data['coords'].to(self.device)}
        coupling = coupling_override if coupling_override is not None else self.diffusion.sample_coupling(
            ed['coords'], pd['coords'], ld,
            no_torsion=bool(getattr(self.model.expert.model, 'no_torsion', False)))
        steps = self.diffusion.inference_steps
        dt = ld['coords'].new_full((1,), 1 / steps)
        ligand, poi, body, _ = self._training_state(ed, pd, ld, coupling, torch.ones_like(dt), dt)
        for index in range(steps):
            t = dt.new_full((1,), 1 - index / steps)
            output = self._forward_state(ed, pd, ld, ligand, poi, body, t, dt, source, conditions)
            normalized = self.diffusion.normalized_predictions(output, t)
            if (getattr(self.model.trifuse, 'direct_ternary_motion', False)
                    and normalized.get('rigid_output_mode') != INDEPENDENT_RIGID_MODE):
                raise ValueError('direct model must execute independent centroid motion')
            velocities = self.diffusion.velocities(normalized, ed['coords'], pd['coords'])
            ligand, poi, body = self.diffusion.differentiable_step(
                ligand, poi, body, velocities, dt, ld)
        scales = self.diffusion.scales(ed['coords'], pd['coords'])
        losses = {'loss_ligand_geometry': ((ligand - ld['coords']) / scales[0]).square().mean(),
                  'loss_protein_geometry': ((poi - pd['coords']) / scales[1]).square().mean()}
        total = sum(losses.values())
        interface_pairs = 0
        if self.interface_loss_weight:
            interface, interface_pairs = terminal_interface_losses(
                ed['coords'], pd['coords'], poi, self.interface_cutoff)
            losses.update(interface)
            total = total + self.interface_loss_weight * sum(interface.values())
        if self.steric_loss_weight:
            steric = coarse_intercomponent_repulsion(ed['coords'], pd['coords'], ld['coords'],
                                                     poi, ligand, self.steric_cutoff)
            losses['loss_clash_geometry'] = steric
            total = total + self.steric_loss_weight * steric
        total = self.endpoint_loss_weight * total
        if not torch.isfinite(total):
            raise NonFiniteTrainingError(f'case={case}: non-finite executed endpoint loss')
        if diagnostics:
            return {'loss_total': total, 'components': losses, 'rms': {},
                    'predicted_ligand': ligand, 'predicted_poi': poi,
                    'training_objective': self.training_objective}
        return {'loss_total': total, 'loss_score': total.detach().item(),
                'native_interface_pairs': interface_pairs,
                **{name: value.detach().item() for name, value in losses.items()}}

    def train_batch(self, batches):
        if len(batches) > self.effective_batch_size:
            raise ValueError('microbatch group exceeds the recorded effective batch size')
        result = super().train_batch(batches)
        with torch.no_grad():
            for name, value in self.model.trifuse.state_dict().items():
                if value.is_floating_point():
                    self.ema[name].lerp_(value.detach(), 1 - self.ema_decay)
                else:
                    self.ema[name].copy_(value)
        return result

    @contextmanager
    def ema_parameters(self):
        if getattr(self, '_ema_active', False):
            yield
            return
        raw = {name: value.detach().clone() for name, value in self.model.trifuse.state_dict().items()}
        self.model.trifuse.load_state_dict(self.ema)
        self._ema_active = True
        try:
            yield
        finally:
            self._ema_active = False
            self.model.trifuse.load_state_dict(raw)

    @torch.no_grad()
    def validate_step(self, dockq_context=None, **batch):
        with self.ema_parameters():
            if dockq_context is not None:
                if self.training_objective != ENDPOINT_OBJECTIVE:
                    raise ValueError('terminal DockQ validation requires the endpoint objective')
                from .deepternary_metrics import compute_fullatom_dockq

                self.model.eval()
                output = self.compute_loss(**batch, diagnostics=True)
                metric = compute_fullatom_dockq(
                    output['predicted_ligand'][0].cpu().numpy(),
                    output['predicted_poi'][0].cpu().numpy(),
                    batch['e3_data']['coords'][0].detach().cpu().numpy(),
                    **dockq_context)
                if not math.isfinite(metric['dockq']) or not 0 <= metric['dockq'] <= 1:
                    raise NonFiniteTrainingError('invalid full-atom validation DockQ')
                return {'loss_total': float(output['loss_total']),
                        **{key: float(value) for key, value in output['components'].items()},
                        **metric}
            return super().validate_step(**batch)

    def checkpoint_payload(self, epoch=0, best_val_loss=float('inf'), best_val_dockq=None):
        """Build the shared training/inference payload without writing files."""
        payload = {
            'checkpoint_version': CONSENSUS_CHECKPOINT_VERSION,
            'architecture': self.model.architecture, 'generative_process': FLOW_VERSION,
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
                'training_endpoint_protocol': self.training_endpoint_protocol,
                'expert_input_grad': getattr(self.model, 'expert_input_grad', True),
                'training_objective': self.training_objective,
                'endpoint_loss_weight': self.endpoint_loss_weight,
                'inference_steps': self.diffusion.inference_steps,
                'interface_loss_weight': self.interface_loss_weight,
                'interface_cutoff': self.interface_cutoff,
                'steric_loss_weight': self.steric_loss_weight,
                'steric_cutoff': self.steric_cutoff,
                'effective_batch_size': self.effective_batch_size,
            },
            'trifuse_state_dict': self.model.trifuse.state_dict(),
            'ema_trifuse_state_dict': self.ema,
            'global_step': self.global_step,
            'training_state': {'epoch': epoch, 'best_val_loss': best_val_loss,
                               'best_val_dockq': (self.resume_best_val_dockq if best_val_dockq is None
                                                  else best_val_dockq),
                               'samples_seen': self.samples_seen},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        validate_consensus_checkpoint(payload)
        return payload

    def save_checkpoint(self, path, epoch=0, best_val_loss=float('inf'), best_val_dockq=None):
        payload = self.checkpoint_payload(epoch, best_val_loss, best_val_dockq)
        temporary = f'{path}.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        validate_consensus_checkpoint(payload)
        if payload.get('checkpoint_kind') == 'inference_only':
            raise ValueError('inference-only weights cannot resume the production training objective')
        saved_objective = payload['runtime_config'].get('training_objective', VELOCITY_OBJECTIVE)
        if saved_objective != self.training_objective:
            raise ValueError('training objective differs; load weights only rather than silently changing supervision')
        if payload['runtime_config'].get('inference_steps', 20) != self.diffusion.inference_steps:
            raise ValueError('checkpoint rollout length differs from the trainer')
        if (self.training_objective == ENDPOINT_OBJECTIVE
                and payload['runtime_config']['endpoint_loss_weight'] != self.endpoint_loss_weight):
            raise ValueError('checkpoint endpoint loss weight differs from the trainer')
        if self.training_objective == ENDPOINT_OBJECTIVE:
            for key, old_default in (('interface_loss_weight', 0.0), ('interface_cutoff', 10.0),
                                     ('steric_loss_weight', 0.0), ('steric_cutoff', 2.0)):
                if payload['runtime_config'].get(key, old_default) != getattr(self, key):
                    raise ValueError(f'checkpoint {key} differs from the trainer')
        if payload['runtime_config'].get('effective_batch_size', 1) != self.effective_batch_size:
            raise ValueError('checkpoint effective batch size differs from the trainer')
        validate_expert_base_identity(payload, self.model.expert)
        saved_grad = payload['runtime_config'].get('expert_input_grad', False)
        if saved_grad != getattr(self.model, 'expert_input_grad', True):
            raise ValueError('expert input-gradient contract differs; use matching '
                             'expert_input_grad for resume or load weights only for a new run')
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
        if payload['checkpoint_version'] in (
                PREVIOUS_CONSENSUS_CHECKPOINT_VERSION,
                ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
                OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                INTERFACE_OBJECTIVE_CHECKPOINT_VERSION,
                CONSENSUS_CHECKPOINT_VERSION):
            expected.update({
                'flow_time_contract': 'shared_flow_expert_clock_v1',
                'candidate_source_contract': SOURCE_CONTRACT_VERSION,
                'training_source_mix': self.training_source_mix,
            })
        if payload['checkpoint_version'] in (ENDPOINT_PROTOCOL_CHECKPOINT_VERSION,
                                            OBJECTIVE_CONTRACT_CHECKPOINT_VERSION,
                                            INTERFACE_OBJECTIVE_CHECKPOINT_VERSION, CONSENSUS_CHECKPOINT_VERSION):
            expected['training_endpoint_protocol'] = self.training_endpoint_protocol
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
        self.resume_best_val_dockq = payload['training_state'].get('best_val_dockq', float('-inf'))
        self.samples_seen = payload['training_state'].get('samples_seen', self.global_step)
