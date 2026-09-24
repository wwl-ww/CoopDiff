"""
Training pipeline for interface-coordinate TriCoopDiff.

Gradient flow (the pretrained expert stays frozen while TriFuse is trainable):
  L_E3L → frozen E3--Ligand expert → TriFuse
  L_POL → frozen POI--Ligand expert → TriFuse
  L_tor → per-bond coordinator → both expert torsion opinions

Current supervised phase:
  Phase 1: Train residual consensus and torsion coordination. Auxiliary
  judge/state heads stay frozen because the dataset has no corresponding labels.

Current implementation uses B=1 sequential optimization; batched variable-size
graphs remain future work.
"""

import os

import torch
import numpy as np
from typing import Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import TriCoopDiff
from .sdpi_flow import SDPIFlow, _rotation_exp
from .rigid import assemble_interface_pair


class NonFiniteTrainingError(FloatingPointError):
    """Raised when a training sample cannot produce a valid optimization step."""


class TriCoopDiffTrainer:
    """
    Trainer for TriCoopDiff model.

    Handles conditional velocity matching and gradient isolation of frozen experts.
    """

    def __init__(
        self,
        model: TriCoopDiff,
        diffusion: SDPIFlow,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = 'cuda',
        activation_clamp: float = 100.0,
        torsion_expert_aux_weight: float = 0.25,
        ab_consistency_weight: float = 0.0,
    ):
        if not isinstance(diffusion, SDPIFlow):
            raise ValueError('training requires SDPIFlow, not the legacy DSM process')
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        if activation_clamp <= 0:
            raise ValueError("activation_clamp must be positive")
        self.activation_clamp = float(activation_clamp)
        # The vendored expert integration currently reads this process-level
        # setting inside each convolution. Configure it at the owning runtime
        # boundary so library callers cannot silently run a different model
        # from the trainer's recorded checkpoint contract.
        os.environ['NAN_CLAMP'] = str(self.activation_clamp)
        os.environ['NAN_CLAMP_MODE'] = 'componentwise'
        if torsion_expert_aux_weight < 0:
            raise ValueError("torsion_expert_aux_weight must be non-negative")
        self.torsion_expert_aux_weight = float(torsion_expert_aux_weight)
        if ab_consistency_weight < 0:
            raise ValueError("ab_consistency_weight must be non-negative")
        self.ab_consistency_weight = float(ab_consistency_weight)
        # Only TriFuse is trainable; experts are frozen
        parameter_groups = [{
            'params': list(model.trifuse.parameters()),
            'lr': lr,
            'name': 'trifuse',
        }]
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

        self.global_step = 0
        self.resume_epoch = 0
        self.resume_best_val_loss = float('inf')

    def _training_state(self, e3_data, poi_data, ligand_data, coupling, t, dt):
        """Return the state and velocity target used by one loss evaluation."""
        return self.diffusion.path(
            ligand_data, poi_data['coords'], coupling, t,
        )

    def _forward_state(self, e3_data, poi_data, ligand_data,
                       ligand, poi, body, t, dt, source=0, conditions=None):
        """Run the model at an explicitly supplied flow state."""
        conditioning = self.diffusion.conditioning(
            t, dt, e3_data['coords'], poi_data['coords'])
        tr_condition, _, _ = self.diffusion.t_to_sigma(t, t, t)
        crop = (tr_condition * 3 + 20
                if bool(getattr(self.model.expert.model, 'dynamic_max_cross', False)) else None)
        condition_kwargs = {'conditions': conditions} if conditions is not None else {}
        return self.model(
            e3_data, {**poi_data, 'coords': poi},
            {**ligand_data, 'coords': ligand},
            t_e3l=t, t_pol=t, r_e3_to_poi=body,
            diffusion_state=conditioning, receptor_crop_cutoff=crop,
            source=source, **condition_kwargs,
        )

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
        # The ordinary flow losses supervise E3--L and POI--L independently.
        # This auxiliary term composes both predicted rigid updates and compares
        # the resulting ternary pose against the composed target update.  It is
        # deliberately optional so existing checkpoints remain reproducible.
        if self.ab_consistency_weight > 0:
            pred_vel = self.diffusion.velocities(predictions, e3_data['coords'], poi_data['coords'])
            target_norm = {
                'e3_ligand': {
                    'tr': targets['e3_ligand']['tr'] / scales[0],
                    'rot': targets['e3_ligand']['rot'] / np.pi,
                },
                'poi_ligand': {
                    'tr': targets['poi_ligand']['tr'] / scales[1],
                    'rot': targets['poi_ligand']['rot'] / np.pi,
                },
                'torsion': tor_target,
            }
            target_vel = self.diffusion.velocities(
                target_norm, e3_data['coords'], poi_data['coords'])

            def compose_body(pose, velocities):
                old_center = pose[0].mean(dim=0, keepdim=True)
                dt_value = 1.0 / float(self.diffusion.inference_steps)
                transforms = []
                for branch in ('e3_ligand', 'poi_ligand'):
                    rot = _rotation_exp(-dt_value * velocities[branch]['rot'])
                    center = old_center - dt_value * velocities[branch]['tr']
                    offset = center - (rot @ old_center.unsqueeze(-1)).squeeze(-1)
                    transforms.append((rot, offset))
                # assemble_interface_pair uses D_A ∘ inverse(D_B), exactly the
                # body-frame update used by SDPIFlow.step().
                _, increment = assemble_interface_pair(
                    poi_data['coords'], *transforms[0], *transforms[1])
                return increment @ body, increment, transforms

            pred_body, pred_increment, pred_transforms = compose_body(ligand, pred_vel)
            target_body, target_increment, target_transforms = compose_body(ligand, target_vel)
            pred_poi, _ = assemble_interface_pair(
                poi, *pred_transforms[0], *pred_transforms[1])
            target_poi, _ = assemble_interface_pair(
                poi, *target_transforms[0], *target_transforms[1])
            # Body rotation captures relative orientation; normalized POI
            # coordinates capture the associated relative translation.
            coord_scale = scales.mean().detach().clamp_min(1.0)
            losses['loss_ab_consistency'] = (
                (pred_body - target_body).square().mean()
                + ((pred_poi - target_poi) / coord_scale).square().mean()
            )
        if any(not torch.isfinite(value) for value in losses.values()):
            raise NonFiniteTrainingError(f"case={case}: non-finite SDPI velocity loss")
        total = (losses['loss_e3l_tr'] + losses['loss_e3l_rot']
                 + losses['loss_pol_tr'] + losses['loss_pol_rot'] + losses['loss_score_tor']
                 + self.torsion_expert_aux_weight * 0.5 * (losses['loss_e3l_tor'] + losses['loss_pol_tor']))
        if 'residual_penalty' in output:
            losses['loss_residual'] = output['residual_penalty']
            total = total + self.residual_weight * losses['loss_residual']
        if 'loss_ab_consistency' in losses:
            total = total + self.ab_consistency_weight * losses['loss_ab_consistency']
        if diagnostics:
            return {'loss_total': total, 'components': losses, 'rms': rms}
        return {'loss_total': total, 'loss_score': total.item(),
                **{name: value.item() for name, value in losses.items()}}

    def train_step(self, **batch) -> dict:
        """Forward + backward + optimizer step."""
        self.model.train()  # dropout / attention dropout active during training
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(**batch)
        total = loss_dict['loss_total']
        if not torch.isfinite(total):
            raise NonFiniteTrainingError("non-finite total loss")
        total.backward()
        trainable = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group['params']
            if parameter.requires_grad
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise NonFiniteTrainingError("non-finite trainable gradients")
        self.optimizer.step()
        self.global_step += 1

        result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        result['lr'] = self.optimizer.param_groups[0]['lr']
        return result

    @torch.no_grad()
    def validate_step(self, **batch) -> dict:
        """Forward + loss only, no backward."""
        self.model.eval()  # dropout / attention dropout off during validation
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
            for param in self.model.trifuse.judge_head.parameters():
                param.requires_grad = False
            for param in self.model.trifuse.state_head.parameters():
                param.requires_grad = False
        else:
            raise ValueError(
                "only phase=1 is implemented; phase=2 requires explicit "
                "confidence/state labels and losses"
            )

# so3_score_norm — imported from tricoopdiff.diffusion where needed
