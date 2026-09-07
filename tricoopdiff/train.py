"""
Training pipeline for interface-coordinate TriCoopDiff.

Gradient flow (the pretrained expert base stays frozen while independent LoRA
adapters and TriFuse are trainable):
  L_E3L → E3--Ligand expert → sub1 LoRA / FiLM / bridge → TriFuse
  L_POL → POI--Ligand expert → sub2 LoRA / FiLM / bridge → TriFuse
  L_tor → per-bond coordinator → both expert torsion opinions

Current supervised phase:
  Phase 1: Train interface control, FiLM, bridge, torsion coordination, and
  expert adapters. Auxiliary judge/state heads stay frozen because the current
  dataset does not provide their labels.

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
from .dual_consensus import DualConsensusFlow as SDPIFlow, CHECKPOINT_VERSION, FLOW_VERSION, ARCHITECTURE, CONTRACT
from .diffdock.utils.torus import SCORE_NORM_VERSION, validate_score_norm_version
from .checkpoint_contract import expert_base_identity, validate_expert_base_identity


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
        lora_lr: float = 1e-5,
        lora_early_lr: float = 5e-6,
        lora_early_layers: int = 4,
        activation_clamp: float = 100.0,
        torsion_expert_aux_weight: float = 0.25,
        lora_warmup_steps: int = 100,
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
        if torsion_expert_aux_weight < 0:
            raise ValueError("torsion_expert_aux_weight must be non-negative")
        self.torsion_expert_aux_weight = float(torsion_expert_aux_weight)
        if lora_warmup_steps < 0:
            raise ValueError("lora_warmup_steps must be non-negative")
        self.lora_warmup_steps = int(lora_warmup_steps)

        # Only TriFuse is trainable; experts are frozen
        parameter_groups = [{
            'params': list(model.trifuse.parameters()),
            'lr': lr,
            'name': 'trifuse',
        }]
        from .lora import expert_lora_parameter_groups
        lora_groups = expert_lora_parameter_groups(
            model.expert, early_layers=lora_early_layers,
        )
        if lora_groups['early']:
            parameter_groups.append({
                'params': lora_groups['early'],
                'lr': lora_early_lr,
                'name': 'expert_lora_early',
            })
        if lora_groups['late']:
            parameter_groups.append({
                'params': lora_groups['late'],
                'lr': lora_lr,
                'name': 'expert_lora_late',
            })
        self._lora_params = lora_groups['early'] + lora_groups['late']
        self._set_lora_trainable(False)
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

    def _set_lora_trainable(self, trainable: bool):
        for parameter in self._lora_params:
            parameter.requires_grad = bool(trainable)

    def compute_loss(
        self, e3_data: dict, poi_data: dict, ligand_data: dict,
        case: str = "unknown", t_override: Optional[torch.Tensor] = None,
        t_pol_override: Optional[torch.Tensor] = None,
        coupling_override=None, diagnostics: bool = False,
    ) -> dict:
        from .dual_consensus import dual_loss
        return dual_loss(self, e3_data, poi_data, ligand_data, case, t_override,
                         t_pol_override, coupling_override, diagnostics)

    def train_step(self, **batch) -> dict:
        """Forward + backward + optimizer step."""
        self._set_lora_trainable(self.global_step >= self.lora_warmup_steps)
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

    def save_checkpoint(self, path: str, epoch: int = 0,
                        best_val_loss: float = float('inf')):
        """Save model checkpoint.

        Stores TriFuse and adapter-only expert state. The frozen expert base is
        omitted to avoid duplicating the pretrained checkpoint.
        """
        from .lora import expert_lora_state_dict
        lora_state = expert_lora_state_dict(self.model.expert)
        if len(lora_state) != 336:
            raise RuntimeError(
                f"checkpoint contract requires 336 LoRA tensors, got {len(lora_state)}"
            )
        payload = {
            'checkpoint_version': CHECKPOINT_VERSION,
            'generative_process': FLOW_VERSION,
            'architecture': ARCHITECTURE,
            'dual_consensus_contract': CONTRACT.copy(),
            'expert_base_identity': expert_base_identity(self.model.expert),
            'runtime_config': {
                'dual_stage': self.model.dual_stage,
                'torus_score_norm_version': SCORE_NORM_VERSION,
                'activation_clamp': self.activation_clamp,
                'torsion_expert_aux_weight': self.torsion_expert_aux_weight,
                'lora_warmup_steps': self.lora_warmup_steps,
                'optimizer_groups': self.optimizer_config,
            },
            'global_step': self.global_step,
            'training_state': {
                'epoch': int(epoch),
                'best_val_loss': float(best_val_loss),
            },
            'trifuse_state_dict': self.model.trifuse.state_dict(),
            'expert_lora_state_dict': lora_state,
            'expert_lora_config': getattr(
                self.model.expert, 'lora_config', None,
            ),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scheduler_type': type(self.scheduler).__name__,
        }
        temporary_path = f"{path}.tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)

    def load_checkpoint(self, path: str):
        """Resume only the matching v18 dual architecture and runtime contract."""
        ckpt = torch.load(path, map_location=self.device)
        validate_score_norm_version(ckpt.get('runtime_config', {}))
        validate_expert_base_identity(ckpt, self.model.expert)
        if ckpt.get('checkpoint_version') != CHECKPOINT_VERSION:
            raise ValueError(
                "Dual consensus training requires checkpoint_version=18"
            )
        if ckpt.get('generative_process') != FLOW_VERSION:
            raise ValueError('checkpoint generative process is not dual_consensus_flow_v1')
        if ckpt.get('dual_consensus_contract') != CONTRACT:
            raise ValueError('checkpoint dual consensus contract differs')
        if ckpt.get('architecture') != ARCHITECTURE:
            raise ValueError(
                "checkpoint architecture is not dual_consensus_v1"
            )
        saved_lora_config = ckpt.get('expert_lora_config')
        if ckpt.get('runtime_config', {}).get('dual_stage') != self.model.dual_stage:
            raise ValueError('checkpoint dual stage differs; resume must preserve the trained field')
        runtime_lora_config = getattr(self.model.expert, 'lora_config', None)
        if saved_lora_config != runtime_lora_config:
            raise ValueError(
                f"checkpoint expert_lora_config={saved_lora_config!r} does not "
                f"match runtime config={runtime_lora_config!r}"
            )
        if len(ckpt.get('expert_lora_state_dict', {})) != 336:
            raise ValueError("checkpoint must contain exactly 336 LoRA tensors")
        saved_clamp = ckpt.get('runtime_config', {}).get('activation_clamp')
        if saved_clamp != self.activation_clamp:
            raise ValueError(
                f"checkpoint activation_clamp={saved_clamp!r} does not match "
                f"runtime activation_clamp={self.activation_clamp!r}"
            )
        saved_aux_weight = ckpt.get('runtime_config', {}).get(
            'torsion_expert_aux_weight'
        )
        if saved_aux_weight != self.torsion_expert_aux_weight:
            raise ValueError(
                f"checkpoint torsion_expert_aux_weight={saved_aux_weight!r} "
                f"does not match runtime value={self.torsion_expert_aux_weight!r}"
            )
        saved_warmup = ckpt.get('runtime_config', {}).get('lora_warmup_steps')
        if saved_warmup != self.lora_warmup_steps:
            raise ValueError(
                f"checkpoint lora_warmup_steps={saved_warmup!r} does not "
                f"match runtime value={self.lora_warmup_steps!r}"
            )
        saved_optimizer_config = ckpt.get('runtime_config', {}).get(
            'optimizer_groups'
        )
        if saved_optimizer_config != self.optimizer_config:
            raise ValueError(
                f"checkpoint optimizer groups={saved_optimizer_config!r} do not "
                f"match runtime groups={self.optimizer_config!r}"
            )
        self.model.trifuse.load_state_dict(ckpt['trifuse_state_dict'])
        if ckpt.get('expert_lora_state_dict'):
            from .lora import load_expert_lora_state_dict
            load_expert_lora_state_dict(
                self.model.expert, ckpt['expert_lora_state_dict'],
            )
        self.global_step = ckpt.get('global_step', 0)
        training_state = ckpt.get('training_state', {})
        self.resume_epoch = int(training_state.get('epoch', 0))
        self.resume_best_val_loss = float(
            training_state.get('best_val_loss', float('inf'))
        )
        self._set_lora_trainable(self.global_step >= self.lora_warmup_steps)
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if (
            'scheduler_state_dict' in ckpt
            and ckpt.get('scheduler_type') == type(self.scheduler).__name__
        ):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])


# so3_score_norm — imported from tricoopdiff.diffusion where needed
