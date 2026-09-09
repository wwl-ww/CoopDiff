"""Explicit factorial experiments; checkpoints are separate from production v2."""
from contextlib import contextmanager, nullcontext
import os

import torch

from .checkpoint_contract import expert_base_identity, validate_expert_base_identity
from .consensus import ConsensusCoopDiff
from .consensus_train import ConsensusTrainer
from .lora import (expert_lora_state_dict, inject_expert_lora,
                   load_expert_lora_state_dict, set_expert_lora_branch)

EXPERIMENT_ARCHITECTURE = 'residual_consensus_factorial_v1'


def clone_state(state):
    return {name: value.detach().clone() for name, value in state.items()}


class ExperimentalConsensus(ConsensusCoopDiff):
    """Optional independent branch adapters, with immutable pretrained base."""
    architecture = EXPERIMENT_ARCHITECTURE

    def __init__(self, expert, trifuse, lora_config=None):
        super().__init__(expert, trifuse)
        self.experiment_lora_config = dict(lora_config) if lora_config else None
        if lora_config:
            if inject_expert_lora(self.expert, **lora_config) == 0:
                raise ValueError('LoRA configuration selected no layers')
            self.expert.to(next(self.expert.model.parameters()).device).eval()

    def _expert_context(self):
        # Respect outer no_grad during evaluation; never force enable_grad.
        return nullcontext() if self.experiment_lora_config else torch.no_grad()

    def _select_expert_branch(self, index):
        if self.experiment_lora_config:
            set_expert_lora_branch(self.expert, ('sub1', 'sub2')[index])


class ExperimentalConsensusTrainer(ConsensusTrainer):
    """Train and serialize both consensus and adapter EMAs without base weights."""
    def __init__(self, *args, experiment_contract, **kwargs):
        self.experiment_contract = dict(experiment_contract)
        super().__init__(*args, **kwargs)
        self.ema_lora = clone_state(expert_lora_state_dict(self.model.expert))

    def train_step(self, **batch):
        result = super().train_step(**batch)
        with torch.no_grad():
            for name, value in expert_lora_state_dict(self.model.expert).items():
                self.ema_lora[name].lerp_(value.detach(), 1 - self.ema_decay)
        return result

    def _load_adapters(self, state):
        expected = expert_lora_state_dict(self.model.expert)
        if set(state) != set(expected):
            raise ValueError('experimental adapter checkpoint keys differ')
        if any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError('experimental adapter checkpoint shape differs')
        if expected:
            load_expert_lora_state_dict(self.model.expert, state)

    @contextmanager
    def ema_parameters(self):
        raw = clone_state(expert_lora_state_dict(self.model.expert))
        self._load_adapters(self.ema_lora)
        try:
            with super().ema_parameters():
                yield
        finally:
            self._load_adapters(raw)

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
            expert_lora_config=self.model.experiment_lora_config,
            expert_lora_state_dict=expert_lora_state_dict(self.model.expert),
            ema_expert_lora_state_dict=self.ema_lora,
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
                        expert_lora_config=self.model.experiment_lora_config,
                        model_config=self.model.trifuse.config)
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f'experimental checkpoint mismatch: {name}')
        validate_expert_base_identity(payload, self.model.expert)
        self._load_adapters(payload['expert_lora_state_dict'])
        # Validate EMA keys/shapes before accepting it.
        raw = clone_state(expert_lora_state_dict(self.model.expert))
        self._load_adapters(payload['ema_expert_lora_state_dict'])
        self._load_adapters(raw)
        self.model.trifuse.load_state_dict(payload['trifuse_state_dict'])
        self.ema = clone_state(payload['ema_trifuse_state_dict'])
        self.ema_lora = clone_state(payload['ema_expert_lora_state_dict'])
        self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        self.scheduler.load_state_dict(payload['scheduler_state_dict'])
        self.global_step = payload['global_step']
        self.resume_epoch = payload['training_state']['epoch']
        self.resume_best_val_loss = payload['training_state']['best_val_loss']
