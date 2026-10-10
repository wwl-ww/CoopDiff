"""Explicit factorial experiments; checkpoints are separate from production v2."""

import torch

from tricoopdiff.models.expert import validate_expert_base_identity
from tricoopdiff.models.assembly import TernaryAssemblyModel
from tricoopdiff.train import AssemblyTrainer, TriCoopDiffTrainer
from tricoopdiff.runtime import atomic_torch_save

EXPERIMENT_ARCHITECTURE = 'residual_consensus_factorial_v1'


class ExperimentalConsensus(TernaryAssemblyModel):
    """Factorial consensus model with an immutable pretrained expert."""
    architecture = EXPERIMENT_ARCHITECTURE

class ExperimentalConsensusTrainer(AssemblyTrainer):
    """Train and serialize a factorial frozen-expert consensus model."""
    training_objective = 'sdpi_velocity_mse_v1'
    mixed_endpoint_protocol = 'online_bitseed_training_c1_m1_v1'
    compute_loss = TriCoopDiffTrainer.compute_loss

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
            **self._optimization_payload(dict(epoch=epoch, best_val_loss=best_val_loss)))
        atomic_torch_save(payload, path)

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
        self._restore_optimization(payload)
