"""Lightweight checks for branch-specific expert LoRA adapters."""

import copy
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from tricoopdiff.lora import (
    DualBranchLoRALinear,
    expert_lora_parameter_groups,
    expert_lora_state_dict,
    load_expert_lora_state_dict,
    set_expert_lora_branch,
    without_expert_lora,
)
from tricoopdiff.consensus_experiment import ExperimentalConsensusTrainer
from tricoopdiff.sdpi_flow import SDPIFlow


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = DualBranchLoRALinear(nn.Linear(5, 4), 2, 4.0, 0)
        self.late = DualBranchLoRALinear(nn.Linear(4, 3), 4, 8.0, 7)

    def forward(self, inputs):
        return self.late(torch.relu(self.early(inputs)))


def test_zero_init_is_transparent_and_branches_are_independent():
    torch.manual_seed(0)
    expert = TinyExpert()
    inputs = torch.randn(6, 5)
    set_expert_lora_branch(expert, "sub1")
    initial_sub1 = expert(inputs)
    set_expert_lora_branch(expert, "sub2")
    initial_sub2 = expert(inputs)
    assert torch.equal(initial_sub1, initial_sub2), "zero LoRA must be transparent"

    with torch.no_grad():
        expert.early.lora_b["sub1"].weight.fill_(0.25)
    set_expert_lora_branch(expert, "sub1")
    adapted_sub1 = expert(inputs)
    set_expert_lora_branch(expert, "sub2")
    unchanged_sub2 = expert(inputs)
    assert not torch.equal(adapted_sub1, unchanged_sub2), "branch adapters leaked"

    set_expert_lora_branch(expert, "sub1")
    with without_expert_lora(expert):
        base_only = expert(inputs)
        assert all(module.active_branch is None for module in
                   (expert.early, expert.late))
    assert expert.early.active_branch == expert.late.active_branch == "sub1"
    set_expert_lora_branch(expert, "sub2")
    assert torch.equal(base_only, expert(inputs))


def test_parameter_groups_and_adapter_state_roundtrip():
    torch.manual_seed(0)
    expert = TinyExpert()
    groups = expert_lora_parameter_groups(expert, early_layers=4)
    assert len(groups["early"]) == 4, "expected A/B tensors for both early branches"
    assert len(groups["late"]) == 4, "expected A/B tensors for both late branches"

    with torch.no_grad():
        expert.late.lora_b["sub2"].weight.fill_(0.5)
    state = expert_lora_state_dict(expert)
    restored = TinyExpert()
    load_expert_lora_state_dict(restored, copy.deepcopy(state))
    assert torch.equal(
        restored.late.lora_b["sub2"].weight,
        expert.late.lora_b["sub2"].weight,
    ), "LoRA checkpoint roundtrip changed adapter weights"


def test_adapter_restore_rejects_missing_tensor():
    expert = TinyExpert()
    state = expert_lora_state_dict(expert)
    state.pop(next(iter(state)))
    try:
        load_expert_lora_state_dict(TinyExpert(), state)
    except RuntimeError as error:
        assert "missing LoRA checkpoint keys" in str(error)
    else:
        raise AssertionError("incomplete LoRA state must be rejected")


def test_experimental_trainer_preserves_warmup_and_adapter_checkpoint():
    class TinyFuse(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = nn.Linear(2, 2)
            self.judge_head = nn.Linear(2, 1)
            self.state_head = nn.Linear(2, 1)
            self.config = {'tiny': True}

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.expert = TinyExpert()
            self.expert.provenance = {
                'diffdock_commit': 'test',
                'score_checkpoint_sha256': 'abc',
                'no_torsion': False,
            }
            self.experiment_lora_config = {'rank': 4}
            self.trifuse = TinyFuse()

    contract = {'version': 'test_lora'}
    trainer = ExperimentalConsensusTrainer(
        TinyModel(), SDPIFlow(), device='cpu', experiment_contract=contract,
        lora_warmup_steps=17,
    )
    assert trainer.lora_warmup_steps == 17
    assert {group['name'] for group in trainer.optimizer.param_groups} == {
        'trifuse', 'expert_lora_early', 'expert_lora_late',
    }
    assert all(not parameter.requires_grad for parameter in trainer._lora_params)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'lora.pt'
        trainer.save_checkpoint(path, epoch=3, best_val_loss=1.5)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['runtime_config']['lora_warmup_steps'] == 17
        assert payload['expert_lora_state_dict']
        assert payload['ema_expert_lora_state_dict']
        restored = ExperimentalConsensusTrainer(
            TinyModel(), SDPIFlow(), device='cpu', experiment_contract=contract,
            lora_warmup_steps=17,
        )
        restored.load_checkpoint(path)
        assert restored.resume_epoch == 3
        assert restored.resume_best_val_loss == 1.5


if __name__ == "__main__":
    test_zero_init_is_transparent_and_branches_are_independent()
    test_parameter_groups_and_adapter_state_roundtrip()
    test_adapter_restore_rejects_missing_tensor()
    test_experimental_trainer_preserves_warmup_and_adapter_checkpoint()
    print("LoRA tests passed")
