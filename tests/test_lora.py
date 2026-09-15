"""Lightweight checks for branch-specific expert LoRA adapters."""

import copy
import sys
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
)


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


if __name__ == "__main__":
    test_zero_init_is_transparent_and_branches_are_independent()
    test_parameter_groups_and_adapter_state_roundtrip()
    test_adapter_restore_rejects_missing_tensor()
    print("LoRA tests passed")
