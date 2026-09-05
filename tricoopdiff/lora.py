"""Branch-specific LoRA adapters for the shared DiffDock expert."""

import math

import torch
import torch.nn as nn


class DualBranchLoRALinear(nn.Module):
    """Frozen linear layer plus independent E3-L and POI-L low-rank updates."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float,
                 layer_index: int):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.layer_index = layer_index
        self.active_branch = "sub1"
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_a = nn.ModuleDict()
        self.lora_b = nn.ModuleDict()
        for branch in ("sub1", "sub2"):
            self.lora_a[branch] = nn.Linear(base.in_features, rank, bias=False)
            self.lora_b[branch] = nn.Linear(rank, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_a[branch].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b[branch].weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_b[self.active_branch](self.lora_a[self.active_branch](inputs))
        return self.base(inputs) + update * self.scale


def _replace_linears(module: nn.Module, rank: int, alpha: float,
                     layer_index: int) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, DualBranchLoRALinear(
                child, rank, alpha, layer_index,
            ))
            count += 1
        else:
            count += _replace_linears(child, rank, alpha, layer_index)
    return count


def inject_expert_lora(expert, rank: int = 4, alpha: float = 8.0,
                       start_layer: int = 0, early_layers: int = 4,
                       early_rank: int = 2, early_alpha: float = 4.0) -> int:
    """Add independent adapters to every selected expert convolution layer.

    Early layers use a smaller rank so they can adapt long-PROTAC messages
    without having the same capacity to overwrite pretrained representations as
    the later layers where the activation runaway becomes visible.
    """
    count = 0
    for layer_index, layer in enumerate(expert.model.conv_layers):
        if layer_index < start_layer:
            continue
        layer_rank = early_rank if layer_index < early_layers else rank
        layer_alpha = early_alpha if layer_index < early_layers else alpha
        count += _replace_linears(
            layer.fc, layer_rank, layer_alpha, layer_index,
        )
    expert.lora_enabled = count > 0
    expert.lora_config = {
        "rank": rank,
        "alpha": alpha,
        "start_layer": start_layer,
        "early_layers": early_layers,
        "early_rank": early_rank,
        "early_alpha": early_alpha,
    }
    return count


def set_expert_lora_branch(expert, branch: str) -> None:
    """Select the adapter used by subsequent calls through the shared expert."""
    if branch not in ("sub1", "sub2"):
        raise ValueError(f"unknown LoRA branch: {branch}")
    for module in expert.modules():
        if isinstance(module, DualBranchLoRALinear):
            module.active_branch = branch


def expert_lora_parameter_groups(expert, early_layers: int = 4):
    """Return early/late adapter parameters as separate optimizer groups."""
    groups = {"early": [], "late": []}
    for module in expert.modules():
        if not isinstance(module, DualBranchLoRALinear):
            continue
        group = "early" if module.layer_index < early_layers else "late"
        for name, parameter in module.named_parameters():
            if name.startswith(("lora_a.", "lora_b.")):
                groups[group].append(parameter)
    return groups


def expert_lora_state_dict(expert):
    """Serialize adapter tensors without duplicating frozen expert weights."""
    return {
        name: tensor for name, tensor in expert.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_expert_lora_state_dict(expert, state_dict):
    """Restore adapter-only state while leaving the pretrained base untouched."""
    incompatible = expert.load_state_dict(state_dict, strict=False)
    unexpected = [name for name in incompatible.unexpected_keys
                  if ".lora_a." in name or ".lora_b." in name]
    if unexpected:
        raise RuntimeError(f"unexpected LoRA checkpoint keys: {unexpected}")
