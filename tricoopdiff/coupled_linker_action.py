"""Joint linker action tokens with explicit periodic and cross-bond geometry."""

import math

import torch
from torch import nn

from .anchored_action_adapter import AnchoredActionAdapter
from .linker_action_geometry import linker_geometry_features


COUPLED_ADAPTER_POLICY = 'anchored_unified_linker_action_v3'
COUPLED_ADAPTER_ARCHITECTURE = 'conditioned_consensus_v8_with_unified_linker_action_v3'


class CoupledLinkerActionAdapter(nn.Module):
    """One action head jointly reads body motion and cross-bond geometry.

    Bond tokens have no arbitrary sequence-index embedding. Learned attention
    biases encode relative axes, distance, moved-subtree overlap and POI motion.
    Geometry is invariant; the decoder remains the only coordinate updater.
    """

    policy = COUPLED_ADAPTER_POLICY

    def __init__(self, hidden_dim):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError('coupled action width must be divisible by four')
        self.coupled_input = nn.Sequential(nn.Linear(2 * hidden_dim + 21, hidden_dim), nn.SiLU())
        self.coupled_pair = nn.Sequential(nn.Linear(4, hidden_dim // 4), nn.SiLU(),
                                          nn.Linear(hidden_dim // 4, 4))
        self.coupled_attention = nn.ModuleList([nn.MultiheadAttention(hidden_dim, 4, dropout=0., batch_first=True)
                                               for _ in range(2)])
        self.coupled_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(4)])
        self.coupled_ff = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
                                                       nn.Linear(2 * hidden_dim, hidden_dim)) for _ in range(2)])
        self.coupled_output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.coupled_output.weight)
        nn.init.zeros_(self.coupled_output.bias)

    def forward(self, flow, state, context, velocity, scales, time):
        count = len(flow.plan.bonds)
        if not count:
            return velocity.new_zeros(0)
        coordinates = flow.decode(state)[0].detach()
        scale = torch.as_tensor(scales, device=state.device, dtype=state.dtype).mean()
        local, pair = linker_geometry_features(coordinates, flow.plan, context['edge_index'], scale)
        body_features = AnchoredActionAdapter.features(flow, state, context, velocity, scales, time)
        tokens = self.coupled_input(torch.cat((body_features, local), -1))[None]
        bias = self.coupled_pair(pair).permute(2, 0, 1)
        for index, attention in enumerate(self.coupled_attention):
            normalized = self.coupled_norm[2 * index](tokens)
            attended, _ = attention(normalized, normalized, normalized, attn_mask=bias, need_weights=False)
            tokens = tokens + attended
            tokens = tokens + self.coupled_ff[index](self.coupled_norm[2 * index + 1](tokens))
        return math.pi * torch.tanh(self.coupled_output(tokens)[0, :, 0])
