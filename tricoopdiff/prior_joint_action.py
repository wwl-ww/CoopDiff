"""Binary evidence drives one joint torsion/body decision, not another updater."""

import math

import torch
from torch import nn

from .anchored_action_adapter import AnchoredActionAdapter


PRIOR_ACTION_POLICY = 'binary_evidence_joint_action_v1'
PRIOR_ACTION_ARCHITECTURE = 'conditioned_consensus_v8_with_binary_evidence_joint_action_v1'


def prior_energies(ligand, poi, e3, conditions):
    """Whole-map local template errors and coarse receptor overlap; no native input."""
    energies = []
    for side, protein in enumerate((e3, poi), 1):
        evidence = conditions[f'binary{side}'].to(ligand)
        if evidence.ndim != 4 or evidence.shape[1:3] != (len(ligand), len(protein)):
            raise ValueError('binary evidence must retain complete map alternatives and atom order')
        desired = evidence[..., :2].mean(-1)
        confidence = evidence[..., 2]
        weights = confidence * (desired < 10).to(confidence)
        distance = torch.cdist(ligand, protein)
        error = torch.log1p(((distance[None] - desired) / 2).square())
        count = weights.sum((1, 2))
        if not bool((count > 0).any()):
            raise ValueError('joint prior action requires observed local contacts on both sides')
        alternatives = (error * weights).sum((1, 2)) / count.clamp_min(1)
        energies.append(alternatives.masked_fill(count == 0, float('inf')).min())
    # Receptor nodes are coarse-grained, so this is an overlap cue, NOT an
    # all-atom physical clash score or a docking energy.
    overlap = torch.relu(4 - torch.cdist(e3, poi)).square()
    energies.append(overlap.sum() / max(min(len(e3), len(poi)), 1))
    return torch.stack(energies)


def prior_state_features(flow, state, e3, conditions):
    with torch.enable_grad():
        query = state.detach().requires_grad_(True)
        ligand, poi, _ = flow.decode(query)
        energies = prior_energies(ligand, poi, e3.detach(), conditions)
        derivatives = [torch.autograd.grad(value, query, retain_graph=i < 2)[0]
                       for i, value in enumerate(energies)]
    return energies.detach(), torch.stack(derivatives).detach()


class PriorJointActionAdapter(nn.Module):
    """Attention across all bond tokens and four equivariant body-vector tokens.

    Learned scalar weights combine proposal and prior-gradient vectors. No
    componentwise vector nonlinearities or absolute-frame embeddings are used.
    Replaces the previous action module; outputs all K+12 state corrections.
    """

    policy = PRIOR_ACTION_POLICY
    full_state_action = True

    def __init__(self, hidden_dim):
        super().__init__()
        self.bond_input = nn.Sequential(nn.Linear(2 * hidden_dim + 18, hidden_dim), nn.SiLU())
        self.body_input = nn.Sequential(nn.Linear(hidden_dim + 16, hidden_dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, 2 * hidden_dim, dropout=0.,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, flow, state, context, velocity, scales, time):
        count = len(flow.plan.bonds)
        if state.numel() != count + 12:
            raise ValueError('joint binary evidence action requires a two-sided pose chart')
        energies, gradients = prior_state_features(flow, state, context['e3_coords'], context['conditions'])
        body = AnchoredActionAdapter.features(flow, state, context, velocity, scales, time)
        bond_features = torch.cat((body, torch.log1p(energies).expand(count, -1),
                                   torch.tanh(gradients[:, :count].T)), -1)
        bond_tokens = self.bond_input(bond_features)
        proposals = velocity[count:].reshape(4, 3)
        prior = gradients[:, count:].reshape(3, 4, 3).permute(1, 0, 2)
        vectors = torch.cat((proposals[:, None], prior), 1)
        norms = vectors.norm(dim=-1, keepdim=True)
        unit = vectors / (1 + norms)
        correlations = (unit[:, :1] * unit[:, 1:]).sum(-1)
        types = torch.eye(4, dtype=state.dtype, device=state.device)
        scalars = torch.cat((torch.log1p(norms[..., 0]), correlations,
                            torch.log1p(energies).expand(4, -1), types,
                            state.new_full((4, 1), float(time)),
                            state.new_ones((4, 1))), -1)
        body_tokens = self.body_input(torch.cat((context['global'].expand(4, -1), scalars), -1))
        tokens = self.attention(torch.cat((bond_tokens, body_tokens), 0)[None])[0]
        weights = torch.tanh(self.output(tokens))
        torsions = math.pi * weights[:count, 0]
        offsets = (weights[count:, :, None] * unit).sum(1).flatten()
        return torch.cat((torsions, offsets))
