"""Shared invariant token negotiation and output heads.

Kept independent of the legacy FiLM controller for residual consensus models.
"""

import torch
from torch import nn
from .geometry_encoders import BondContextEncoder


BRANCHES = ('e3_ligand', 'poi_ligand')


def bounded_vector(value):
    """Isotropic vector compression, preserving rotation equivariance."""
    return value / (1 + torch.linalg.vector_norm(value, dim=-1, keepdim=True))


class TokenNegotiator(nn.Module):
    """Let global, interface, linker, and diffusion-state tokens communicate."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float,
                 num_token_types: int = 5):
        super().__init__()
        self.token_types = nn.Parameter(
            torch.randn(num_token_types, hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != self.token_types.shape[0]:
            raise ValueError(
                f"expected [B,{self.token_types.shape[0]},D] tokens, "
                f"got {tuple(tokens.shape)}"
            )
        return self.encoder(tokens + self.token_types.unsqueeze(0))


class JudgeHead(nn.Module):
    """Reserved global confidence head; frozen until labels are available."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),  # scalar confidence
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1] confidence score."""
        return self.confidence_net(x)


class StateHead(nn.Module):
    """
    Predicts thermodynamic/functional state indicators:
    - ΔΔG estimate
    - Productive complex probability
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),  # [ΔΔG, p_productive]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 2] with [ΔΔG_estimate, productive_probability]."""
        return self.net(x)


class TorsionCoordinator(nn.Module):
    """Resolve two interface opinions into one score per rotatable bond.

    A PROTAC has one shared set of torsional degrees of freedom.  The two
    DiffDock experts therefore provide evidence, not independently executable
    torsion actions.  This head sees both per-bond scores together with the
    ternary context and predicts a per-bond mixture weight and residual.

    The final projection is zero-initialized, so an untrained coordinator starts
    at the neutral expert mean while remaining fully learnable.
    """

    def __init__(self, context_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        # Per-bond evidence is bounded before entering the MLP. This prevents
        # large but finite expert scores from overflowing in a raw product.
        self.net = nn.Sequential(
            nn.Linear(context_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        score_e3l: torch.Tensor,
        score_pol: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return coordinated score, E3-side weight, and learned residual."""
        if score_e3l.shape != score_pol.shape:
            raise ValueError(
                "torsion experts must describe the same rotatable bonds: "
                f"{tuple(score_e3l.shape)} != {tuple(score_pol.shape)}"
            )
        if score_e3l.ndim != 2:
            raise ValueError(
                f"torsion scores must have shape [B,K], got {score_e3l.shape}"
            )
        if context.shape[0] != score_e3l.shape[0]:
            raise ValueError("torsion context and scores have different batches")

        n_torsions = score_e3l.shape[1]
        if context.ndim == 2:
            context_per_bond = context.unsqueeze(1).expand(-1, n_torsions, -1)
        elif context.ndim == 3 and context.shape[1] == n_torsions:
            context_per_bond = context
        else:
            raise ValueError(
                "torsion context must have shape [B,D] or [B,K,D], got "
                f"{tuple(context.shape)} for K={n_torsions}"
            )
        bounded_e3l = torch.tanh(score_e3l)
        bounded_pol = torch.tanh(score_pol)
        bond_evidence = torch.stack(
            (
                bounded_e3l,
                bounded_pol,
                bounded_e3l - bounded_pol,
                bounded_e3l * bounded_pol,
            ),
            dim=-1,
        )
        coordination = self.net(
            torch.cat((context_per_bond, bond_evidence), dim=-1)
        )
        weight_e3l = torch.sigmoid(coordination[..., 0])
        residual = 10.0 * torch.tanh(coordination[..., 1])
        score = (
            weight_e3l * score_e3l
            + (1.0 - weight_e3l) * score_pol
            + residual
        )
        return score, weight_e3l, residual


class LegacyResidualMotion:
    """Historical checkpoint behavior, isolated from the direct motion path.

    A mixin keeps historical parameter names unchanged. No legacy head is
    instantiated when the production controller selects independent motion.
    """

    def _init_legacy_motion(self, node_dim):
        self.gate_head = nn.Linear(node_dim, 4)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.)
        self.residual_head = nn.Linear(node_dim, 4 * 8)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.bond_context_encoder = BondContextEncoder(node_dim)
        self.torsion_coordinator = TorsionCoordinator(node_dim)
        self.judge_head = JudgeHead(node_dim).requires_grad_(False)
        self.state_head = StateHead(node_dim).requires_grad_(False)
        if self.protein_pair_motion:
            self.protein_pair_head = nn.Sequential(
                nn.Linear(2 * node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, 4))
            nn.init.zeros_(self.protein_pair_head[-1].weight)
            nn.init.zeros_(self.protein_pair_head[-1].bias)

    def _legacy_motion(self, pooled, graph, proposals, context, basis, raw,
                       condition_gate_bias, source_context):
        coords, types = graph['node_coords'], graph['node_types']
        gate_logits = self.gate_head(context)
        if condition_gate_bias is not None:
            gate_logits = gate_logits + condition_gate_bias
        if source_context is not None:
            gate_logits = gate_logits + self.source_gate_head(source_context)
        gates = torch.sigmoid(gate_logits)
        coefficients = self.residual_head(context).reshape(-1, 4, 8)
        residual = torch.einsum('bik,bkj->bij', coefficients, basis)
        pair_motion = torch.zeros_like(residual)
        if self.protein_pair_motion:
            contacts = pooled['protein_pair_contacts'][0]
            weights = self.protein_pair_head(torch.cat((
                contacts['tokens'], context.expand(len(contacts['tokens']), -1)), dim=-1))
            direction = contacts['direction']
            torque = torch.cross(bounded_vector(contacts['lever']), direction, dim=-1)
            pair_motion = torch.stack([
                (weights[:, i:i+1] * (direction if i % 2 == 0 else torque)).mean(0)
                for i in range(4)], dim=0).unsqueeze(0)
            residual = residual + pair_motion
        prior = gates.unsqueeze(-1) * torch.stack(raw, 1)
        field = prior + residual
        bond_context = self.bond_context_encoder(
            pooled['ligand_tokens'][0], coords[types == 2], graph['lig_edge_index'],
            graph['lig_edge_mask'], graph['lig_mask_rotate'], context)
        tor_a, tor_b = (proposals[b]['tor'] for b in BRANCHES)
        torsion, tor_gate, tor_residual = self.torsion_coordinator(bond_context, tor_a, tor_b)
        predictions = {
            branch: {'tr': field[:, 2 * i], 'rot': field[:, 2 * i + 1],
                     'tor': proposals[branch]['tor']}
            for i, branch in enumerate(BRANCHES)}
        predictions['torsion'] = torsion
        residual_penalty = residual.square().mean()
        if tor_residual.numel():
            residual_penalty = residual_penalty + tor_residual.square().mean()
        conflicts = torch.stack([
            1 - torch.nn.functional.cosine_similarity(raw[i], raw[i + 2], dim=-1)
            for i in (0, 1)], dim=-1)
        return {
            'normalized_velocity': predictions, 'rank_features': context,
            'residual_penalty': residual_penalty,
            'consensus_diagnostics': {
                'rigid_gates': gates.detach(), 'direction_disagreement': conflicts.detach(),
                'rigid_prior_rms': prior.detach().square().mean().sqrt(),
                'rigid_residual_rms': residual.detach().square().mean().sqrt(),
                'protein_pair_motion_rms': pair_motion.detach().square().mean().sqrt(),
                'torsion_gate': tor_gate.detach(),
                'torsion_disagreement': (tor_a - tor_b).detach().abs()},
            'torsion_coordination': {'e3l_weight': tor_gate, 'residual': tor_residual},
            'confidence': self.judge_head(context), 'state': self.state_head(context)}
