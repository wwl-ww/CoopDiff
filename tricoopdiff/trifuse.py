"""Legacy FiLM controller retained for historical experiments.

Production model-v2 uses consensus.py. Shared layers are re-exported here to
preserve existing imports; checkpoint state_dict parameter names are unchanged.
"""

import math

import torch
from torch import nn

from .geometry_encoders import LigandFeatureEncoder, BondContextEncoder, TernaryGeometryEncoder
from .consensus_layers import TokenNegotiator, JudgeHead, StateHead, TorsionCoordinator


TELEMETRY_PER_LAYER = 8
TELEMETRY_TAIL = 8


def fourier_time_embedding(t: torch.Tensor, n_freqs: int = 16) -> torch.Tensor:
    """
    Sinusoidal embedding of a scalar diffusion time ``t`` ∈ [0, 1].

    Args:
        t: [B] diffusion time (noise-schedule variable, not physical time)
        n_freqs: number of geometric frequencies (1, 2, 4, …, 2^{n_freqs-1})

    Returns:
        [B, 2·n_freqs] concatenation of sin/cos at each frequency.
    """
    freqs = 2.0 ** torch.arange(n_freqs, device=t.device, dtype=t.dtype)
    ang = math.pi * freqs[None, :] * t[:, None]   # [B, n_freqs]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ScoreControlHead(nn.Module):
    """Predict bounded equivariant-safe gains for the five executable scores."""

    def __init__(self, hidden_dim: int = 128, max_delta: float = 0.25):
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.max_delta * torch.tanh(self.net(context))


class LayerwiseExpertBridge(nn.Module):
    """Exchange aligned ligand scalar evidence between experts after each conv."""

    def __init__(self, scalar_dim: int, hidden_dim: int,
                 num_layers: int, max_delta: float = 0.1):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.max_delta = float(max_delta)
        self.input = nn.Sequential(
            nn.LayerNorm(scalar_dim * 4),
            nn.Linear(scalar_dim * 4, hidden_dim),
            nn.SiLU(),
        )
        self.layer_embedding = nn.Embedding(num_layers, hidden_dim)
        self.output = nn.Linear(hidden_dim, scalar_dim * 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, scalar_e3l: torch.Tensor, scalar_pol: torch.Tensor,
                layer_index: int):
        if scalar_e3l.shape != scalar_pol.shape:
            raise ValueError("dual expert ligand scalar shapes differ")
        if scalar_e3l.ndim != 2 or scalar_e3l.shape[1] != self.scalar_dim:
            raise ValueError("expert bridge expects [N_lig, scalar_dim]")
        bounded_e3l = torch.tanh(scalar_e3l / 10.0)
        bounded_pol = torch.tanh(scalar_pol / 10.0)
        hidden = self.input(torch.cat((
            bounded_e3l, bounded_pol,
            bounded_e3l - bounded_pol,
            bounded_e3l * bounded_pol,
        ), dim=-1))
        hidden = hidden + self.layer_embedding.weight[layer_index]
        delta = self.max_delta * torch.tanh(self.output(hidden))
        e3l_gamma, e3l_beta, pol_gamma, pol_beta = delta.chunk(4, dim=-1)
        return (e3l_gamma, e3l_beta), (pol_gamma, pol_beta)


class TriFuse(nn.Module):
    """
    Geometry-aware controller coordinating both docking interfaces.

    Takes the full ternary graph → typed geometry tokens → token negotiation →
    branch-specific FiLM, bounded score controls, and bond decisions. Reserved
    confidence/state outputs remain frozen until supervised. Rigid scores come from the
    two equivariant DiffDock experts; TriFuse no longer predicts an unrelated
    Cartesian POI score.

    Expert LoRA adapters are trainable alongside this coordinator.
    """

    def __init__(
        self,
        node_dim: int = 128,
        input_dim: int = None,
        ns: int = 32,
        nv: int = 4,
        num_transformer_layers: int = 3,
        num_conv_layers_per_expert: int = 11,
        film_dim: int = 48,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_dim: internal representation dimension
            input_dim: input feature dimension (e.g. 1280 for ESM2).
                       Defaults to node_dim for backwards compatibility.
            ns: retained CLI compatibility parameter; rigid vector channels are
                owned by DiffDock.
            nv: retained CLI compatibility parameter; rigid vector channels are
                owned by DiffDock.
            num_transformer_layers: depth of scalar context processor
            num_conv_layers_per_expert: number of conv layers in each DiffDock expert
                                       (used to size the FiLM head list). DiffDock-L
                                       has 11 conv layers (NOT 6).
            film_dim: number of scalar channels to modulate per conv layer.
                      Must equal the expert's `ns` (48 for DiffDock-L). Only the
                      scalar block `node_attr[:, :film_dim]` is FiLM-modulated to
                      preserve SE(3) equivariance.
            dropout: dropout rate
        """
        super().__init__()
        if input_dim is None:
            input_dim = node_dim
        self.node_dim = node_dim
        self.input_dim = input_dim
        self.num_conv_layers = num_conv_layers_per_expert
        self.film_dim = film_dim

        self.pooling = TernaryGeometryEncoder(
            input_dim=input_dim, hidden_dim=node_dim,
        )

        # Five compact invariant tokens communicate without materializing dense
        # residue/atom attention. Spatial vector predictions remain owned by
        # the equivariant DiffDock experts.
        self.transformer = TokenNegotiator(
            hidden_dim=node_dim,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )
        self.diffusion_state_encoder = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

        # Scalar auxiliary heads. Rigid-body vector scores must remain
        # equivariant and are therefore supplied by the DiffDock experts.
        self.judge_head = JudgeHead(hidden_dim=node_dim)         # ~16k
        self.state_head = StateHead(hidden_dim=node_dim)         # ~16k
        self.score_control_head = ScoreControlHead(hidden_dim=node_dim)
        self.layerwise_bridge = LayerwiseExpertBridge(
            scalar_dim=film_dim, hidden_dim=node_dim,
            num_layers=num_conv_layers_per_expert,
        )
        self.torsion_coordinator = TorsionCoordinator(
            context_dim=node_dim,
        )
        self.bond_context_encoder = BondContextEncoder(node_dim)
        telemetry_dim = (
            num_conv_layers_per_expert * TELEMETRY_PER_LAYER + TELEMETRY_TAIL
        )
        self.telemetry_encoder = nn.Sequential(
            nn.LayerNorm(telemetry_dim * 2),
            nn.Linear(telemetry_dim * 2, node_dim * 2),
            nn.SiLU(),
            nn.Linear(node_dim * 2, node_dim),
        )

        # FiLM: shared MLP → per-layer heads (12 heads = 6 layers × 2 subgraphs)
        self.film_shared = nn.Sequential(
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )  # ~66k

        # 2 × num_conv_layers_per_expert independent FiLM heads
        # (num_conv_layers × 2 subgraphs). Each head: node_dim → film_dim*2
        # (gamma + beta), zero-initialized for transparency at training start.
        self.film_heads = nn.ModuleList([
            FiLMHead(node_dim, film_dim) for _ in range(num_conv_layers_per_expert * 2)
        ])
        self.film_layer_embedding = nn.Embedding(
            num_conv_layers_per_expert, node_dim,
        )
        self.film_branch_embedding = nn.Embedding(2, node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        node_types: torch.Tensor,
        batch: torch.Tensor = None,
        t_e3l: torch.Tensor = None,
        t_pol: torch.Tensor = None,
        lig_edge_index: torch.Tensor = None,
        lig_edge_mask: torch.Tensor = None,
        lig_mask_rotate=None,
        diffusion_state: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            node_features: [N_total, D] all ternary graph node features
            node_coords: [N_total, 3] all node coordinates
            node_types: [N_total] 0=E3, 1=POI, 2=PROTAC
            batch: [N_total] batch index
            t_e3l: [B] diffusion time for the E3--Ligand expert.
            t_pol: [B] diffusion time for the POI--Ligand expert. Defaults to
                ``t_e3l`` when both interface processes use the same schedule.

        Returns:
            dict with:
              'confidence': [B, 1] clash confidence
              'state': [B, 2] (ΔΔG, p_productive)
              'film_params': list of (gamma, beta) tuples, length = 2 * num_conv_layers
                             First num_conv_layers are for subgraph 1 (E3-L),
                             Next num_conv_layers are for subgraph 2 (POI-L)
        """
        # 1. Pool the ternary graph to compact tokens
        pooled = self.pooling(
            node_features, node_coords, node_types, batch,
            lig_edge_index=lig_edge_index,
        )

        # 2. Negotiate between both interfaces, linker, and the physical noise
        # state. Fourier(t) alone does not carry sigma units or solver step size.
        if diffusion_state is None:
            batch_size = pooled['global'].shape[0]
            diffusion_state = pooled['global'].new_zeros(batch_size, 9)
        if diffusion_state.shape != (pooled['global'].shape[0], 9):
            raise ValueError(
                f"diffusion_state must have shape [B,9], got {diffusion_state.shape}"
            )
        state_token = self.diffusion_state_encoder(diffusion_state)
        linker_token = torch.stack([
            tokens.mean(dim=0) for tokens in pooled['ligand_tokens']
        ], dim=0)
        tokens = torch.stack((
            pooled['global'], pooled['e3l'], pooled['pol'], linker_token,
            state_token,
        ), dim=1)
        negotiated = self.transformer(tokens)
        h_global, h_e3l, h_pol, h_linker, h_state = negotiated.unbind(dim=1)
        h_cond = h_global + 0.5 * (h_e3l + h_pol)
        if len(pooled['ligand_tokens']) != 1:
            raise ValueError("TriFuse bond contexts currently require one graph")
        ligand_mask = node_types == 2
        bond_context = self.bond_context_encoder(
            pooled['ligand_tokens'][0], node_coords[ligand_mask],
            lig_edge_index, lig_edge_mask, lig_mask_rotate, h_cond,
        )

        # 3. Auxiliary scalar heads.
        confidence = self.judge_head(h_cond)      # [B, 1]
        state = self.state_head(h_cond)           # [B, 2]
        score_gains = self.score_control_head(h_state)

        # 4. Each expert is conditioned by its own interface clock. Reusing one
        # embedding here would recreate the train/inference mismatch that the
        # interface-coordinate formulation is intended to remove.
        if t_pol is None:
            t_pol = t_e3l
        h_film_e3l = h_global + h_e3l
        h_film_pol = h_global + h_pol
        if t_e3l is not None:
            h_film_e3l = h_film_e3l + fourier_time_embedding(
                t_e3l, self.node_dim // 2,
            )
        if t_pol is not None:
            h_film_pol = h_film_pol + fourier_time_embedding(
                t_pol, self.node_dim // 2,
            )
        ligand_tokens = pooled['ligand_tokens'][0]
        film_params = []
        for index, head in enumerate(self.film_heads):
            branch = 0 if index < self.num_conv_layers else 1
            layer = index % self.num_conv_layers
            branch_context = h_film_e3l if branch == 0 else h_film_pol
            atom_context = (
                ligand_tokens
                + branch_context[0]
                + self.film_layer_embedding.weight[layer]
                + self.film_branch_embedding.weight[branch]
            )
            h_film = self.film_shared(atom_context)
            gamma, beta = head(h_film)
            film_params.append((gamma, beta))

        return {
            'confidence': confidence,
            'state': state,
            'score_gains': score_gains,
            'film_params': film_params,  # list of (gamma, beta), len=num_conv_layers*2
            'h_cond': h_cond,
            'h_e3l': h_e3l,
            'h_pol': h_pol,
            'h_linker': h_linker,
            'h_state': h_state,
            'bond_context': bond_context,
        }

    def condition_bonds_on_telemetry(
        self,
        bond_context: torch.Tensor,
        telemetry_e3l: torch.Tensor,
        telemetry_pol: torch.Tensor,
    ) -> torch.Tensor:
        """Inject compact current-step expert evidence into every bond token."""
        if bond_context.ndim != 3:
            raise ValueError("bond_context must have shape [B,K,D]")
        expected = self.telemetry_encoder[0].normalized_shape[0] // 2
        if telemetry_e3l.shape != telemetry_pol.shape:
            raise ValueError("expert telemetry shapes differ")
        if telemetry_e3l.shape != (bond_context.shape[0], expected):
            raise ValueError(
                f"telemetry must have shape [B,{expected}], got "
                f"{tuple(telemetry_e3l.shape)}"
            )
        evidence = self.telemetry_encoder(torch.cat((
            telemetry_e3l, telemetry_pol,
        ), dim=-1))
        return bond_context + evidence.unsqueeze(1)


class FiLMHead(nn.Module):
    """
    Single FiLM head: produces (gamma, beta) for one conv layer of one subgraph.
    Zero-initialized for transparency at training start (adaLN-Zero paradigm).
    """

    def __init__(self, hidden_dim: int = 128, film_dim: int = 48):
        super().__init__()
        self.film_dim = film_dim
        self.net = nn.Linear(hidden_dim, film_dim * 2)
        # Zero-initialize
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, hidden_dim] conditioning
        Returns:
            gamma: [B, film_dim], beta: [B, film_dim]
        """
        out = self.net(h)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta
