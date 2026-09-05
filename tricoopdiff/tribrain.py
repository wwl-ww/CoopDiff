"""
TriBrain: lightweight conditioning network for two equivariant experts.

The ONLY trainable module in TriCoopDiff (~2.2M params). Operates on the full
ternary graph (E3 + POI + PROTAC, all edges including direct E3-POI).

Architecture:
  MultiScalePooling → token processor → Multi-Head Outputs
  - FiLM heads: per-layer modulation params for two subgraphs
  - Judge head: confidence
  - State head: ΔΔG, productive probability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch_scatter is a common dependency but may need separate install.
# Fall back to manual scatter if unavailable.
try:
    from torch_scatter import scatter_mean, scatter_sum
    _has_torch_scatter = True
except ImportError:
    _has_torch_scatter = False

    def scatter_mean(src, index, dim=0, dim_size=None):
        """Fallback scatter_mean using torch operations."""
        if dim_size is None:
            dim_size = index.max().item() + 1
        out = src.new_zeros(dim_size, src.shape[1])
        ones = src.new_ones(src.shape[0], 1)
        out.index_add_(0, index, src)
        count = out.new_zeros(dim_size, 1)
        count.index_add_(0, index, ones)
        count = count.clamp(min=1)
        return out / count

    def scatter_sum(src, index, dim=0, dim_size=None):
        """Fallback scatter_sum using torch operations."""
        if dim_size is None:
            dim_size = index.max().item() + 1
        out = src.new_zeros(dim_size, src.shape[1])
        out.index_add_(0, index, src)
        return out


# ---------------------------------------------------------------------------
# Multi-Scale Pooling
# ---------------------------------------------------------------------------

class MultiScalePooling(nn.Module):
    """
    Pool the ternary-graph node features into a single [B, token_dim] token.

    Despite the name, this is NOT distance-tiered pooling: there is no distance
    cutoff and no Cα coarse-graining. All three branches below operate on the
    SAME full node set (every E3 + POI + PROTAC node); they differ only in the
    linear projection and the pooling operator:

    - ``short``:  ``short_proj``  + attention pooling (learnable ``short_query``)
    - ``mid``:    ``mid_proj``    + attention pooling (learnable ``mid_query``)
    - ``global``: ``global_proj`` + (mean + max) / 2 pooling

    The three pooled [B, token_dim] tokens are SUMMED (not concatenated) into the
    final output. ``node_coords`` is accepted for interface compatibility but is
    currently unused. Real distance-tiered pooling is not implemented here;
    geometric reasoning is delegated to the two equivariant experts.
    """

    def __init__(
        self,
        node_dim: int = 128,
        token_dim: int = 128,
        input_dim: int = None,
    ):
        """
        Args:
            node_dim: internal representation dimension (legacy, defaults to input_dim)
            token_dim: output token dimension
            input_dim: actual input feature dimension (e.g. 1280 for ESM2).
                       Defaults to node_dim for backwards compatibility.
        """
        super().__init__()
        if input_dim is None:
            input_dim = node_dim
        self.node_dim = node_dim
        self.token_dim = token_dim
        self.input_dim = input_dim

        # Projection layers for each scale
        self.short_proj = nn.Linear(input_dim, token_dim)
        self.mid_proj = nn.Linear(input_dim, token_dim)
        self.global_proj = nn.Linear(input_dim, token_dim)

        # Attention pooling: learnable query vectors
        self.short_query = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.mid_query = nn.Parameter(torch.randn(1, token_dim) * 0.02)

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        node_types: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, D] all node features
            node_coords: [N, 3] all node coordinates
            node_types: [N] 0=E3, 1=POI, 2=PROTAC
            batch: [N] batch index (optional, defaults to all zeros)

        Returns:
            [B, token_dim] pooled token (the SUM of the three branch pools;
            they are not concatenated, despite older comments claiming 3×128→384).
        """
        if batch is None:
            batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device)
        n_batch = batch.max().item() + 1

        # Branch 1 ("short"): projection + attention pooling (learnable query).
        short_feat = self.short_proj(node_features)  # [N, token_dim]
        # Simple attention: dot product with learnable query
        attn_scores = torch.matmul(short_feat, self.short_query.T)  # [N, 1]
        attn_weights = torch_softmax_per_batch(attn_scores, batch, n_batch)  # [N, 1]
        short_pooled = scatter_sum(short_feat * attn_weights, batch, dim=0, dim_size=n_batch)  # [B, token_dim]

        # Branch 2 ("mid"): projection + attention pooling (learnable query).
        # (No Cα coarse-graining is performed; the caller passes the full node set.)
        mid_feat = self.mid_proj(node_features)
        attn_scores_mid = torch.matmul(mid_feat, self.mid_query.T)
        attn_weights_mid = torch_softmax_per_batch(attn_scores_mid, batch, n_batch)
        mid_pooled = scatter_sum(mid_feat * attn_weights_mid, batch, dim=0, dim_size=n_batch)

        # Global: mean + max pooling
        global_feat = self.global_proj(node_features)
        global_mean = scatter_mean(global_feat, batch, dim=0, dim_size=n_batch)
        global_max = scatter_max_per_batch(global_feat, batch, n_batch)

        # Sum the three pooled [B, token_dim] tokens (NOT concatenation). The
        # three branches are parallel views of the same node set, so their sum
        # is the final pooled representation.
        pooled = short_pooled + mid_pooled + (global_mean + global_max) / 2.0

        return pooled  # [B, token_dim]


def torch_softmax_per_batch(scores, batch, n_batch):
    """Softmax within each batch element."""
    # scores: [N, 1], batch: [N]
    outputs = []
    for b in range(n_batch):
        mask = batch == b
        if mask.sum() > 0:
            sm = F.softmax(scores[mask].squeeze(-1), dim=0)
            outputs.append(sm)
        else:
            outputs.append(torch.zeros(0, device=scores.device))
    # Reconstruct
    result = torch.zeros(scores.shape[0], device=scores.device)
    for b in range(n_batch):
        mask = batch == b
        if mask.sum() > 0:
            result[mask] = outputs[b]
    return result.unsqueeze(-1)


def scatter_max_per_batch(feat, batch, n_batch):
    """Max pooling within each batch element."""
    outputs = []
    for b in range(n_batch):
        mask = batch == b
        if mask.sum() > 0:
            outputs.append(feat[mask].max(dim=0)[0])
        else:
            outputs.append(torch.zeros(feat.shape[1], device=feat.device))
    return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Equivariant Transformer
# ---------------------------------------------------------------------------

class EquivariantTransformer(nn.Module):
    """
    Scalar/vector token processor operating on the SINGLE pooled token.

    NOTE ON THE NAME: "EquivariantTransformer" is a legacy/misleading name. This
    is NOT an SE(3)-equivariant transformer over the ternary graph — it does not
    read per-node coordinates or features and performs no cross-product /
    geometric interaction. It only maps the pooled [B, node_dim] token through
    scalar and vector linear projections and a stack of layers (scalar
    self-attention + scalar-gated vector magnitude update). The vector channels
    are updated equivariantly (see ``EquivariantTransformerLayer``), but all
    geometry enters through the interface graphs consumed by the experts.

    MVP configuration: 3 layers, single-head self-attention over the batch,
    32 scalar channels + 4 vector channels.
    """

    def __init__(
        self,
        node_dim: int = 128,
        ns: int = 32,
        nv: int = 4,
        num_layers: int = 3,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ns = ns
        self.nv = nv
        self.node_dim = node_dim

        # Project input to equivariant representation
        self.scalar_proj = nn.Linear(node_dim, ns)
        self.vector_proj = nn.Linear(node_dim, nv * 3)  # 3 channels per vector dim

        self.layers = nn.ModuleList([
            EquivariantTransformerLayer(ns, nv, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.scalar_out = nn.Linear(ns, node_dim)
        self.vector_out = nn.Linear(nv * 3, node_dim)

    def forward(self, pooled: torch.Tensor) -> dict:
        """
        Args:
            pooled: [B, node_dim] pooled token from MultiScalePooling.

        Returns:
            dict with 'scalar': [B, node_dim], 'vector': [B, node_dim], and
            'combined': [B, node_dim] (element-wise sum of the two). Note the
            'vector' key holds a flat [B, node_dim] projection of the nv vector
            channels, not the raw [B, nv, 3] channels.
        """
        B = pooled.shape[0]

        # Project to equivariant representation
        s = self.scalar_proj(pooled)  # [B, ns]
        v = self.vector_proj(pooled).view(B, self.nv, 3)  # [B, nv, 3]

        for layer in self.layers:
            s, v = layer(s, v)

        scalar_out = self.scalar_out(s)  # [B, node_dim]
        vector_out = self.vector_out(v.reshape(B, self.nv * 3))  # [B, node_dim]

        return {
            'scalar': scalar_out,
            'vector': vector_out,
            'combined': scalar_out + vector_out,
        }


class EquivariantTransformerLayer(nn.Module):
    """Single layer of the equivariant transformer."""

    def __init__(self, ns: int, nv: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.ns = ns
        self.nv = nv

        # Scalar update: self-attention + FFN
        self.scalar_norm1 = nn.LayerNorm(ns)
        self.scalar_attn = nn.MultiheadAttention(ns, num_heads, dropout=dropout, batch_first=True)
        self.scalar_norm2 = nn.LayerNorm(ns)
        self.scalar_ffn = nn.Sequential(
            nn.Linear(ns, ns * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ns * 4, ns),
            nn.Dropout(dropout),
        )

        # Vector update: scalar-gated magnitude modulation (equivariant).
        # NOTE: ``vector_norm`` (nn.LayerNorm over the flattened nv*3 components)
        # is retained ONLY so that checkpoints saved before this fix still load
        # under strict state_dict matching. It is intentionally NOT used in
        # forward: LayerNorm over flattened x/y/z components mixes spatial
        # directions and breaks O(3) equivariance. The equivariant replacement is
        # per-vector magnitude normalization (F.normalize) in forward.
        self.vector_norm = nn.LayerNorm(nv * 3)
        self.vector_gate = nn.Linear(ns, nv)  # scalar → vector gate

    def forward(
        self, s: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: [B, ns] scalar features
            v: [B, nv, 3] vector features

        Returns:
            updated (s, v)
        """
        B = s.shape[0]

        # --- Scalar update with self-attention ---
        s_normed = self.scalar_norm1(s)
        # Self-attention over batch (treat batch as sequence length 1, so it's identity-ish)
        # For single-token-per-batch, attention reduces to FFN. Multi-sample batches benefit.
        s_attn, _ = self.scalar_attn(s_normed.unsqueeze(1), s_normed.unsqueeze(1), s_normed.unsqueeze(1))
        s = s + s_attn.squeeze(1)
        s = s + self.scalar_ffn(self.scalar_norm2(s))

        # --- Vector update ---
        # Gate: scalar features modulate vector magnitudes in an O(3)-equivariant
        # way — each 3-vector is normalized to unit length (direction preserved,
        # since rotations preserve norms and directions), then scaled by a
        # per-channel scalar gate. The old flattened nn.LayerNorm(nv*3) mixed the
        # x/y/z axes and broke equivariance; it is no longer used.
        gate = self.vector_gate(s).sigmoid()  # [B, nv]
        v = F.normalize(v, dim=-1, eps=1e-8)  # [B, nv, 3], unit direction
        v = v * gate.unsqueeze(-1)

        return s, v


# ---------------------------------------------------------------------------
# Output Heads
# ---------------------------------------------------------------------------

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


class JudgeHead(nn.Module):
    """Predicts per-atom clash mask and global confidence."""

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
        # Per-bond inputs are: sub1, sub2, their difference, and their product.
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
        context_per_bond = context.unsqueeze(1).expand(-1, n_torsions, -1)
        bond_evidence = torch.stack(
            (
                score_e3l,
                score_pol,
                score_e3l - score_pol,
                score_e3l * score_pol,
            ),
            dim=-1,
        )
        coordination = self.net(
            torch.cat((context_per_bond, bond_evidence), dim=-1)
        )
        weight_e3l = torch.sigmoid(coordination[..., 0])
        residual = coordination[..., 1]
        score = (
            weight_e3l * score_e3l
            + (1.0 - weight_e3l) * score_pol
            + residual
        )
        return score, weight_e3l, residual


# ---------------------------------------------------------------------------
# Full TriBrain
# ---------------------------------------------------------------------------

class TriBrain(nn.Module):
    """
    TriBrain: the coordination brain of TriCoopDiff.

    Takes the full ternary graph → pools to tokens → equivariant transformer →
    produces branch-specific FiLM modulation parameters, clash confidence,
    and state predictions. Rigid interface scores come exclusively from the
    two equivariant DiffDock experts; TriBrain no longer predicts an unrelated
    Cartesian POI score.

    This is the ONLY trainable module. ~2.2M params in MVP configuration.
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
            ns: scalar channel count
            nv: vector channel count
            num_transformer_layers: depth of equivariant transformer
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

        # Multi-scale pooling: ~66k params
        self.pooling = MultiScalePooling(
            node_dim=node_dim, token_dim=node_dim, input_dim=input_dim
        )

        # Equivariant transformer: ~1.1M params
        self.transformer = EquivariantTransformer(
            node_dim=node_dim,
            ns=ns,
            nv=nv,
            num_layers=num_transformer_layers,
            dropout=dropout,
        )

        # Scalar auxiliary heads. Rigid-body vector scores must remain
        # equivariant and are therefore supplied by the DiffDock experts.
        self.judge_head = JudgeHead(hidden_dim=node_dim)         # ~16k
        self.state_head = StateHead(hidden_dim=node_dim)         # ~16k
        self.torsion_coordinator = TorsionCoordinator(
            context_dim=node_dim,
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

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        node_types: torch.Tensor,
        batch: torch.Tensor = None,
        t_e3l: torch.Tensor = None,
        t_pol: torch.Tensor = None,
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
        pooled = self.pooling(node_features, node_coords, node_types, batch)

        # 2. Scalar/vector token processor for global conditioning.
        transformer_out = self.transformer(pooled)
        h_cond = transformer_out['combined']  # [B, node_dim]

        # 3. Auxiliary scalar heads.
        confidence = self.judge_head(h_cond)      # [B, 1]
        state = self.state_head(h_cond)           # [B, 2]

        # 4. Each expert is conditioned by its own interface clock. Reusing one
        # embedding here would recreate the train/inference mismatch that the
        # interface-coordinate formulation is intended to remove.
        if t_pol is None:
            t_pol = t_e3l
        h_film_e3l = h_cond
        h_film_pol = h_cond
        if t_e3l is not None:
            h_film_e3l = h_film_e3l + fourier_time_embedding(
                t_e3l, self.node_dim // 2,
            )
        if t_pol is not None:
            h_film_pol = h_film_pol + fourier_time_embedding(
                t_pol, self.node_dim // 2,
            )
        h_film_e3l = self.film_shared(h_film_e3l)
        h_film_pol = self.film_shared(h_film_pol)
        film_params = []
        for index, head in enumerate(self.film_heads):
            h_film = h_film_e3l if index < self.num_conv_layers else h_film_pol
            gamma, beta = head(h_film)
            film_params.append((gamma, beta))

        return {
            'confidence': confidence,
            'state': state,
            'film_params': film_params,  # list of (gamma, beta), len=num_conv_layers*2
            'h_cond': h_cond,
        }


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
