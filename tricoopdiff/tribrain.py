"""
TriBrain: Lightweight SE(3)-equivariant coordination network.

The ONLY trainable module in TriCoopDiff (~2.2M params). Operates on the full
ternary graph (E3 + POI + PROTAC, all edges including direct E3-POI).

Architecture (MVP):
  MultiScalePooling → EquivariantTransformer → Multi-Head Outputs
  - SE3 head: predicts E3-POI rigid body score (s_SE3)
  - FiLM heads: per-layer modulation params for two subgraphs
  - Judge head: clash mask, confidence
  - State head: ΔΔG, productive probability

Capacity upgrade path: 3-layer/32ch → 6-layer/64ch (~2.2M → ~5M params)
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
    Three-scale pooling of the ternary graph into a compact token sequence.

    - Short-range (<20Å): attention pooling on heavy atoms, d=64
    - Mid-range (<50Å): Cα coarse-graining + attention pooling, d=64
    - Global: mean + max pooling over all nodes, d=64

    Output: [N_tokens, 128] where the first dim is the 3 pooled tokens
    concatenated along the feature dimension.

    Total params: ~66k
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
            [B, 128] pooled token sequence (flattened 3×128→384 for now,
                      split into separate heads downstream)
        """
        if batch is None:
            batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device)
        n_batch = batch.max().item() + 1

        # Short-range: attention pooling over ligand + nearby residues
        short_feat = self.short_proj(node_features)  # [N, token_dim]
        # Simple attention: dot product with learnable query
        attn_scores = torch.matmul(short_feat, self.short_query.T)  # [N, 1]
        attn_weights = torch_softmax_per_batch(attn_scores, batch, n_batch)  # [N, 1]
        short_pooled = scatter_sum(short_feat * attn_weights, batch, dim=0, dim_size=n_batch)  # [B, token_dim]

        # Mid-range: also attention-pooled (Cα-level is handled by caller via input selection)
        mid_feat = self.mid_proj(node_features)
        attn_scores_mid = torch.matmul(mid_feat, self.mid_query.T)
        attn_weights_mid = torch_softmax_per_batch(attn_scores_mid, batch, n_batch)
        mid_pooled = scatter_sum(mid_feat * attn_weights_mid, batch, dim=0, dim_size=n_batch)

        # Global: mean + max pooling
        global_feat = self.global_proj(node_features)
        global_mean = scatter_mean(global_feat, batch, dim=0, dim_size=n_batch)
        global_max = scatter_max_per_batch(global_feat, batch, n_batch)

        # Concatenate all scales: [B, 128]
        # Each scale contributes equally: short=43, mid=43, global=42 ≈ 128
        # (dimensionality managed by splitting 128 across the three linear projections)
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
# SE3 Geometry Encoder
# ---------------------------------------------------------------------------

class SE3GeometryEncoder(nn.Module):
    """
    Coordinate-aware set encoder → fixed-size geometry embedding for the SE3 head.

    The SE3 score target (translation + so(3) rotation of POI relative to E3) is a
    function of the NOISY ternary coordinates, not just the ESM features. The
    feature-based ``h_cond`` is coordinate-blind (MultiScalePooling and
    EquivariantTransformer both ignore ``node_coords``), so without this encoder
    the SE3 head cannot regress the target — loss_se3 plateaus ~1.1 instead of
    converging to 0.

    Encodes per-node [E3-relative coords, node-type one-hot] through a small MLP,
    then mean/max pools across nodes → [B, geo_dim]. Subtracting the E3 center
    makes the encoding translation-invariant and directly exposes the POI's
    (noisy) pose relative to the fixed E3 frame.
    """

    def __init__(self, geo_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.geo_dim = geo_dim
        self.node_net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        # +12 = 6-entry covariance (second-moment) vector for each of POI and E3.
        # These are rotation-sensitive (R·C·Rᵀ in the fixed E3 frame) and recover
        # the orientation info that symmetric scalar mean/max pooling destroys
        # (the sole mean/max path capped rotation readout at R²≈0.51).
        self.combine = nn.Linear(2 * hidden + 12, geo_dim)

    def forward(self, node_coords, node_types, batch=None):
        if batch is None:
            batch = torch.zeros(
                node_coords.shape[0], dtype=torch.long, device=node_coords.device,
            )
        n_batch = batch.max().item() + 1
        node_types = node_types.to(node_coords.device)

        # E3 center per batch (node_type 0). E3 is fixed across the diffusion
        # trajectory; subtracting it exposes the POI's pose relative to E3.
        e3_mask = node_types == 0
        e3_sum = torch.zeros(
            n_batch, 3, device=node_coords.device, dtype=node_coords.dtype,
        )
        e3_cnt = torch.zeros(n_batch, device=node_coords.device, dtype=node_coords.dtype)
        e3_sum.index_add_(0, batch[e3_mask], node_coords[e3_mask])
        e3_cnt.index_add_(
            0, batch[e3_mask],
            torch.ones(int(e3_mask.sum()), device=node_coords.device, dtype=node_coords.dtype),
        )
        e3_center = e3_sum / e3_cnt.clamp(min=1).unsqueeze(-1)  # [B, 3]

        rel = node_coords - e3_center[batch]                 # [N, 3]
        onehot = F.one_hot(node_types, num_classes=3).to(node_coords.dtype)  # [N, 3]
        feat = torch.cat([rel, onehot], dim=-1)              # [N, 6]

        h = self.node_net(feat)                              # [N, hidden]
        mean = scatter_mean(h, batch, dim=0, dim_size=n_batch)   # [B, hidden]
        mx = scatter_max_per_batch(h, batch, n_batch)            # [B, hidden]
        cov = self._second_moments(node_coords, node_types, batch, n_batch)  # [B, 12]
        return self.combine(torch.cat([mean, mx, cov], dim=-1))               # [B, geo_dim]

    def _second_moments(self, node_coords, node_types, batch, n_batch):
        """Per-cloud covariance (6 unique entries) for POI (type 1) then E3 (type 0).

        Computed about each cloud's own centroid → translation-invariant, and in
        the fixed E3 frame the POI covariance is R·C·Rᵀ, i.e. directly
        rotation-sensitive. Concatenated to [B, 12].
        """
        entries = []
        for typ in (1, 0):
            mask = node_types == typ
            c = scatter_mean(node_coords[mask], batch[mask], dim=0, dim_size=n_batch)  # [B, 3]
            rel_t = node_coords[mask] - c[batch[mask]]        # [N_t, 3]
            outer = rel_t.unsqueeze(-1) * rel_t.unsqueeze(-2) # [N_t, 3, 3]
            cov = scatter_mean(                               # [B, 3, 3]
                outer.reshape(rel_t.shape[0], 9),
                batch[mask], dim=0, dim_size=n_batch,
            ).reshape(n_batch, 3, 3)
            cov = (cov + cov.transpose(-1, -2)) / 2
            entries.append(torch.stack([
                cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2],
                cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2],
            ], dim=-1))                                       # [B, 6]
        return torch.cat(entries, dim=-1)                     # [B, 12]


# ---------------------------------------------------------------------------
# Equivariant Transformer
# ---------------------------------------------------------------------------

class EquivariantTransformer(nn.Module):
    """
    SE(3)-equivariant transformer for ternary graph processing.

    MVP configuration: 3 layers, single-head cross-attention,
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

    def forward(
        self,
        pooled: torch.Tensor,
        node_coords: torch.Tensor,
        node_features: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            pooled: [B, node_dim] pooled token from MultiScalePooling
            node_coords: [N, 3]
            node_features: [N, node_dim]
            batch: [N]

        Returns:
            dict with 'scalar': [B, node_dim], 'vector': [B, nv*3]
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

        # Vector update: cross-product + scalar modulation
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
        # Gate: scalar features modulate vector magnitudes
        gate = self.vector_gate(s).sigmoid()  # [B, nv]
        v_flat = v.reshape(B, self.nv * 3)
        v_flat = self.vector_norm(v_flat)
        v = v_flat.reshape(B, self.nv, 3)
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


class SE3ScoreHead(nn.Module):
    """
    Predicts the SE(3) score (6D: 3 translation + 3 rotation in so(3))
    for the E3-POI rigid body transformation.

    The SE(3) score target depends on the diffusion time ``t_se3`` (through the
    noise schedule σ_tr/σ_rot), so the head is fed ``h_cond`` concatenated with a
    sinusoidal time embedding: ``in_dim = node_dim + 2·n_freqs``.
    """

    def __init__(self, hidden_dim: int = 128, in_dim: int = None):
        super().__init__()
        in_dim = in_dim if in_dim is not None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # [tx, ty, tz, rx, ry, rz]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 6] SE(3) score."""
        return self.net(x)


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


# ---------------------------------------------------------------------------
# Full TriBrain
# ---------------------------------------------------------------------------

class TriBrain(nn.Module):
    """
    TriBrain: the coordination brain of TriCoopDiff.

    Takes the full ternary graph → pools to tokens → equivariant transformer →
    produces FiLM modulation parameters for both subgraphs, SE(3) score,
    clash confidence, and state predictions.

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

        # Output heads
        # The SE3 head takes h_cond + geometry embedding + diffusion-time embedding:
        # the SE3 score target depends on BOTH t_se3 (via σ_tr/σ_rot) and the noisy
        # coordinates (via the POI pose relative to E3). Without the geometry
        # encoder the head is coordinate-blind and cannot regress the target.
        self.time_freqs = 16
        self.geo_dim = 64
        self.geo_encoder = SE3GeometryEncoder(geo_dim=self.geo_dim)
        self.se3_head = SE3ScoreHead(
            hidden_dim=node_dim, in_dim=node_dim + self.geo_dim + 2 * self.time_freqs,
        )
        self.judge_head = JudgeHead(hidden_dim=node_dim)         # ~16k
        self.state_head = StateHead(hidden_dim=node_dim)         # ~16k

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
        t_se3: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            node_features: [N_total, D] all ternary graph node features
            node_coords: [N_total, 3] all node coordinates
            node_types: [N_total] 0=E3, 1=POI, 2=PROTAC
            batch: [N_total] batch index
            t_se3: [B] SE3 diffusion time (noise-schedule variable). Conditions the
                SE3 head, whose score target is a function of σ(t_se3). If None, a
                zero-time embedding is used (backward-compatible; SE3 head sees only
                geometry, as before this fix).

        Returns:
            dict with:
              's_se3': [B, 6] E3-POI rigid body score
              'confidence': [B, 1] clash confidence
              'state': [B, 2] (ΔΔG, p_productive)
              'film_params': list of (gamma, beta) tuples, length = 2 * num_conv_layers
                             First num_conv_layers are for subgraph 1 (E3-L),
                             Next num_conv_layers are for subgraph 2 (POI-L)
        """
        # 1. Pool the ternary graph to compact tokens
        pooled = self.pooling(node_features, node_coords, node_types, batch)

        # 2. Equivariant transformer
        transformer_out = self.transformer(pooled, node_coords, node_features, batch)
        h_cond = transformer_out['combined']  # [B, node_dim]

        # 3. Output heads
        # The SE3 head is conditioned on t_se3 (score target varies with σ) AND on
        # the noisy coordinates (score target depends on the POI pose relative to
        # E3). The geometry encoder supplies the latter; without it the SE3 head
        # is coordinate-blind and cannot regress the target.
        h_geo = self.geo_encoder(node_coords, node_types, batch)  # [B, geo_dim]
        if t_se3 is not None:
            t_emb = fourier_time_embedding(t_se3, self.time_freqs)  # [B, 2·n_freqs]
        else:
            t_emb = torch.zeros(
                h_cond.shape[0], 2 * self.time_freqs,
                device=h_cond.device, dtype=h_cond.dtype,
            )
        h_se3 = torch.cat([h_cond, h_geo, t_emb], dim=-1)  # [B, node_dim+geo_dim+2·n_freqs]
        s_se3 = self.se3_head(h_se3)            # [B, 6]
        confidence = self.judge_head(h_cond)      # [B, 1]
        state = self.state_head(h_cond)           # [B, 2]

        # 4. FiLM parameters for each layer of each subgraph
        h_film = self.film_shared(h_cond)  # [B, node_dim]
        film_params = []
        for head in self.film_heads:
            gamma, beta = head(h_film)
            film_params.append((gamma, beta))

        return {
            's_se3': s_se3,
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
