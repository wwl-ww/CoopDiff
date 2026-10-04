"""
Frozen DiffDock two-body expert wrapper.

Wraps DiffDock's CGModel with all parameters frozen. Exposes per-layer outputs
for FiLM modulation injection. The expert processes two physically disconnected
subgraphs (E3-L and POI-L) in separate forward passes.

Each subgraph contains one protein and ligand, preserving DiffDock's input
schema. This does not guarantee that generated poses are in-distribution.
"""

import warnings
import math
import os
import json
import hashlib

import torch
import torch.nn as nn
import numpy as np
from e3nn import o3

# Vendored DiffDock so3/torus score norms — single source of truth so the
# FiLM-path score computation matches native CGModel.forward() exactly
# (required for zero_film_verification to pass).
from tricoopdiff.diffdock.utils import so3, torus
from tricoopdiff.runtime import sha256_file
from tricoopdiff.geometry.equivariant_clamp import validate_clamp_mode


EXPERT_SIGMA_BOUNDS = ((0.1, 19.0), (0.03, 1.55), (0.0314, 3.14))


def configure_expert_policy(expert, limit, mode):
    """Own numerical stabilization on the expert instance, never process state."""
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('expert activation clamp must be finite and positive')
    model = getattr(expert, 'model', expert)
    model.activation_clamp = float(limit)
    model.activation_clamp_mode = validate_clamp_mode(mode)


def expert_runtime_identity(expert):
    """Generation identity includes configuration and actual numerical policy."""
    model = getattr(expert, 'model', None)
    return {**expert_base_identity(expert),
            'model_parameters_sha256': getattr(expert, 'provenance', {}).get('model_parameters_sha256'),
            'effective_model_config_sha256': getattr(expert, 'provenance', {}).get('effective_model_config_sha256'),
            'activation_clamp': getattr(model, 'activation_clamp', None),
            'activation_clamp_mode': getattr(model, 'activation_clamp_mode', 'componentwise'),
            'sigma_bounds': [list(pair) for pair in EXPERT_SIGMA_BOUNDS]}


def validate_expert_sigma_args(args):
    """Reject configurations that disagree with the production preconditioner."""
    for name, bounds in zip(('tr', 'rot', 'tor'), EXPERT_SIGMA_BOUNDS):
        for suffix, expected in zip(('min', 'max'), bounds):
            value = getattr(args, f'{name}_sigma_{suffix}', None)
            if value != expected:
                raise ValueError(f'incompatible expert {name}_sigma_{suffix}: {value}; expected {expected}')


def expert_t_to_sigma(t_tr, t_rot, t_tor):
    """DiffDock-L input preconditioning, not a ternary diffusion process."""
    bounds = EXPERT_SIGMA_BOUNDS
    return tuple(low * (high / low) ** time
                 for time, (low, high) in zip((t_tr, t_rot, t_tor), bounds))


def so3_score_norm(sigma):
    """Reuse the expert's SO(3) table; keep native float32 output units."""
    value = torch.as_tensor(sigma)
    return so3.score_norm(value.detach().cpu().reshape(-1)).reshape(value.shape).to(value.device)


def torus_score_norm(sigma):
    """Torsion score norm — matches DiffDock's torus.score_norm lookup table.

    NOT 1/sigma^2: that approximation holds only for small sigma and diverges
    from the wrapped-Gaussian score norm as sigma approaches pi (relevant at
    high noise during diffusion, where tor_sigma reaches 1.55 rad).
    """
    return torus.score_norm(np.asarray(sigma))


class FrozenDiffDock(nn.Module):
    """
    Wrapper around DiffDock's CGModel that:
    1. Freezes all parameters (no gradient)
    2. Exposes per-layer outputs for FiLM injection
    3. Provides a step_conv_layer() interface for external orchestration

    The TriCoopDiff model orchestrates:
    - Running the embedding phase once
    - Running each conv layer individually, applying FiLM between layers
    - Running the final score heads
    """

    def __init__(self, cg_model: nn.Module, num_conv_layers: int = 11):
        """
        Args:
            cg_model: DiffDock's CGModel with pretrained weights loaded
            num_conv_layers: number of conv layers (11 for DiffDock-L, NOT 6)
        """
        super().__init__()
        self.model = cg_model
        # Snapshot legacy environment defaults only once. Production owners
        # subsequently configure this instance explicitly.
        self.model.activation_clamp = (float(os.environ['NAN_CLAMP'])
                                       if os.environ.get('NAN_CLAMP') else None)
        self.model.activation_clamp_mode = validate_clamp_mode(
            os.environ.get('NAN_CLAMP_MODE', 'componentwise'))
        self.num_conv_layers = num_conv_layers

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model references for convenience
        self.ns = self.model.ns
        self.nv = self.model.nv
        self.sh_irreps = self.model.sh_irreps
        self.device = self.model.device

    def embedding(self, data):
        """
        Run the embedding phase of CGModel.
        Returns initial ligand and receptor node features plus edge info.

        This is a thin wrapper around CGModel.embedding().
        """
        return self.model.embedding(data)

    def train(self, mode: bool = True):
        """Keep the frozen expert in evaluation mode.

        ``requires_grad=False`` prevents parameter updates but does not disable
        dropout or other training-mode behavior. Parent ``model.train()`` calls
        recurse into children, so override it here to keep the pretrained expert
        deterministic while still allowing TriFuse to train.
        """
        super().train(False)
        self.model.eval()
        return self

    def build_cross_graph(self, data, cross_cutoff):
        """Build cross edges between ligand and receptor."""
        return self.model.build_cross_conv_graph(data, cross_cutoff)

    def build_center_graph(self, data):
        """Build center convolution graph for tr/rot score prediction."""
        return self.model.build_center_conv_graph(data)

    def build_bond_graph(self, data):
        """Build bond convolution graph for torsion score prediction."""
        return self.model.build_bond_conv_graph(data)

    def step_conv_layer(
        self,
        layer_idx: int,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr,
        edge_sh: torch.Tensor,
        edge_weight=None,
    ) -> torch.Tensor:
        """
        Run a single conv layer. This is the core per-layer interface
        that allows FiLM modulation between layers.

        Args:
            layer_idx: which conv layer to run (0-indexed)
            node_attr: [N_total, D] current node features (ligand + receptor)
            edge_index: [2, E] combined edge index
            edge_attr: edge features (tensor or list of tensors for multi-group)
            edge_sh: spherical harmonic edge embeddings
            edge_weight: optional edge weights

        Returns:
            updated node_attr [N_total, D']
        """
        return self.model.conv_layers[layer_idx](
            node_attr, edge_index, edge_attr, edge_sh, edge_weight=edge_weight
        )

    def compute_scores(
        self,
        data,
        lig_node_attr: torch.Tensor,
        graph_sigma_emb: torch.Tensor,
        tr_sigma: torch.Tensor,
        rot_sigma: torch.Tensor,
        tor_sigma: torch.Tensor,
    ) -> dict:
        """
        Compute final tr/rot/torsion scores from ligand node features.

        Returns:
            dict with keys: 'tr_score', 'rot_score', 'tor_score'
        """
        model = self.model

        # Build center convolution for tr/rot scores
        center_edge_index, center_edge_attr, center_edge_sh = model.build_center_conv_graph(data)
        center_edge_attr = model.center_edge_embedding(center_edge_attr)
        if model.fixed_center_conv:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1
            )
        else:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1
            )
        global_pred = model.final_conv(
            lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh,
            out_nodes=data.num_graphs
        )

        tr_pred = global_pred[:, :3] + (
            global_pred[:, 6:9] if not model.odd_parity else 0
        )
        rot_pred = global_pred[:, 3:6] + (
            global_pred[:, 9:] if not model.odd_parity else 0
        )

        # Fix magnitude of tr/rot scores. The vendored CGModel uses the same
        # normalization pattern but does not guard zero norms; in TriCoopDiff that
        # can surface during smoke/training as NaNs in score loss / Tweedie phys
        # loss when a pathological graph makes final_conv emit an all-zero vector.
        # Keep the raw norm for the magnitude head (native semantics) but clamp the
        # divisor so direction normalization itself never divides by zero.
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_unit = tr_pred / tr_norm.clamp_min(1e-8)
        tr_pred = tr_unit * model.tr_final_layer(
            torch.cat([tr_norm, graph_sigma_emb], dim=1)
        )
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_unit = rot_pred / rot_norm.clamp_min(1e-8)
        rot_pred = rot_unit * model.rot_final_layer(
            torch.cat([rot_norm, graph_sigma_emb], dim=1)
        )

        if model.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = (
                rot_pred
                * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(lig_node_attr.device)
            )

        # Torsion scores
        if model.no_torsion or data['ligand'].edge_mask.sum() == 0:
            tor_pred = torch.empty(0, device=lig_node_attr.device)
        else:
            tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = \
                model.build_bond_conv_graph(data)
            tor_bond_vec = (
                data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
            )
            tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

            tor_bonds_sh = o3.spherical_harmonics(
                "2e", tor_bond_vec, normalize=True, normalization='component'
            )
            tor_edge_sh = model.final_tp_tor(
                tor_edge_sh, tor_bonds_sh[tor_edge_index[0]]
            )

            tor_edge_attr = torch.cat([
                tor_edge_attr,
                lig_node_attr[tor_edge_index[1], :self.ns],
                tor_bond_attr[tor_edge_index[0], :self.ns],
            ], -1)
            tor_pred = model.tor_bond_conv(
                lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                out_nodes=data['ligand'].edge_mask.sum(), reduce='mean',
                edge_weight=tor_edge_weight
            )
            tor_pred = model.tor_final_layer(tor_pred).squeeze(1)
            # Align with patched cg_model.py: index sigma by the source atoms of
            # the rotatable bonds (tor_bonds[0]), which is guaranteed to match
            # tor_pred's out_nodes, rather than the edge_index→edge_mask chain.
            edge_sigma = tor_sigma[data['ligand'].batch][tor_bonds[0]]

            if model.scale_by_sigma:
                tor_pred = tor_pred * torch.sqrt(
                    torch.tensor(
                        torus_score_norm(edge_sigma.cpu().numpy())
                    ).float().to(lig_node_attr.device)
                )

        return {
            'tr_score': tr_pred,
            'rot_score': rot_pred,
            'tor_score': tor_pred,
        }

    def forward(self, data):
        """
        Standard forward (without FiLM). Used for verification that
        FiLM is transparent when zero-initialized and for L0 ablation.
        """
        return self.model(data)


def _set_time_embeddings(data, t, model):
    """
    Set diffusion time information on the HeteroData object, matching DiffDock's
    `set_time` semantics: `complex_t` holds per-graph [batchsize] tensors and
    `ligand.node_t` holds per-node [num_nodes] tensors. CGModel's
    `timestep_emb_func` (GaussianFourierProjection) requires 1-D tensor inputs,
    so these must be tensors, not Python floats.
    """
    device = data['ligand'].pos.device
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=torch.float32)
    t = t.to(device).reshape(-1)  # [1] (single-complex mode)

    batchsize = int(data.num_graphs)
    num_nodes = data['ligand'].pos.shape[0]

    data.complex_t = {
        'tr': t.expand(batchsize),
        'rot': t.expand(batchsize),
        'tor': t.expand(batchsize),
    }
    data['ligand'].node_t = {
        'tr': t.expand(num_nodes),
        'rot': t.expand(num_nodes),
        'tor': t.expand(num_nodes),
    }
    if 'receptor' in data.node_types:
        num_rec_nodes = data['receptor'].pos.shape[0]
        data['receptor'].node_t = {
            'tr': t.expand(num_rec_nodes),
            'rot': t.expand(num_rec_nodes),
            'tor': t.expand(num_rec_nodes),
        }


def build_diffdock_expert(
    model_dir: str = "checkpoints/score_model",
    device: str = "cuda",
    ckpt: str = "best_ema_inference_epoch_model.pt",
    num_conv_layers: int = None,
    crop_beyond="native",
) -> FrozenDiffDock:
    """
    Build and freeze a DiffDock score model from pretrained weights.

    Args:
        model_dir: directory containing `model_parameters.yml` and the checkpoint.
        device: 'cuda' or 'cpu'.
        ckpt: checkpoint filename inside model_dir.
        num_conv_layers: override (defaults to the value in model_parameters.yml;
            11 for DiffDock-L).
        crop_beyond: receptor-crop policy for the frozen expert. ``"native"``
            (default) preserves the checkpoint's `model_parameters.yml` setting so
            the expert stays on vendored DiffDock's native inference path.
            Pass ``None`` to force the old full-receptor fallback, or a numeric
            value to override the crop radius explicitly.

    Returns:
        FrozenDiffDock wrapping the loaded CGModel (all params frozen).
    """
    from argparse import Namespace
    from functools import partial
    from pathlib import Path

    import yaml
    from tricoopdiff.diffdock.factory import build_cg_model
    from tricoopdiff.diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

    model_dir = Path(model_dir)
    with open(model_dir / "model_parameters.yml") as f:
        args = Namespace(**yaml.full_load(f))
    validate_expert_sigma_args(args)

    if num_conv_layers is None:
        num_conv_layers = getattr(args, "num_conv_layers", 11)

    # Keep the frozen expert on the same crop/dynamic-cross regime as the
    # vendored DiffDock checkpoint unless the caller explicitly overrides it.
    # Previous versions forced ``args.crop_beyond = None`` here, which silently
    # downgraded native inference from
    #   crop_beyond(...) + dynamic_max_cross=True
    # to
    #   full receptor + dynamic_max_cross=False,
    # creating a persistent parity gap for the zero-FiLM baseline.
    if crop_beyond == "native":
        pass
    elif crop_beyond is None:
        args.crop_beyond = None
    else:
        args.crop_beyond = crop_beyond

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    cg_model = build_cg_model(args, device, t_to_sigma)

    ckpt_path = model_dir / ckpt
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    cg_model.load_state_dict(state_dict, strict=True)
    cg_model = cg_model.to(device)
    cg_model.eval()

    expert = FrozenDiffDock(cg_model, num_conv_layers=num_conv_layers)
    expert.provenance = {
        "diffdock_commit": "47eb91a",
        "diffdock_source": "vendored with repository stability patches",
        "score_checkpoint": str(ckpt_path.resolve()),
        "score_checkpoint_sha256": sha256_file(ckpt_path),
        "model_parameters_sha256": sha256_file(model_dir / 'model_parameters.yml'),
        "effective_model_config_sha256": hashlib.sha256(
            json.dumps(vars(args), sort_keys=True).encode()).hexdigest(),
        "no_torsion": bool(getattr(args, "no_torsion", False)),
    }
    return expert


_EXPERT_IDENTITY_KEYS = ('diffdock_commit', 'score_checkpoint_sha256', 'no_torsion')


def expert_base_identity(expert):
    """Return the stable identity fields for the external frozen expert."""
    provenance = getattr(expert, 'provenance', {})
    missing = [key for key in _EXPERT_IDENTITY_KEYS if key not in provenance]
    if missing:
        raise ValueError(f'frozen expert has incomplete provenance: {missing}')
    return {key: provenance[key] for key in _EXPERT_IDENTITY_KEYS}


def validate_expert_base_identity(checkpoint, expert):
    """Reject known base/adapter mismatches and label legacy uncertainty."""
    actual = expert_base_identity(expert)
    saved = checkpoint.get('expert_base_identity')
    if saved is None:
        warnings.warn(
            'Checkpoint has no frozen-expert identity; compatibility with the '
            'runtime DiffDock base cannot be proven.', RuntimeWarning, stacklevel=2)
        return 'legacy_unrecorded'
    if saved != actual:
        raise ValueError(f'checkpoint frozen expert identity {saved!r} does not '
                         f'match runtime {actual!r}')
    return saved
