"""
Build a frozen DiffDock two-body expert from pretrained weights.

This is the single source of truth for constructing the frozen expert. It uses
the *vendored* DiffDock core in ``tricoopdiff/diffdock/`` (factory + models +
utils) rather than the sibling ``../DiffDock`` checkout, so building/running the
expert is self-contained. The architecture exactly matches the pretrained
checkpoint, so ``load_state_dict(strict=True)`` succeeds.

Usage:
    from tricoopdiff.expert_factory import build_diffdock_expert
    expert = build_diffdock_expert("checkpoints/score_model", "cuda")
"""

from pathlib import Path
from functools import partial

import torch
import yaml
from argparse import Namespace

# IMPORTANT: import diffdock_compat FIRST — its import-time side effect patches
# torch_cluster / torch_scatter before the vendored models.cg_model binds those
# symbols (this cluster installs them as stub packages whose native CUDA kernels
# fail to build/load).
from . import diffdock_compat  # noqa: F401  (applies monkey-patch on import)
from .expert import FrozenDiffDock
from .diffdock.factory import build_cg_model
from .diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl


def build_diffdock_expert(
    model_dir: str = "checkpoints/score_model",
    device: str = "cuda",
    ckpt: str = "best_ema_inference_epoch_model.pt",
    num_conv_layers: int = None,
    crop_beyond=None,
) -> FrozenDiffDock:
    """
    Build and freeze a DiffDock score model from pretrained weights.

    Args:
        model_dir: directory containing `model_parameters.yml` and the checkpoint.
        device: 'cuda' or 'cpu'.
        ckpt: checkpoint filename inside model_dir.
        num_conv_layers: override (defaults to the value in model_parameters.yml;
            11 for DiffDock-L).
        crop_beyond: pass None to disable dynamic receptor cropping (recommended
            for POI subgraphs where the ligand starts far from the protein and
            cropping can remove ALL residues → "No edges and no nodes").

    Returns:
        FrozenDiffDock wrapping the loaded CGModel (all params frozen).
    """
    model_dir = Path(model_dir)
    with open(model_dir / "model_parameters.yml") as f:
        args = Namespace(**yaml.full_load(f))

    if num_conv_layers is None:
        num_conv_layers = getattr(args, "num_conv_layers", 11)

    # Disable dynamic receptor cropping (see docstring).
    if crop_beyond is None and getattr(args, "crop_beyond", None) is not None:
        args.crop_beyond = None
    elif crop_beyond is not None:
        args.crop_beyond = crop_beyond

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    cg_model = build_cg_model(args, device, t_to_sigma)

    ckpt_path = model_dir / ckpt
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    cg_model.load_state_dict(state_dict, strict=True)
    cg_model = cg_model.to(device)
    cg_model.eval()

    return FrozenDiffDock(cg_model, num_conv_layers=num_conv_layers)


__all__ = ["build_diffdock_expert"]
