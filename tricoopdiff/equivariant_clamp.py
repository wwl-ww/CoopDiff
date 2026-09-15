"""Bound feature magnitudes without choosing a direction in an irrep space."""

from functools import lru_cache
import math

from e3nn import o3
import torch


CLAMP_MODES = ('componentwise', 'irrep_norm')


def validate_clamp_mode(mode):
    if mode not in CLAMP_MODES:
        raise ValueError(f'unsupported activation clamp mode: {mode!r}')
    return mode


@lru_cache(maxsize=128)
def _blocks(irreps):
    parsed = o3.Irreps(irreps)
    return parsed.dim, tuple((mul, ir.dim, section)
                            for (mul, ir), section in zip(parsed, parsed.slices()))


def clamp_irrep_features(features, irreps, limit):
    """Cap each irrep copy's Euclidean norm, preserving SO(3) equivariance.

    Scalar blocks reduce to ordinary symmetric clipping. Float64 accumulation
    avoids squaring large, finite float32 activations into infinity.
    """
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('activation clamp must be finite and positive')
    dimension, blocks = _blocks(str(irreps))
    if features.shape[-1] != dimension:
        raise ValueError('feature dimension does not match irreps')
    parts = []
    for multiplicity, width, section in blocks:
        values = features[..., section].reshape(*features.shape[:-1], multiplicity, width)
        norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True, dtype=torch.float64)
        factor = (limit / norm.clamp_min(1e-30)).clamp(max=1).to(values.dtype)
        parts.append((values * factor).reshape(*features.shape[:-1], multiplicity * width))
    return torch.cat(parts, dim=-1)
