import numpy as np
import tqdm
import os
import warnings
from pathlib import Path

"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""


def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

_CACHE_DIR = Path(os.environ.get(
    "TRICOOPDIFF_CACHE_DIR",
    Path(__file__).resolve().parents[3] / "results/runtime_cache/diffdock",
))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_P_FILE = _CACHE_DIR / '.p.npy'
_SCORE_FILE = _CACHE_DIR / '.score.npy'

if _P_FILE.exists() and _SCORE_FILE.exists():
    p_ = np.load(_P_FILE)
    score_ = np.load(_SCORE_FILE)
else:
    p_ = p(x, sigma[:, None], N=100)
    np.save(_P_FILE, p_)

    eps = np.finfo(p_.dtype).eps
    score_ = grad(x, sigma[:, None], N=100) / (p_ + eps)
    np.save(_SCORE_FILE, score_)


def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


SCORE_NORM_VERSION = 'wrapped_normal_quadrature_v1'


def validate_score_norm_version(runtime_config):
    """Reject known normalization mismatches; identify unrecorded legacy tables."""
    saved = runtime_config.get('torus_score_norm_version')
    if saved is None:
        warnings.warn(
            'Checkpoint has no torus normalization provenance. Its original '
            'random Monte Carlo table cannot be recovered; using deterministic '
            'quadrature, not an exact legacy numerical replay.',
            RuntimeWarning, stacklevel=2,
        )
        return 'legacy_unrecorded'
    if saved != SCORE_NORM_VERSION:
        raise ValueError(f'checkpoint torus normalization {saved!r} does not match '
                         f'runtime {SCORE_NORM_VERSION!r}')
    return saved


def deterministic_score_norm(density, scores, positive_grid, chunk_size=128):
    """Integrate E[score²] on the existing wrapped-normal grid.

    Both density and score² are even, so the factor two on [0, pi] cancels
    between numerator and normalizer. The missing zero endpoint has score 0.
    Work in chunks rather than allocating a 10,000-by-5,001 Monte Carlo array.
    This function never consumes RNG state.
    """
    if density.shape != scores.shape or density.shape[1] != len(positive_grid):
        raise ValueError('density, score, and grid shapes must agree')
    if chunk_size < 1 or np.any(np.diff(positive_grid) <= 0) or positive_grid[0] <= 0:
        raise ValueError('positive increasing grid and positive chunk size required')
    result = np.empty(density.shape[0], dtype=np.float64)
    for start in range(0, len(result), chunk_size):
        stop = min(start + chunk_size, len(result))
        density_chunk = density[start:stop]
        integrand = scores[start:stop] ** 2 * density_chunk
        numerator = np.trapz(integrand, positive_grid, axis=1)
        numerator += .5 * positive_grid[0] * integrand[:, 0]
        denominator = np.trapz(density_chunk, positive_grid, axis=1)
        denominator += positive_grid[0] * density_chunk[:, 0]
        result[start:stop] = numerator / denominator
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FloatingPointError('invalid deterministic torus score normalizer')
    return result


score_norm_ = deterministic_score_norm(p_, score_, x)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
