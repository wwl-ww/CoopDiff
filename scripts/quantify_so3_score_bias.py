#!/usr/bin/env python
"""
Quantify the rotation score-target mismatch (Cause 1) — CPU only.

Hypothesis under test
---------------------
``sample_so3_noise`` (diffusion.py:619) samples the rotation vector as an
ISOTROPIC R^3 GAUSSIAN  omega ~ N(0, sigma^2 I)  in the Lie algebra so(3).
But ``so3_score_vec`` (diffusion.py:782) / ``so3_score_norm`` (diffusion.py:766)
return the score / E[score^2] of the WRAPPED Gaussian-on-SO(3) density
(DiffDock's ``_score`` expansion), not the R^3-Gaussian score ``-omega/sigma^2``
and ``3/sigma^2``.

If the noise is drawn from one distribution but the target is the score of a
*different* distribution, the rotation score-matching target is systematically
biased. This script measures how big that bias is, over both the SE3 rot range
(sigma in [0.05, 1.0]) and the PROTAC rot range (sigma in [0.03, 1.55]).

Metrics (all evaluated over omega ~ N(0, sigma^2 I), the ACTUAL sampler):
  1. MSE(so3_score_vec - (-omega/sigma^2))       -> how wrong the target is.
  2. Stein residual  E[so3_score_vec * sigma^2 + omega]  -> 0 iff score_vec is
     the true score of the sampled R^3 Gaussian (integration-by-parts identity).
  3. so3_score_norm(sigma)^2  vs  3/sigma^2      -> E[score^2] comparison.
  4. zero-predictor rot-loss floor = E[||score_vec||^2] / norm^2  -> the loss a
     zero-init SE3 head sees; ~1.0 iff target & norm are self-consistent.

It also draws samples from the vendored WRAPPED sampler (so3.sample_vec) and
confirms that for *that* distribution ``score_vec`` is self-consistent
(E[||score_vec||^2]/norm^2 ~= 1), i.e. the correct pairing is
sample_vec <-> score_vec, and the repo is using the wrong sampler.

Run (CPU only, from project root):
    python scripts/quantify_so3_score_bias.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # SO(3) cache .npy files resolve relative to CWD

import numpy as np
import torch

from tricoopdiff.diffusion import so3_score_vec, so3_score_norm  # noqa: E402
from tricoopdiff.diffdock.utils.so3 import sample_vec as wrapped_sample_vec  # noqa: E402


N_R3 = 200_000      # R^3 Gaussian samples (vectorized)
N_WRAP = 40_000     # wrapped-SO(3) samples (reference distribution)


def stein_residual(s, sigma, omega):
    """E[s * sigma^2 + omega] — 0 iff s is the score of N(0, sigma^2 I)."""
    r = s * (sigma ** 2) + omega
    return r.mean(dim=0), r.norm(dim=-1).mean()


def run_sigma(sigma, wrap_samples=None):
    sig = float(sigma)
    sigma_t = torch.full((N_R3,), sig)

    # --- actual sampler: R^3 Gaussian (matches sample_so3_noise) ---
    omega = torch.randn(N_R3, 3) * sig
    s = so3_score_vec(sigma_t, omega)          # [N, 3] wrapped-SO(3) score
    r3_score = -omega / (sig ** 2)             # true score of N(0, sigma^2 I)

    mse = (s - r3_score).pow(2).sum(-1).mean().item()
    rel = ((s - r3_score).norm(dim=-1) / r3_score.norm(dim=-1)).mean().item()

    stein_mean, stein_mag = stein_residual(s, sig, omega)

    norm = so3_score_norm(torch.tensor([sig]))[0].item()
    norm_sq = norm ** 2
    r3_norm_sq = 3.0 / (sig ** 2)

    # zero-predictor rot-loss floor under the ACTUAL (R^3) sampler:
    #   mean(||score_vec||^2 / norm^2), norm^2 = E_wrapped[score^2]
    zero_floor = (s.pow(2).sum(-1) / norm_sq).mean().item()

    # --- reference: wrapped sampler <-> score_vec should be self-consistent ---
    wrap_floor = None
    if wrap_samples is not None:
        w = torch.tensor(np.stack(wrap_samples), dtype=torch.float32)  # [N,3]
        s_w = so3_score_vec(torch.full((len(w),), sig), w)
        wrap_floor = (s_w.pow(2).sum(-1) / norm_sq).mean().item()
        # empirical E[||omega||^2] of the two distributions for contrast
    emp_r3_norm_sq = omega.pow(2).sum(-1).mean().item()
    emp_wrap_norm_sq = (wrap_samples ** 2).sum(-1).mean() if wrap_samples is not None else None

    return {
        "sigma": sig,
        "mse_target_vs_r3": mse,
        "rel_err_target_vs_r3": rel,
        "stein_component": stein_mean.tolist(),
        "stein_magnitude": stein_mag.item(),
        "norm_sq": norm_sq,
        "r3_norm_sq_3over_sigma2": r3_norm_sq,
        "norm_sq_ratio": norm_sq / r3_norm_sq,
        "zero_floor_r3_sampler": zero_floor,
        "zero_floor_wrapped_sampler": wrap_floor,
        "emp_r3_norm_sq": emp_r3_norm_sq,
        "emp_wrap_norm_sq": emp_wrap_norm_sq,
    }


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    se3_rot_sigmas = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    protac_rot_sigmas = [0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.3, 1.55]

    def hdr(title):
        print()
        print("=" * 108)
        print(title)
        print("=" * 108)

    hdr("SE3 ROTATION RANGE  (sigma in [0.05, 1.0])")
    print(f"{'sigma':>6} | {'MSE vs -w/s^2':>15} | {'rel err':>8} | "
          f"{'Stein|E[s*s2+w]|':>17} | {'norm^2':>9} | {'3/s^2':>9} | "
          f"{'norm^2/(3/s^2)':>14} | {'floor(R3)':>10} | {'floor(wrap)':>12}")
    print("-" * 108)

    results = {}
    for sig in se3_rot_sigmas:
        wrap = np.stack([wrapped_sample_vec(sig) for _ in range(N_WRAP)])
        r = run_sigma(sig, wrap)
        results[sig] = r
        stein_comp = r["stein_component"]
        print(f"{sig:>6.3f} | {r['mse_target_vs_r3']:>15.4f} | {r['rel_err_target_vs_r3']:>8.3f} | "
              f"{r['stein_magnitude']:>17.4f} | {r['norm_sq']:>9.4f} | {r['r3_norm_sq_3over_sigma2']:>9.4f} | "
              f"{r['norm_sq_ratio']:>14.4f} | {r['zero_floor_r3_sampler']:>10.4f} | "
              f"{r['zero_floor_wrapped_sampler']:>12.4f}")

    hdr("PROTAC ROTATION RANGE  (sigma in [0.03, 1.55])")
    print(f"{'sigma':>6} | {'MSE vs -w/s^2':>15} | {'rel err':>8} | "
          f"{'Stein|E[s*s2+w]|':>17} | {'norm^2':>9} | {'3/s^2':>9} | "
          f"{'norm^2/(3/s^2)':>14} | {'floor(R3)':>10} | {'floor(wrap)':>12}")
    print("-" * 108)

    for sig in protac_rot_sigmas:
        wrap = np.stack([wrapped_sample_vec(sig) for _ in range(N_WRAP)])
        r = run_sigma(sig, wrap)
        results[sig] = r
        print(f"{sig:>6.3f} | {r['mse_target_vs_r3']:>15.4f} | {r['rel_err_target_vs_r3']:>8.3f} | "
              f"{r['stein_magnitude']:>17.4f} | {r['norm_sq']:>9.4f} | {r['r3_norm_sq_3over_sigma2']:>9.4f} | "
              f"{r['norm_sq_ratio']:>14.4f} | {r['zero_floor_r3_sampler']:>10.4f} | "
              f"{r['zero_floor_wrapped_sampler']:>12.4f}")

    # ------------------------------------------------------------------
    # Diagnostic: the two samplers draw from DIFFERENT distributions.
    # ------------------------------------------------------------------
    hdr("SAMPLER DISTRIBUTION MISMATCH  (E[||omega||^2] : R^3-Gaussian vs wrapped)")
    print(f"{'sigma':>6} | {'E[||w||^2] R^3':>14} | {'E[||w||^2] wrapped':>17} | {'ratio':>8}")
    print("-" * 60)
    for sig in se3_rot_sigmas + protac_rot_sigmas:
        r = results[sig]
        print(f"{sig:>6.3f} | {r['emp_r3_norm_sq']:>14.4f} | {r['emp_wrap_norm_sq']:>17.4f} | "
              f"{r['emp_r3_norm_sq']/r['emp_wrap_norm_sq']:>8.3f}")

    print()
    print("Interpretation:")
    print("  - 'MSE vs -w/s^2' and 'rel err' quantify how far the wrapped target is")
    print("    from the true R^3-Gaussian score at that sigma (0 => consistent).")
    print("  - 'Stein|E[s*s2+w]|' is the bias magnitude; exactly 0 iff score_vec is")
    print("    the true score of the sampled R^3 Gaussian.")
    print("  - 'floor(R3)' is the rot-loss floor a zero-init head sees under the")
    print("    ACTUAL sampler; ~1.0 means target and norm are self-consistent.")
    print("  - 'floor(wrap)' should be ~1.0 for every sigma (self-consistent pairing).")


if __name__ == "__main__":
    main()
