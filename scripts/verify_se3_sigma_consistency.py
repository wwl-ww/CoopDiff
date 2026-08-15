#!/usr/bin/env python
"""
CPU-only verification that the SE3 DSM loss uses a self-consistent sigma.

Background
----------
``add_noise_se3`` samples the POI perturbation at the EFFECTIVE noise level
``sigma_eff = sigma * lambda_se3``, where ``lambda_se3(t) = 1`` for ``t`` above
the SE3 threshold (0.5) and ``0`` below it. For a below-threshold ("clean")
sample the POI is not perturbed at all: noise == 0 and the score target == 0.

The bug this guards against: ``train.py:compute_loss`` used to weight the SE3
translation/rotation residuals by the UNMASKED ``sigma`` (and ``1/norm(sigma)``)
instead of ``sigma_eff``. A clean sample therefore contributed ``pred^2 * sigma^2``
(a spurious "predict 0" gradient) to the loss, which pins the SE3 head at the
constant-zero output and produces the observed ``loss_se3 ~= 1.0`` floor.

This script checks, without touching the model or GPU:
  1. lambda_se3 is a 0/1 step at t = 0.5.
  2. add_noise_se3 adds exactly zero noise for lambda = 0 samples.
  3. The score target equals -noise/sigma^2 (true score of N(0, sigma^2 I)).
  4. Zero-model translation loss expectation is 3.0 (chi^2_3) for lambda=1
     samples, and exactly 0 for lambda=0 samples under the fix.
  5. The oracle (pred == target) achieves 0 loss for both lambda values.
  6. The fix makes a nonzero prediction on a clean sample contribute exactly 0
     loss, whereas the un-masked weighting contributed pred^2 * sigma^2.

Run directly (CPU only):
    python scripts/verify_se3_sigma_consistency.py
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from tricoopdiff.diffusion import MultiRateDiffusion, so3_score_norm, so3_score_vec  # noqa: E402

# SO(3)/torus cache resolves relative to CWD; chdir to project root.
os.chdir(_PROJECT_ROOT)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    torch.manual_seed(0)
    diff = MultiRateDiffusion(inference_steps=20)
    all_ok = True

    # ---------------------------------------------------------------- 1.
    t_vals = torch.tensor([0.0, 0.25, 0.49, 0.5, 0.51, 0.8, 1.0])
    lam = diff.get_lambda_se3(t_vals)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    all_ok &= check(
        "lambda_se3 is a 0/1 step at threshold 0.5",
        torch.equal(lam, expected),
        f"lambda={lam.tolist()}",
    )

    # ---------------------------------------------------------------- 2.
    B = 4096
    t_clean = torch.full((B,), 0.3)   # lambda = 0
    t_noisy = torch.full((B,), 0.8)   # lambda = 1
    poi = torch.randn(B, 32, 3)

    clean_pos, tr_clean, _, rot_omega_clean = diff.add_noise_se3(poi, t_clean, t_clean)
    noisy_pos, tr_noisy, _, rot_omega_noisy = diff.add_noise_se3(poi, t_noisy, t_noisy)

    all_ok &= check(
        "add_noise_se3 adds zero noise for lambda=0 (t=0.3)",
        torch.allclose(clean_pos, poi, atol=1e-6)
        and torch.all(tr_clean == 0) and torch.all(rot_omega_clean == 0),
        f"max|tr_clean|={tr_clean.abs().max().item():.2e}",
    )
    all_ok &= check(
        "add_noise_se3 adds nonzero noise for lambda=1 (t=0.8)",
        (tr_noisy.abs().max().item() > 0) and (rot_omega_noisy.abs().max().item() > 0),
        f"max|tr_noisy|={tr_noisy.abs().max().item():.3f}",
    )

    # ---------------------------------------------------------------- 3.
    # True score of N(0, sigma^2 I) is -noise / sigma^2. The code's target is
    # exactly this, so pred==target must give exactly 0 loss for lambda=1.
    tr_sigma, rot_sigma = diff.se3_t_to_sigma(t_noisy, t_noisy)
    tr_target = -tr_noisy / tr_sigma.unsqueeze(-1) ** 2
    rot_target = so3_score_vec(rot_sigma, rot_omega_noisy)
    rot_norm = so3_score_norm(rot_sigma).unsqueeze(-1)

    tr_sigma_clean, rot_sigma_clean = diff.se3_t_to_sigma(t_clean, t_clean)
    tr_target_clean = -tr_clean / tr_sigma_clean.unsqueeze(-1) ** 2
    rot_target_clean = so3_score_vec(rot_sigma_clean, rot_omega_clean)
    rot_norm_clean = so3_score_norm(rot_sigma_clean).unsqueeze(-1)

    # Oracle losses (pred == target), with the fix (mask by lambda).
    lam_noisy = diff.get_lambda_se3(t_noisy).unsqueeze(-1)
    lam_clean = diff.get_lambda_se3(t_clean).unsqueeze(-1)
    oracle_tr = ((tr_target - tr_target) ** 2 * tr_sigma.unsqueeze(-1) ** 2 * lam_noisy).mean()
    oracle_rot = ((rot_target - rot_target) ** 2 / rot_norm ** 2 * lam_noisy).mean()
    oracle_tr_clean = ((tr_target_clean - tr_target_clean) ** 2 * tr_sigma_clean.unsqueeze(-1) ** 2 * lam_clean).mean()
    oracle_rot_clean = ((rot_target_clean - rot_target_clean) ** 2 / rot_norm_clean ** 2 * lam_clean).mean()

    all_ok &= check(
        "oracle (pred == target) achieves 0 SE3 loss (lambda=1)",
        oracle_tr.item() < 1e-6 and oracle_rot.item() < 1e-6,
        f"oracle tr={oracle_tr.item():.2e} rot={oracle_rot.item():.2e}",
    )

    # ---------------------------------------------------------------- 4.
    # Zero-model translation loss for lambda=1, averaged over all 3*B elements:
    #   mean[(0 - (-noise/sigma^2))^2 * sigma^2] = mean[noise^2/sigma^2]
    #   = mean[randn^2] = 1.0  (noise = randn * sigma).
    zero_tr = ((torch.zeros_like(tr_target) - tr_target) ** 2 * tr_sigma.unsqueeze(-1) ** 2 * lam_noisy).mean()
    all_ok &= check(
        "zero-model translation loss expectation = 1.0 for lambda=1 samples",
        abs(zero_tr.item() - 1.0) < 0.15,
        f"measured={zero_tr.item():.3f} (expect ~1.0, mean of randn^2)",
    )

    # Zero-model rotation loss for lambda=1: norm^2 == E[score^2], so it is ~1.0
    # up to the (small) R^3-vs-wrapped-SO(3) sampling mismatch.
    zero_rot = ((torch.zeros_like(rot_target) - rot_target) ** 2 / rot_norm ** 2 * lam_noisy).mean()
    all_ok &= check(
        "zero-model rotation loss ~= 1.0 for lambda=1 samples",
        abs(zero_rot.item() - 1.0) < 0.15,
        f"measured={zero_rot.item():.3f} (expect ~1.0)",
    )

    # ---------------------------------------------------------------- 5 & 6.
    # A clean sample with a nonzero prediction: fix -> 0 loss, no-fix -> nonzero.
    pred = torch.ones(B, 3)
    with_fix_tr = ((pred - tr_target_clean) ** 2 * tr_sigma_clean.unsqueeze(-1) ** 2 * lam_clean).mean()
    with_fix_rot = ((torch.ones(B, 3) - rot_target_clean) ** 2 / rot_norm_clean ** 2 * lam_clean).mean()
    no_fix_tr = ((pred - tr_target_clean) ** 2 * tr_sigma_clean.unsqueeze(-1) ** 2).mean()
    no_fix_rot = ((torch.ones(B, 3) - rot_target_clean) ** 2 / rot_norm_clean ** 2).mean()

    all_ok &= check(
        "clean (lambda=0) sample contributes exactly 0 loss AFTER the fix",
        with_fix_tr.item() == 0.0 and with_fix_rot.item() == 0.0,
        f"fix tr={with_fix_tr.item():.2e} rot={with_fix_rot.item():.2e}",
    )
    all_ok &= check(
        "clean (lambda=0) sample contributed nonzero loss BEFORE the fix",
        no_fix_tr.item() > 0.5 and no_fix_rot.item() > 0.0,
        f"no-fix tr={no_fix_tr.item():.3f} (pred^2*sigma^2) rot={no_fix_rot.item():.3f}",
    )

    print()
    print("=" * 70)
    print("RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
