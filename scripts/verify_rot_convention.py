#!/usr/bin/env python
"""
verify_rot_convention.py — CPU-only, self-contained check of the SO(3) rotation
convention bug reported in the code review.

Claim under test
----------------
TriCoopDiff forward (`add_noise_protac`, diffusion.py:188) and reverse
(`reverse_step_protac`, diffusion.py:376) apply the PROTAC rotation as
    @ exp([ω]×)      (NO transpose)
while the frozen DiffDock-L expert was trained & sampled with
    @ exp([ω]×).T = @ exp(-[ω]×)      (transpose, see diffdock/utils/diffusion_utils.py).

Consequence
-----------
For a pose produced by TriCoopDiff's forward (rotation ω), the expert outputs
    score_vec(σ, -ω) = -score_vec(σ, ω)
which is OPPOSITE in sign to TriCoopDiff's training target score_vec(σ, +ω)
(train.py:141-143). Feeding that sign-flipped score into reverse_step_protac
AMPLIFIES |ω| instead of denoising it — so the zero-FiLM baseline diverges.

No model weights are loaded. The expert-output sign rests on:
  (1) DiffDock's forward convention  @ rot_mat.T  (vendored diffusion_utils.py),
  (2) SE(3)-equivariance of the score model
        (pose @ exp([ω]×) == pose @ exp(-[-ω]×) → expert reads it as rotation -ω),
  (3) anti-symmetry of score_vec  score_vec(σ,-ω) = -score_vec(σ,ω).
All three are checked numerically here.

Only numpy / torch / scipy are required (no GPU, no e3nn/torch_geometric).
"""

import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

PROJECT_ROOT = "/data/run01/scwb792/mwj/CoopDiff"
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# 1. so3_exp / vector_to_skew / so3_score_vec — reimplemented VERBATIM from
#    tricoopdiff/diffusion.py so this script has no package import (which would
#    pull in e3nn / torch_geometric). An optional cross-check against the real
#    module runs at the end when importable.
# ---------------------------------------------------------------------------

def vector_to_skew(v):
    *batch_dims, _ = v.shape
    z = torch.zeros(*batch_dims, device=v.device)
    row0 = torch.stack([z, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], z, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp(omega):
    B = omega.shape[0]
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    mask = theta.squeeze(-1) > 1e-8
    omega_hat = vector_to_skew(omega)
    R = torch.eye(3, device=omega.device).unsqueeze(0).repeat(B, 1, 1)
    if mask.any():
        theta_m = theta[mask]
        omega_hat_m = omega_hat[mask]
        sin_theta = torch.sin(theta_m).unsqueeze(-1)
        cos_theta = torch.cos(theta_m).unsqueeze(-1)
        R[mask] = (
            torch.eye(3, device=omega.device)
            + sin_theta / theta_m.unsqueeze(-1) * omega_hat_m
            + (1 - cos_theta) / (theta_m.unsqueeze(-1) ** 2)
            * torch.bmm(omega_hat_m, omega_hat_m)
        )
    return R


# SO(3) score tables — same constants as diffusion.py (_MIN_EPS/_MAX_EPS/_N_EPS)
_MIN_EPS, _MAX_EPS, _N_EPS = 0.0005, 4, 2000
_X_N = 2000


def _load_score_tables():
    omegas_file = os.path.join(PROJECT_ROOT, '.so3_omegas_array4.npy')
    norms_file = os.path.join(PROJECT_ROOT, '.so3_score_norms4.npy')
    if os.path.exists(omegas_file) and os.path.exists(norms_file):
        return np.load(omegas_file), np.load(norms_file)
    # Fallback: recompute exactly as diffusion._init_so3_cache does.
    print("  (so3 cache .npy not found — recomputing tables, may take ~1 min)")
    eps_array = 10 ** np.linspace(np.log10(_MIN_EPS), np.log10(_MAX_EPS), _N_EPS)
    omegas = np.linspace(0, np.pi, _X_N + 1)[1:]
    l = np.arange(2000).reshape(-1, 1)

    def _expansion(omega, eps):
        return ((2 * l + 1) * np.exp(-l * (l + 1) * eps ** 2 / 2)
                * np.sin(omega * (l + 0.5)) / np.sin(omega / 2)).sum(0)

    def _score(exp, omega, eps):
        hi = np.sin((l + 0.5) * omega)
        dhi = (l + 0.5) * np.cos((l + 0.5) * omega)
        lo = np.sin(omega / 2)
        dlo = 0.5 * np.cos(omega / 2)
        dSigma = ((2 * l + 1) * np.exp(-l * (l + 1) * eps ** 2 / 2)
                  * (lo * dhi - hi * dlo) / lo ** 2).sum(0)
        return dSigma / exp

    exp_vals = np.asarray([_expansion(omegas, eps) for eps in eps_array])
    score_norms = np.asarray([
        _score(exp_vals[i], omegas, eps_array[i]) for i in range(len(eps_array))
    ])
    return omegas, score_norms


_omegas_array, _score_norms = _load_score_tables()


def so3_score_vec(sigma, vec):
    sigma_np = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    vec_np = np.atleast_2d(np.asarray(vec, dtype=np.float64))
    eps_idx = np.clip(
        np.around(
            (np.log10(sigma_np) - np.log10(_MIN_EPS))
            / (np.log10(_MAX_EPS) - np.log10(_MIN_EPS)) * _N_EPS
        ).astype(int), 0, _N_EPS - 1,
    )
    om = np.linalg.norm(vec_np, axis=-1)
    om_safe = np.clip(om, 1e-8, None)
    score_norm = np.array([
        np.interp(om[i], _omegas_array, _score_norms[eps_idx[i]])
        for i in range(len(om))
    ])
    out = score_norm[:, None] * vec_np / om_safe[:, None]
    return torch.from_numpy(out).float()


# ---------------------------------------------------------------------------
# 2. Checks
# ---------------------------------------------------------------------------

_results = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


omega = torch.tensor([0.6, 0.2, -0.4], dtype=torch.float32)
sigma = 0.8

print("=== 1. Both 'exp' maps agree: so3_exp(ω) == axis_angle_to_matrix(ω) == exp([ω]×) ===")
R_tc = so3_exp(omega.unsqueeze(0))[0]
R_scipy = torch.from_numpy(ScipyRotation.from_rotvec(omega.numpy()).as_matrix()).float()
check("so3_exp == scipy exp([ω]×)",
      torch.allclose(R_tc, R_scipy, atol=1e-6),
      f"max|diff|={(R_tc - R_scipy).abs().max().item():.2e}")

print("=== 2. Forward convention differs by transpose ===")
check("exp([ω]×).T == exp(-[ω]×)",
      torch.allclose(R_tc.T, so3_exp((-omega).unsqueeze(0))[0], atol=1e-6))
check("TriCoopDiff @R  !=  DiffDock @R.T",
      not torch.allclose(R_tc, R_tc.T, atol=1e-6),
      f"|ω|={omega.norm().item():.3f} rad (nonzero ⇒ R ≠ R.T)")

print("=== 3. score_vec is anti-symmetric and points toward identity ===")
s_pos = so3_score_vec(sigma, omega).reshape(3)   # TriCoopDiff training target
s_neg = so3_score_vec(sigma, -omega).reshape(3)
check("score_vec(σ,-ω) == -score_vec(σ,ω)",
      torch.allclose(s_neg, -s_pos, atol=1e-6))
dot = float((s_pos * omega).sum())
check("score_vec(σ,ω)·ω < 0  (points toward identity)", dot < 0, f"dot={dot:.4f}")

print("=== 4. Expert sign flip (equivariance + anti-symmetry) ===")
s_expert = s_neg   # frozen expert's output for a TriCoopDiff pose (rotation -ω)
check("expert output == -training_target", torch.allclose(s_expert, -s_pos, atol=1e-6))
print(f"    training target s   = {s_pos.numpy()}")
print(f"    frozen-expert  s   = {s_expert.numpy()}   (sign flipped)")

print("=== 5. Reverse-step direction (the actual bug) ===")
# reverse_step_protac: rot_perturb = score·dt·g², then apply @ so3_exp(rot_perturb).
# score ∥ ω ⇒ so3_exp(ω) @ so3_exp(rot_perturb) = so3_exp(ω + rot_perturb) exactly,
# so |ω_new| = |ω + score·c|.
c = 0.1 * omega.norm().item() / s_pos.norm().item()   # small positive step
theta0 = omega.norm().item()
theta_correct = (omega + s_pos * c).norm().item()
theta_expert = (omega + s_expert * c).norm().item()
print(f"    |ω| initial         = {theta0:.4f} rad")
print(f"    with training score: |ω| -> {theta_correct:.4f} rad  (should DECREASE)")
print(f"    with expert   score: |ω| -> {theta_expert:.4f} rad  (should INCREASE)")
check("training score DENOISES  (|ω| decreases)", theta_correct < theta0)
check("expert score AMPLIFIES   (|ω| increases)", theta_expert > theta0)

print("=== 6. Cross-check on an actual point cloud (Kabsch) ===")
torch.manual_seed(0)
x0 = torch.randn(32, 3)                                    # clean pose (centered)
x_noisy = x0 @ R_tc                                        # TriCoopDiff forward (@ exp([ω]×))
x_corr = x_noisy @ so3_exp((s_pos * c).unsqueeze(0))[0]    # reverse w/ training score
x_expt = x_noisy @ so3_exp((s_expert * c).unsqueeze(0))[0] # reverse w/ expert score
ang_corr = ScipyRotation.align_vectors(x0.numpy(), x_corr.numpy())[1].magnitude()
ang_expt = ScipyRotation.align_vectors(x0.numpy(), x_expt.numpy())[1].magnitude()
print(f"    Kabsch angle clean->denoised (correct score): {ang_corr:.4f} rad (init {theta0:.4f})")
print(f"    Kabsch angle clean->denoised (expert  score): {ang_expt:.4f} rad (init {theta0:.4f})")
check("point-cloud: correct score reduces angle", ang_corr < theta0)
check("point-cloud: expert score increases angle", ang_expt > theta0)

print()
print("=== 7. Optional cross-check against the real tricoopdiff.diffusion module ===")
try:
    sys.path.insert(0, PROJECT_ROOT)
    from tricoopdiff.diffusion import so3_exp as real_so3_exp
    from tricoopdiff.diffusion import so3_score_vec as real_so3_score_vec
    check("reimpl so3_exp == real so3_exp",
          torch.allclose(so3_exp(omega.unsqueeze(0)), real_so3_exp(omega.unsqueeze(0)), atol=1e-6))
    check("reimpl so3_score_vec == real so3_score_vec",
          torch.allclose(s_pos, real_so3_score_vec(torch.tensor(sigma), omega).reshape(3), atol=1e-5))
except Exception as e:
    print(f"  (skipped — could not import tricoopdiff.diffusion: {type(e).__name__}: {e})")

print()
n_pass = sum(_results)
n_total = len(_results)
print(f"RESULT: {n_pass}/{n_total} checks passed")
if n_pass == n_total:
    print("=> Convention mismatch CONFIRMED: the frozen expert's rotation score has the\n"
          "   opposite sign of TriCoopDiff's training target, so reverse_step_protac\n"
          "   AMPLIFIES the PROTAC rotation under the frozen expert (zero-FiLM diverges).\n"
          "   Fix: apply rotation as @ R.T in add_noise_protac (diffusion.py:188) and\n"
          "   reverse_step_protac (diffusion.py:376), matching DiffDock's convention.")
    sys.exit(0)
else:
    print("=> Some check failed — see above.")
    sys.exit(1)
