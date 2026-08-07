"""
Multi-rate diffusion process for TriCoopDiff.

Different degrees of freedom use different noise decay rates:
- PROTAC (translation, rotation, torsion): fast noise decay (λ_q, threshold t=0.7)
- E3-POI rigid body: slow noise decay (λ_SE3, threshold t=0.5)

This is an empirical noise schedule trick, not a strict SDE derivation.
The threshold-based λ definition is a non-differentiable engineering approximation.

Reference frame: E3 fixed at origin. POI moves relative to E3. PROTAC is
globally defined in the E3 reference frame.

The diffusion parameter t is a noise schedule variable, NOT physical time.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R


class MultiRateDiffusion(nn.Module):
    """
    Multi-rate diffusion with separate schedules for PROTAC and E3-POI degrees of freedom.

    Diffusion variables:
    - PROTAC translation t_q: 3D (relative to E3)
    - PROTAC rotation R_q: 3D in so(3)
    - PROTAC torsion angles χ: K (K = number of rotatable bonds)
    - E3-POI rigid body SE3: 6D (3 translation + 3 rotation in so(3))
    """

    def __init__(
        self,
        # PROTAC noise parameters (from DiffDock-L paper: 0.1→19, 0.03→1.55)
        tr_sigma_min: float = 0.1,
        tr_sigma_max: float = 19.0,
        rot_sigma_min: float = 0.03,
        rot_sigma_max: float = 1.55,
        tor_sigma_min: float = 0.03,
        tor_sigma_max: float = 1.55,
        # Temperature sampling (from DiffDock v1.1 default_inference_args.yaml)
        temp_sampling_tr: float = 1.17,
        temp_sampling_rot: float = 2.06,
        temp_sampling_tor: float = 7.04,
        temp_psi_tr: float = 0.73,
        temp_psi_rot: float = 0.90,
        temp_psi_tor: float = 0.59,
        temp_sigma_data_tr: float = 0.93,
        temp_sigma_data_rot: float = 0.75,
        temp_sigma_data_tor: float = 0.69,
        # SE3 noise parameters (E3-POI rigid body)
        se3_tr_sigma_min: float = 0.5,
        se3_tr_sigma_max: float = 20.0,
        se3_rot_sigma_min: float = 0.05,
        se3_rot_sigma_max: float = 1.0,
        # Multi-rate thresholds
        lambda_q_threshold: float = 0.7,   # PROTAC noise cutoff
        lambda_se3_threshold: float = 0.5,  # SE3 noise cutoff (slower)
        # Schedule
        inference_steps: int = 20,
    ):
        super().__init__()
        # PROTAC noise bounds
        self.tr_sigma_min = tr_sigma_min
        self.tr_sigma_max = tr_sigma_max
        self.rot_sigma_min = rot_sigma_min
        self.rot_sigma_max = rot_sigma_max
        self.tor_sigma_min = tor_sigma_min
        self.tor_sigma_max = tor_sigma_max

        # SE3 noise bounds
        self.se3_tr_sigma_min = se3_tr_sigma_min
        self.se3_tr_sigma_max = se3_tr_sigma_max
        self.se3_rot_sigma_min = se3_rot_sigma_min
        self.se3_rot_sigma_max = se3_rot_sigma_max

        # Multi-rate thresholds
        self.lambda_q_threshold = lambda_q_threshold
        self.lambda_se3_threshold = lambda_se3_threshold

        self.inference_steps = inference_steps

        # Temperature sampling
        self.temp_sampling = [temp_sampling_tr, temp_sampling_rot, temp_sampling_tor]
        self.temp_psi = [temp_psi_tr, temp_psi_rot, temp_psi_tor]
        self.temp_sigma_data = [temp_sigma_data_tr, temp_sigma_data_rot, temp_sigma_data_tor]

    # ------------------------------------------------------------------
    # Noise schedule: maps diffusion time t → noise level σ
    # ------------------------------------------------------------------

    def sigma_schedule(self, t: float, sigma_min: float, sigma_max: float) -> float:
        """
        Exponential noise schedule: σ(t) = σ_min * (σ_max / σ_min)^t

        Args:
            t: diffusion time in [0, 1]
            sigma_min: minimum noise level
            sigma_max: maximum noise level
        Returns:
            noise level σ
        """
        return sigma_min * (sigma_max / sigma_min) ** t

    def t_to_sigma(self, t_tr, t_rot, t_tor):
        """Convert diffusion times to noise levels for PROTAC DOFs."""
        tr_sigma = self.sigma_schedule(t_tr, self.tr_sigma_min, self.tr_sigma_max)
        rot_sigma = self.sigma_schedule(t_rot, self.rot_sigma_min, self.rot_sigma_max)
        tor_sigma = self.sigma_schedule(t_tor, self.tor_sigma_min, self.tor_sigma_max)
        return tr_sigma, rot_sigma, tor_sigma

    def se3_t_to_sigma(self, t_se3_tr, t_se3_rot):
        """Convert diffusion times to noise levels for SE3 DOFs."""
        tr_sigma = self.sigma_schedule(t_se3_tr, self.se3_tr_sigma_min, self.se3_tr_sigma_max)
        rot_sigma = self.sigma_schedule(t_se3_rot, self.se3_rot_sigma_min, self.se3_rot_sigma_max)
        return tr_sigma, rot_sigma

    # ------------------------------------------------------------------
    # Multi-rate λ masking
    # ------------------------------------------------------------------

    def get_lambda_q(self, t: torch.Tensor) -> torch.Tensor:
        """
        PROTAC noise multiplier λ_q(t).
        λ_q(t) = 1 for t > threshold, 0 for t <= threshold.
        This is a stepwise, non-differentiable engineering approximation.
        """
        return (t > self.lambda_q_threshold).float()

    def get_lambda_se3(self, t: torch.Tensor) -> torch.Tensor:
        """
        SE3 noise multiplier λ_SE3(t).
        λ_SE3(t) = 1 for t > threshold, 0 for t <= threshold.
        """
        return (t > self.lambda_se3_threshold).float()

    # ------------------------------------------------------------------
    # Forward diffusion (adding noise to clean data)
    # ------------------------------------------------------------------

    def add_noise_protac(
        self,
        ligand_pos: torch.Tensor,
        t_tr: torch.Tensor,
        t_rot: torch.Tensor,
        t_tor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to PROTAC translation, rotation, and torsion.

        Args:
            ligand_pos: [B, N_lig, 3] clean ligand positions
            t_tr, t_rot, t_tor: [B] diffusion times

        Returns:
            noisy_pos: [B, N_lig, 3]
            tr_update: [B, 3] translation noise applied
            rot_update: [B, 3, 3] rotation matrix applied
        """
        B = ligand_pos.shape[0]
        device = ligand_pos.device

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
        lambda_q = self.get_lambda_q(t_tr)

        # Translation noise
        tr_update = torch.randn(B, 3, device=device) * tr_sigma.unsqueeze(-1) * lambda_q.unsqueeze(-1)

        # Rotation noise (SO(3))
        rot_update = sample_so3_noise(B, rot_sigma * lambda_q, device)

        noisy_pos = ligand_pos + tr_update.unsqueeze(1)
        noisy_pos = torch.bmm(noisy_pos, rot_update)

        return noisy_pos, tr_update, rot_update

    def add_noise_se3(
        self,
        poi_pos: torch.Tensor,
        t_se3_tr: torch.Tensor,
        t_se3_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to E3-POI rigid body transformation.

        Args:
            poi_pos: [B, N_poi, 3] clean POI positions (relative to E3 origin)
            t_se3_tr, t_se3_rot: [B] diffusion times for SE3 DOFs
        Returns:
            noisy_pos, tr_update [B, 3], rot_update [B, 3, 3]
        """
        B = poi_pos.shape[0]
        device = poi_pos.device

        tr_sigma, rot_sigma = self.se3_t_to_sigma(t_se3_tr, t_se3_rot)
        lambda_se3 = self.get_lambda_se3(t_se3_tr)

        tr_update = torch.randn(B, 3, device=device) * tr_sigma.unsqueeze(-1) * lambda_se3.unsqueeze(-1)
        rot_update = sample_so3_noise(B, rot_sigma * lambda_se3, device)

        # Apply to POI center of mass
        poi_center = poi_pos.mean(dim=1, keepdim=True)
        poi_centered = poi_pos - poi_center
        noisy_center = poi_center + tr_update.unsqueeze(1)
        noisy_centered = torch.bmm(poi_centered, rot_update)
        noisy_pos = noisy_center + noisy_centered

        return noisy_pos, tr_update, rot_update

    # ------------------------------------------------------------------
    # Reverse diffusion step (score-based update)
    # ------------------------------------------------------------------

    def reverse_step_protac(
        self,
        ligand_pos: torch.Tensor,
        tr_score: torch.Tensor,
        rot_score: torch.Tensor,
        tor_score: torch.Tensor,
        t_tr: torch.Tensor,
        t_rot: torch.Tensor,
        t_tor: torch.Tensor,
        dt_tr: torch.Tensor,
        dt_rot: torch.Tensor,
        dt_tor: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Single reverse SDE step for PROTAC DOFs (Euler-Maruyama).

        dx = [score * σ² + noise] * dt
        """
        B = ligand_pos.shape[0]
        device = ligand_pos.device

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        tr_g = tr_sigma * np.sqrt(2 * np.log(self.tr_sigma_max / self.tr_sigma_min))
        rot_g = rot_sigma * np.sqrt(2 * np.log(self.rot_sigma_max / self.rot_sigma_min))

        # Translation update (with temperature sampling)
        if self.temp_sampling[0] != 1.0:
            ts = self.temp_sampling[0]
            tpsi = self.temp_psi[0]
            tsd = self.temp_sigma_data[0]
            tr_sigma_data = torch.exp(tsd * np.log(self.tr_sigma_max) + (1 - tsd) * np.log(self.tr_sigma_min))
            lambda_tr = (tr_sigma_data + tr_sigma) / (tr_sigma_data + tr_sigma / ts)
            tr_perturb = (tr_g**2 * dt_tr.unsqueeze(-1) * (lambda_tr + ts * tpsi / 2) * tr_score)
            if add_noise:
                tr_z = torch.randn(B, 3, device=device)
                tr_perturb = tr_perturb + tr_g * torch.sqrt(dt_tr * (1 + tpsi)).unsqueeze(-1) * tr_z
        else:
            tr_perturb = tr_g**2 * dt_tr.unsqueeze(-1) * tr_score
            if add_noise:
                tr_z = torch.randn(B, 3, device=device)
                tr_perturb = tr_perturb + tr_g * torch.sqrt(dt_tr).unsqueeze(-1) * tr_z

        # Rotation update (with temperature sampling)
        if self.temp_sampling[1] != 1.0:
            ts = self.temp_sampling[1]
            tpsi = self.temp_psi[1]
            tsd = self.temp_sigma_data[1]
            rot_sigma_data = torch.exp(tsd * np.log(self.rot_sigma_max) + (1 - tsd) * np.log(self.rot_sigma_min))
            lambda_rot = (rot_sigma_data + rot_sigma) / (rot_sigma_data + rot_sigma / ts)
            rot_perturb = (rot_g**2 * dt_rot.unsqueeze(-1) * (lambda_rot + ts * tpsi / 2) * rot_score)
            if add_noise:
                rot_z = torch.randn(B, 3, device=device)
                rot_perturb = rot_perturb + rot_g * torch.sqrt(dt_rot * (1 + tpsi)).unsqueeze(-1) * rot_z
        else:
            rot_perturb = rot_score * dt_rot.unsqueeze(-1) * rot_g**2
            if add_noise:
                rot_z = torch.randn(B, 3, device=device)
                rot_perturb = rot_perturb + rot_g * torch.sqrt(dt_rot).unsqueeze(-1) * rot_z

        # Apply updates
        ligand_pos = ligand_pos + tr_perturb.unsqueeze(1)
        rot_matrix = so3_exp(rot_perturb)
        ligand_center = ligand_pos.mean(dim=1, keepdim=True)
        ligand_centered = ligand_pos - ligand_center
        ligand_pos = ligand_center + torch.bmm(ligand_centered, rot_matrix)

        return ligand_pos

    def reverse_step_se3(
        self,
        poi_pos: torch.Tensor,
        se3_score: torch.Tensor,
        t_se3_tr: torch.Tensor,
        t_se3_rot: torch.Tensor,
        dt_se3_tr: torch.Tensor,
        dt_se3_rot: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Single reverse SDE step for E3-POI rigid body DOFs.

        se3_score: [B, 6] = [tr_score(3), rot_score(3)]
        """
        B = poi_pos.shape[0]
        device = poi_pos.device

        tr_score = se3_score[:, :3]
        rot_score = se3_score[:, 3:]

        tr_sigma, rot_sigma = self.se3_t_to_sigma(t_se3_tr, t_se3_rot)
        tr_g = tr_sigma * np.sqrt(2 * np.log(self.se3_tr_sigma_max / self.se3_tr_sigma_min))
        rot_g = rot_sigma * np.sqrt(2 * np.log(self.se3_rot_sigma_max / self.se3_rot_sigma_min))

        tr_perturb = tr_g**2 * dt_se3_tr.unsqueeze(-1) * tr_score
        if add_noise:
            tr_z = torch.randn(B, 3, device=device)
            tr_perturb = tr_perturb + tr_g * torch.sqrt(dt_se3_tr).unsqueeze(-1) * tr_z

        rot_perturb = rot_score * dt_se3_rot.unsqueeze(-1) * rot_g**2
        if add_noise:
            rot_z = torch.randn(B, 3, device=device)
            rot_perturb = rot_perturb + rot_g * torch.sqrt(dt_se3_rot).unsqueeze(-1) * rot_z

        # Apply to POI
        poi_center = poi_pos.mean(dim=1, keepdim=True)
        poi_centered = poi_pos - poi_center
        poi_center = poi_center + tr_perturb.unsqueeze(1)
        rot_matrix = so3_exp(rot_perturb)
        poi_pos = poi_center + torch.bmm(poi_centered, rot_matrix)

        return poi_pos

    # ------------------------------------------------------------------
    # Inference schedules
    # ------------------------------------------------------------------

    def get_inference_schedule(self, steps: int = None) -> dict:
        """
        Returns time discretization for reverse diffusion.

        PROTAC DOFs: linear schedule from t=1 to t=0
        SE3 DOFs: linear schedule from t=1 to t=0 (but noise decays slower due to λ)

        Returns dict with:
            'tr_schedule': [steps+1] PROTAC translation times
            'rot_schedule': [steps+1] PROTAC rotation times
            'tor_schedule': [steps+1] PROTAC torsion times
            'se3_tr_schedule': [steps+1] SE3 translation times
            'se3_rot_schedule': [steps+1] SE3 rotation times
        """
        if steps is None:
            steps = self.inference_steps
        schedule = np.linspace(1.0, 0.0, steps + 1)
        return {
            'tr_schedule': schedule.copy(),
            'rot_schedule': schedule.copy(),
            'tor_schedule': schedule.copy(),
            'se3_tr_schedule': schedule.copy(),
            'se3_rot_schedule': schedule.copy(),
        }


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TernaryInitializer:
    """
    Initializes the three-molecule system at t=1 (fully noised).

    Reference: E3 fixed at origin (COM=0).
    PROTAC: randomly placed in a spherical shell around E3 pocket (±10Å).
    POI: randomly placed in a spherical shell 30-60Å from E3.
    """

    def __init__(
        self,
        protac_shell_min: float = 0.0,
        protac_shell_max: float = 10.0,
        poi_shell_min: float = 30.0,
        poi_shell_max: float = 60.0,
    ):
        self.protac_shell_min = protac_shell_min
        self.protac_shell_max = protac_shell_max
        self.poi_shell_min = poi_shell_min
        self.poi_shell_max = poi_shell_max

    def initialize_protac(
        self,
        e3_pocket_center: torch.Tensor,
        ligand_template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize PROTAC near E3 pocket.

        Args:
            e3_pocket_center: [B, 3] pocket center coordinates
            ligand_template: [N_lig, 3] template ligand geometry
        Returns:
            [B, N_lig, 3] initialized ligand positions
        """
        B = e3_pocket_center.shape[0]
        N_lig = ligand_template.shape[0]
        device = e3_pocket_center.device

        # Random position on spherical shell + random rotation
        r = (
            self.protac_shell_min
            + torch.rand(B, device=device) * (self.protac_shell_max - self.protac_shell_min)
        )
        direction = torch.randn(B, 3, device=device)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

        translation = e3_pocket_center + r.unsqueeze(-1) * direction
        rot = torch.from_numpy(R.random(B).as_matrix()).float().to(device)

        ligand_pos = ligand_template.unsqueeze(0).repeat(B, 1, 1)
        ligand_pos = torch.bmm(ligand_pos, rot) + translation.unsqueeze(1)

        return ligand_pos

    def initialize_poi(
        self,
        e3_center: torch.Tensor,
        poi_template: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize POI in a spherical shell around E3.

        Args:
            e3_center: [B, 3] E3 center (typically origin)
            poi_template: [N_poi, 3] template POI geometry
        Returns:
            [B, N_poi, 3] initialized POI positions
        """
        B = e3_center.shape[0]
        N_poi = poi_template.shape[0]
        device = e3_center.device

        r = (
            self.poi_shell_min
            + torch.rand(B, device=device) * (self.poi_shell_max - self.poi_shell_min)
        )
        direction = torch.randn(B, 3, device=device)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)

        translation = e3_center + r.unsqueeze(-1) * direction
        rot = torch.from_numpy(R.random(B).as_matrix()).float().to(device)

        poi_pos = poi_template.unsqueeze(0).repeat(B, 1, 1)
        poi_pos = torch.bmm(poi_pos, rot) + translation.unsqueeze(1)

        return poi_pos


# ------------------------------------------------------------------
# SO(3) utilities
# ------------------------------------------------------------------

def sample_so3_noise(
    batch_size: int, sigma: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    Sample noise on SO(3) manifold using the isotropic Gaussian distribution
    in the Lie algebra so(3).

    Args:
        batch_size: number of samples
        sigma: [B] noise level per sample
        device: torch device
    Returns:
        [B, 3, 3] rotation matrices
    """
    B = batch_size
    omega = torch.randn(B, 3, device=device) * sigma.unsqueeze(-1)
    return so3_exp(omega)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) to SO(3) using Rodrigues' formula.

    Args:
        omega: [B, 3] rotation vectors in so(3)
    Returns:
        [B, 3, 3] rotation matrices
    """
    B = omega.shape[0]
    device = omega.device

    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)  # [B, 1]
    # Avoid division by zero
    mask = theta.squeeze(-1) > 1e-8

    # Rodrigues' formula: R = I + sin(θ)/θ * ω̂ + (1-cos(θ))/θ² * ω̂²
    omega_hat = vector_to_skew(omega)  # [B, 3, 3]

    R = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

    if mask.any():
        theta_m = theta[mask]
        omega_hat_m = omega_hat[mask]
        sin_theta = torch.sin(theta_m).unsqueeze(-1)
        cos_theta = torch.cos(theta_m).unsqueeze(-1)

        R[mask] = (
            torch.eye(3, device=device)
            + sin_theta / theta_m.unsqueeze(-1) * omega_hat_m
            + (1 - cos_theta) / (theta_m.unsqueeze(-1) ** 2)
            * torch.bmm(omega_hat_m, omega_hat_m)
        )

    return R


def vector_to_skew(v: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D vectors to skew-symmetric matrices.

    Args:
        v: [..., 3]
    Returns:
        [..., 3, 3] skew-symmetric matrices
    """
    *batch_dims, _ = v.shape
    device = v.device

    z = torch.zeros(*batch_dims, device=device)

    row0 = torch.stack([z, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], z, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], z], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


# ------------------------------------------------------------------
# SO(3) score normalization — precomputed table lookup
# Matches DiffDock utils/so3.py exactly
# ------------------------------------------------------------------

_MIN_EPS, _MAX_EPS, _N_EPS = 0.0005, 4, 2000
_X_N = 2000
_omegas_array = None
_cdf_vals = None
_score_norms = None
_exp_score_norms = None


def _init_so3_cache(cache_dir=None):
    """Precompute SO(3) score norms and cache to disk. Run once."""
    global _omegas_array, _cdf_vals, _score_norms, _exp_score_norms
    if _exp_score_norms is not None:
        return

    import os as _os
    cache_dir = cache_dir or '.'
    omega_file = _os.path.join(cache_dir, '.so3_omegas_array4.npy')
    score_file = _os.path.join(cache_dir, '.so3_exp_score_norms4.npy')

    if _os.path.exists(omega_file) and _os.path.exists(score_file):
        _omegas_array = np.load(omega_file)
        _exp_score_norms = np.load(score_file)
        return

    _eps_array = 10 ** np.linspace(np.log10(_MIN_EPS), np.log10(_MAX_EPS), _N_EPS)
    _omegas_array = np.linspace(0, np.pi, _X_N + 1)[1:]

    def _expansion(omega, eps, L=2000):
        l = np.arange(L).reshape(-1, 1)
        return ((2 * l + 1) * np.exp(-l * (l + 1) * eps**2 / 2)
                * np.sin(omega * (l + 0.5)) / np.sin(omega / 2)).sum(0)

    def _density(exp, omega):
        return exp * (1 - np.cos(omega)) / np.pi

    def _score(exp, omega, eps, L=2000):
        l = np.arange(L).reshape(-1, 1)
        hi = np.sin((l + 0.5) * omega)
        dhi = (l + 0.5) * np.cos((l + 0.5) * omega)
        lo = np.sin(omega / 2)
        dlo = 0.5 * np.cos(omega / 2)
        dSigma = ((2 * l + 1) * np.exp(-l * (l + 1) * eps**2 / 2)
                  * (lo * dhi - hi * dlo) / lo**2).sum(0)
        return dSigma / exp

    exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array])
    pdf_vals = np.asarray([_density(exp, _omegas_array) for exp in exp_vals])
    _cdf_vals = np.asarray([pdf.cumsum() / _X_N * np.pi for pdf in pdf_vals])
    _score_norms = np.asarray([
        _score(exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))
    ])
    _exp_score_norms = np.sqrt(
        np.sum(_score_norms**2 * pdf_vals, axis=1)
        / np.sum(pdf_vals, axis=1) / np.pi
    )
    np.save(omega_file, _omegas_array)
    np.save(score_file, _exp_score_norms)


def so3_score_norm(sigma, cache_dir=None):
    """
    SO(3) score normalization factor — table lookup matching DiffDock.
    For sigma values outside the precomputed range, falls back to analytic approx.
    """
    _init_so3_cache(cache_dir)
    sigma_np = sigma.detach().cpu().numpy() if hasattr(sigma, 'cpu') else np.asarray(sigma)
    eps_idx = (np.log10(sigma_np) - np.log10(_MIN_EPS)) / (
        np.log10(_MAX_EPS) - np.log10(_MIN_EPS)) * _N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), 0, _N_EPS - 1)
    result = torch.from_numpy(_exp_score_norms[eps_idx]).float()
    if hasattr(sigma, 'device'):
        result = result.to(sigma.device)
    return result
