"""
Diffusion process for TriCoopDiff.

The E3--Ligand and POI--Ligand relative poses both use the native DiffDock
translation, rotation and torsion schedules. Each interface receives an
independent noise draw at the same diffusion time, and one globally consistent
ternary pose is reconstructed from those two relative states.

Reference frame: E3 is fixed. The first expert moves Ligand relative to E3; the
second expert moves POI by the exact inverse of its proposed Ligand update.

The diffusion parameter t is a noise schedule variable, NOT physical time.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

from .diffdock.utils.geometry import rigid_transform_Kabsch_3D_torch_batch


class InterfaceDiffusion(nn.Module):
    """
    Native DiffDock diffusion applied to two relative interface poses.

    Diffusion variables:
    - PROTAC translation t_q: 3D (relative to E3)
    - PROTAC rotation R_q: 3D in so(3)
    - PROTAC torsion angles χ: K (K = number of rotatable bonds)
    - POI-Ligand relative translation/rotation: second independent rigid pose
    """

    def __init__(
        self,
        # PROTAC noise parameters (from checkpoints/score_model/model_parameters.yml:
        # tr 0.1→19, rot 0.03→1.55, tor 0.0314→3.14). The tor bounds MUST match the
        # yml — the expert's baked-in score normalization (expert.py, via
        # model.t_to_sigma) uses 0.0314/3.14, so a mismatched 0.03/1.55 here makes
        # the reverse-step tor_g ~4.8× too weak and decouples forward/reverse σ.
        tr_sigma_min: float = 0.1,
        tr_sigma_max: float = 19.0,
        rot_sigma_min: float = 0.03,
        rot_sigma_max: float = 1.55,
        tor_sigma_min: float = 0.0314,
        tor_sigma_max: float = 3.14,
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

    # ------------------------------------------------------------------
    # Forward diffusion (adding noise to clean data)
    # ------------------------------------------------------------------

    def add_noise_protac(
        self,
        ligand_pos: torch.Tensor,
        t_tr: torch.Tensor,
        t_rot: torch.Tensor,
        t_tor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to PROTAC translation and rotation (rigid-body DOFs).

        The PROTAC forward process is FULL-RANGE: noise is sampled at the raw
        σ(t) level for all t ∈ [0, 1], with NO λ_q cutoff. This keeps the forward
        process consistent with ``reverse_step_protac`` (which denoises at the full
        σ schedule, never applying λ) and with the DSM training target. The previous
        λ_q cutoff made the forward schedule asymmetric with the reverse step.

        Torsion angles are perturbed separately by ``add_torsion_noise`` (applied
        after this rigid transform), mirroring DiffDock's ``modify_conformer``.

        Args:
            ligand_pos: [B, N_lig, 3] clean ligand positions
            t_tr, t_rot, t_tor: [B] diffusion times

        Returns:
            noisy_pos: [B, N_lig, 3]
            tr_update: [B, 3] translation noise applied
            rot_update: [B, 3, 3] rotation matrix applied
            rot_omega: [B, 3] so(3) rotation vector (pre-exp) for score target
        """
        B = ligand_pos.shape[0]
        device = ligand_pos.device

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)

        # Translation noise
        tr_update = torch.randn(B, 3, device=device) * tr_sigma.unsqueeze(-1)

        # Rotation noise (SO(3)) — wrapped sampling, matching the frozen expert's
        # and so3_score_vec's IGSO(3) convention (see sample_so3_noise_wrapped).
        rot_update, rot_omega = sample_so3_noise_wrapped(B, rot_sigma, device)

        # Rotate about the ligand centre, then translate in the world frame —
        # mirrors DiffDock's modify_conformer: (pos - centre) @ Rᵀ + tr + centre.
        # The transpose is REQUIRED: DiffDock applies the rotation as @ Rᵀ =
        # @ exp(-[ω]×), while the frozen expert was trained on that convention. A
        # bare @ R = @ exp([ω]×) makes the expert read the pose as rotation -ω and
        # emit score_vec(σ,-ω) = -score_vec(σ,ω) — the sign opposite of the training
        # target score_vec(σ,ω) — so reverse_step_protac amplifies |ω| (zero-FiLM
        # PROTAC-rotation divergence, the documented DockQ collapse).
        # (The previous (pos + tr) @ R rotated about the ORIGIN and rotated the
        # translation vector too, so the wrapped sampler's near-π angles teleported
        # the ligand away from its pocket — exploding the tr-score target mismatch.)
        lig_center = ligand_pos.mean(dim=1, keepdim=True)
        noisy_pos = torch.bmm(ligand_pos - lig_center, rot_update.transpose(-1, -2)) + tr_update.unsqueeze(1) + lig_center

        return noisy_pos, tr_update, rot_update, rot_omega

    def add_noise_interface_pair(
        self,
        ligand_pos: torch.Tensor,
        poi_pos: torch.Tensor,
        t_e3l: torch.Tensor,
        t_pol: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        """Noise two relative interface poses and reconstruct one ternary pose.

        ``E3--Ligand`` and ``POI--Ligand`` are independent DiffDock forward
        variables.  We first noise Ligand once relative to E3 and once relative
        to the clean POI.  A rigid alignment then maps the second noisy Ligand
        pose onto the first and applies the same transform to POI.  Consequently
        each expert observes exactly one native DiffDock marginal, while all
        coordinates remain in one consistent global frame.
        """
        noisy_ligand, tr_e3l, rot_e3l, omega_e3l = self.add_noise_protac(
            ligand_pos, t_e3l, t_e3l, t_e3l,
        )
        noisy_ligand_pol, tr_pol, rot_pol, omega_pol = self.add_noise_protac(
            ligand_pos, t_pol, t_pol, t_pol,
        )
        poi_to_global, poi_translation = rigid_transform_Kabsch_3D_torch_batch(
            noisy_ligand_pol, noisy_ligand,
        )
        noisy_poi = (
            torch.bmm(poi_pos, poi_to_global.transpose(-1, -2))
            + poi_translation.transpose(-1, -2)
        )
        noise = {
            "e3_ligand": {
                "tr": tr_e3l,
                "rot_matrix": rot_e3l,
                "rot_omega": omega_e3l,
            },
            "poi_ligand": {
                "tr_local": tr_pol,
                "rot_matrix_local": rot_pol,
                "rot_omega_local": omega_pol,
            },
        }
        return noisy_ligand, noisy_poi, noise, poi_to_global

    def add_torsion_noise(
        self,
        ligand_pos: torch.Tensor,
        t_tor: torch.Tensor,
        lig_edge_index: torch.Tensor,
        lig_edge_mask: torch.Tensor,
        lig_mask_rotate,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to PROTAC rotatable-bond torsion angles.

        Full-range (no λ_q cutoff), matching ``reverse_step_torsion`` and the
        torsion DSM target over all t ∈ [0, 1] — the previous λ_q gate made the
        forward schedule asymmetric with the reverse step.

        Translation/rotation perturb the whole conformer; torsion rotates each
        rotatable bond's downstream atoms around the bond axis. This mirrors
        DiffDock's ``modify_conformer``: the torsion-perturbed conformer is
        Kabsch-aligned back to its pre-torsion pose so the internal-coordinate
        (torsion) perturbation does not leak a global drift into the tr/rot
        score-matching targets. Applied AFTER ``add_noise_protac``.

        Args:
            ligand_pos: [B, N_lig, 3] conformer (already tr+rot noised)
            t_tor: [B] torsion diffusion times
            lig_edge_index: [2, E] ligand internal bond edges
            lig_edge_mask: [E] bool — rotatable bond mask
            lig_mask_rotate: [n_rot, N_lig] bool, or list of per-complex masks

        Returns:
            noisy_pos: [B, N_lig, 3] conformer with torsion angles perturbed
            tor_update: [B, n_rot] torsion angle noise (used as the score target)
        """
        from .diffdock.utils.torsion import modify_conformer_torsion_angles

        B = ligand_pos.shape[0]
        device = ligand_pos.device

        _, _, tor_sigma = self.t_to_sigma(t_tor, t_tor, t_tor)

        # Rotatable bond edges (one direction each), same row order as mask_rotate.
        rot_edges = lig_edge_index.T[lig_edge_mask]  # [n_rot, 2]
        n_rot = rot_edges.shape[0]

        tor_update = torch.randn(B, n_rot, device=device) * tor_sigma.unsqueeze(-1)

        # A batch of 1 may carry mask_rotate as a bare ndarray; a batch of N as a
        # list of per-complex masks. Normalize to a per-element list.
        if not isinstance(lig_mask_rotate, (list, tuple)):
            lig_mask_rotate = [lig_mask_rotate] * B

        noisy_pos = ligand_pos.clone()
        for b in range(B):
            rigid_pos = noisy_pos[b].unsqueeze(0)  # [1, N_lig, 3] pre-torsion pose
            mr = lig_mask_rotate[b]
            if isinstance(mr, torch.Tensor):
                mr = mr.detach().cpu().numpy()
            tu = tor_update[b].detach().cpu().numpy()
            flexible_pos = modify_conformer_torsion_angles(
                rigid_pos[0].detach().cpu(), rot_edges, mr, tu, as_numpy=False
            ).to(device).unsqueeze(0)  # [1, N_lig, 3] torsion-perturbed pose

            # Kabsch-align the torsion-perturbed conformer back to its pre-torsion
            # pose (DiffDock's `modify_conformer` does exactly this): rotating
            # downstream side-groups moves the global pose, which would otherwise
            # contaminate the tr/rot score-matching targets.
            R_align, t_align = rigid_transform_Kabsch_3D_torch_batch(flexible_pos, rigid_pos)
            noisy_pos[b] = flexible_pos[0] @ R_align[0].T + t_align[0].T

        return noisy_pos, tor_update

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single reverse SDE step for PROTAC rigid-body DOFs (Euler-Maruyama).

        Mirrors DiffDock's ``modify_conformer_batch(...)`` rigid stage exactly:
        compute tr/rot perturbations, apply the rotation about the ligand COM,
        and return both the updated coordinates and the rigid-frame target that
        the torsion step should Kabsch-align back to.
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
            tr_sigma_data = torch.exp(torch.tensor(
                tsd * np.log(self.tr_sigma_max) + (1 - tsd) * np.log(self.tr_sigma_min),
                device=device, dtype=tr_sigma.dtype,
            ))
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
            rot_sigma_data = torch.exp(torch.tensor(
                tsd * np.log(self.rot_sigma_max) + (1 - tsd) * np.log(self.rot_sigma_min),
                device=device, dtype=rot_sigma.dtype,
            ))
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

        # Apply rigid-body updates in the same order as native DiffDock:
        # rotate about the ligand COM, then translate in world coordinates.
        # The returned rigid pose is the pre-torsion reference that the native
        # torsion branch Kabsch-aligns back to. ``tor_score`` is intentionally
        # unused here because native ``modify_conformer_batch(...)`` applies
        # torsion in a second stage after the rigid update.
        ligand_center = ligand_pos.mean(dim=1, keepdim=True)
        rot_matrix = so3_exp(rot_perturb)
        rigid_pos = torch.bmm(ligand_pos - ligand_center, rot_matrix.transpose(-1, -2)) + tr_perturb.unsqueeze(1) + ligand_center

        return rigid_pos, rigid_pos.clone()

    def reverse_step_torsion(
        self,
        ligand_pos: torch.Tensor,
        tor_score: torch.Tensor,
        t_tor: torch.Tensor,
        dt_tor: torch.Tensor,
        lig_edge_index: torch.Tensor,
        lig_edge_mask: torch.Tensor,
        lig_mask_rotate,
        add_noise: bool = True,
        rigid_target_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Single reverse SDE step for PROTAC torsion DOFs (Euler-Maruyama).

        Complements ``reverse_step_protac`` (which handles tr/rot but ignores
        ``tor_score``): updates rotatable-bond angles via the torsion score, then
        Kabsch-aligns the result back to the pre-torsion pose — mirroring
        ``add_torsion_noise`` — so the internal-coordinate update does not leak a
        global drift into the already-denoised tr/rot DOFs.

        Args:
            ligand_pos: [B, N_lig, 3] current rigid-body conformer (after tr/rot reverse)
            tor_score: [B, n_rot] torsion score
            t_tor: [B] torsion diffusion time
            dt_tor: [B] reverse step size (t_i - t_{i+1} > 0)
            lig_edge_index: [2, E] ligand internal bond edges
            lig_edge_mask: [E] bool rotatable-bond mask
            lig_mask_rotate: [n_rot, N_lig] bool, or per-complex list
            add_noise: add stochastic noise (False on the final step)
            rigid_target_pos: [B, N_lig, 3] optional pre-torsion rigid pose to
                align back to. When None, defaults to ``ligand_pos``.

        Returns:
            [B, N_lig, 3] conformer with torsion angles updated
        """
        from .diffdock.utils.torsion import modify_conformer_torsion_angles

        B = ligand_pos.shape[0]
        device = ligand_pos.device

        if rigid_target_pos is None:
            rigid_target_pos = ligand_pos

        _, _, tor_sigma = self.t_to_sigma(t_tor, t_tor, t_tor)
        tor_g = tor_sigma * np.sqrt(2 * np.log(self.tor_sigma_max / self.tor_sigma_min))

        # Torsion update (with temperature sampling). Mirrors the tr/rot branches
        # in reverse_step_protac and DiffDock's sampling.py:202-205. Without this,
        # temp_sampling_tor=7.04 was defined but never read, collapsing the torsion
        # drift to ~1/25 of native and letting noise dominate the denoising.
        if self.temp_sampling[2] != 1.0:
            ts = self.temp_sampling[2]
            tpsi = self.temp_psi[2]
            tsd = self.temp_sigma_data[2]
            tor_sigma_data = torch.exp(torch.tensor(
                tsd * np.log(self.tor_sigma_max) + (1 - tsd) * np.log(self.tor_sigma_min),
                device=device, dtype=tor_sigma.dtype,
            ))
            lambda_tor = (tor_sigma_data + tor_sigma) / (tor_sigma_data + tor_sigma / ts)
            tor_perturb = (tor_g**2 * dt_tor.unsqueeze(-1) * (lambda_tor + ts * tpsi / 2) * tor_score)
            if add_noise:
                tor_z = torch.randn(B, tor_score.shape[-1], device=device)
                tor_perturb = tor_perturb + tor_g * torch.sqrt(dt_tor * (1 + tpsi)).unsqueeze(-1) * tor_z
        else:
            tor_perturb = tor_g**2 * dt_tor.unsqueeze(-1) * tor_score
            if add_noise:
                tor_z = torch.randn(B, tor_score.shape[-1], device=device)
                tor_perturb = tor_perturb + tor_g * torch.sqrt(dt_tor).unsqueeze(-1) * tor_z

        rot_edges = lig_edge_index.T[lig_edge_mask]  # [n_rot, 2]
        if not isinstance(lig_mask_rotate, (list, tuple)):
            lig_mask_rotate = [lig_mask_rotate] * B

        out_pos = ligand_pos.clone()
        for b in range(B):
            rigid_pos = rigid_target_pos[b].unsqueeze(0)  # [1, N_lig, 3] native rigid reference
            torsion_input = ligand_pos[b].unsqueeze(0)    # [1, N_lig, 3] pose to deform torsionally
            mr = lig_mask_rotate[b]
            if isinstance(mr, torch.Tensor):
                mr = mr.detach().cpu().numpy()
            tp = tor_perturb[b].detach().cpu().numpy()
            flexible_pos = modify_conformer_torsion_angles(
                torsion_input[0].detach().cpu(), rot_edges, mr, tp, as_numpy=False
            ).to(device).unsqueeze(0)  # [1, N_lig, 3] torsion-perturbed pose

            # Kabsch-align back to the native rigid reference so the torsion update
            # stays an internal-coordinate change (no global drift into tr/rot).
            R_align, t_align = rigid_transform_Kabsch_3D_torch_batch(flexible_pos, rigid_pos)
            out_pos[b] = flexible_pos[0] @ R_align[0].T + t_align[0].T

        return out_pos

    def reverse_step_poi_ligand_interface(
        self,
        poi_pos: torch.Tensor,
        ligand_pos: torch.Tensor,
        tr_score: torch.Tensor,
        rot_score: torch.Tensor,
        t_tr: torch.Tensor,
        t_rot: torch.Tensor,
        dt_tr: torch.Tensor,
        dt_rot: torch.Tensor,
        add_noise: bool = True,
        return_rotation: bool = False,
    ):
        """Denoise POI--Ligand pose while keeping the Ligand pose fixed.

        The POI--Ligand expert predicts the rigid update that DiffDock would
        apply to Ligand with POI fixed.  This method constructs that proposal
        with the native DiffDock reverse step, then applies its exact inverse
        rigid transform to POI.  The resulting relative pose is identical, but
        the E3--Ligand pose established by the first expert is not disturbed.
        """
        empty_torsion = tr_score.new_zeros((tr_score.shape[0], 0))
        proposed_ligand, _ = self.reverse_step_protac(
            ligand_pos,
            tr_score,
            rot_score,
            empty_torsion,
            t_tr,
            t_rot,
            t_tr,
            dt_tr,
            dt_rot,
            dt_tr,
            add_noise=add_noise,
        )
        inverse_rotation, inverse_translation = (
            rigid_transform_Kabsch_3D_torch_batch(proposed_ligand, ligand_pos)
        )
        updated_poi = (
            torch.bmm(poi_pos, inverse_rotation.transpose(-1, -2))
            + inverse_translation.transpose(-1, -2)
        )
        if return_rotation:
            return updated_poi, inverse_rotation
        return updated_poi

    # ------------------------------------------------------------------
    # Inference schedules
    # ------------------------------------------------------------------

    def get_inference_schedule(self, steps: int = None, t_start: float = 1.0) -> dict:
        """
        Returns time discretization for reverse diffusion.

        Both interface processes use the same linear clock from ``t_start`` to
        zero; their stochastic noise draws remain independent.

        Returns dict with:
            'tr_schedule': [steps+1] PROTAC translation times
            'rot_schedule': [steps+1] PROTAC rotation times
            'tor_schedule': [steps+1] PROTAC torsion times
        """
        if steps is None:
            steps = self.inference_steps
        if not 0.0 < t_start <= 1.0:
            raise ValueError(f"t_start must be in (0, 1], got {t_start}")
        schedule = np.linspace(float(t_start), 0.0, steps + 1)
        return {
            'tr_schedule': schedule.copy(),
            'rot_schedule': schedule.copy(),
            'tor_schedule': schedule.copy(),
        }


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TernaryInitializer:
    """
    Initializes the three-molecule system at t=1 (fully noised).

    Reference: E3 fixed at origin (COM=0).
    ``native`` reproduces DiffDock's wide PROTAC prior and places POI on a
    30-60 Å shell. ``coupled5a`` is the validated numerical fallback. The
    ``anchor_local`` mode preserves the observed E3-POI separation and places
    PROTAC near a predicted interface anchor with bounded local perturbations.
    """

    def __init__(
        self,
        protac_shell_min: float = 0.0,
        protac_shell_max: float = 10.0,
        poi_shell_min: float = 30.0,
        poi_shell_max: float = 60.0,
        coupled_radius: float = 5.0,
    ):
        self.protac_shell_min = protac_shell_min
        self.protac_shell_max = protac_shell_max
        self.poi_shell_min = poi_shell_min
        self.poi_shell_max = poi_shell_max
        self.coupled_radius = coupled_radius

    def initialize_protac(
        self,
        e3_center: torch.Tensor,
        ligand_template: torch.Tensor,
        lig_edge_index=None,
        lig_edge_mask=None,
        lig_mask_rotate=None,
        tr_sigma_max: float = None,
        init_mode: str = "native",
        anchor_center: torch.Tensor = None,
        local_radius: float = 5.0,
    ) -> torch.Tensor:
        """
        Initialize the PROTAC at t=1 (fully noised).

        Two modes:

        * **Legacy shell** (``tr_sigma_max is None``): random point on a
          spherical shell ``[protac_shell_min, protac_shell_max]`` around
          ``e3_center`` + a random rotation about the origin, keeping the
          template torsion angles. Kept for backward compatibility with the old
          call sites; NOT what native DiffDock does.

        * **Native** (``tr_sigma_max`` given, matching DiffDock's
          ``randomize_position`` — the 0.844 E3-L baseline path): (1) uniform
          torsion angles in [-π, π] on all rotatable bonds (``no_torsion: false``),
          (2) rotate the conformer about its own COM, (3) translate its COM onto
          the receptor COM (``receptor.pos.mean(0)`` = ``e3_center``), (4) add a
          Gaussian translation N(0, tr_sigma_max²). This is what makes the
          zero-FiLM baseline seed-comparable to native DiffDock.

        Args:
            e3_center: [B, 3] receptor (E3) centroid — native ``receptor.pos.mean(0)``.
            ligand_template: [N_lig, 3] template ligand geometry.
            lig_edge_index: [2, E] ligand internal bond edges (native init only).
            lig_edge_mask: [E] bool rotatable-bond mask (native init only).
            lig_mask_rotate: [n_rot, N_lig] bool, or list of per-complex masks.
            tr_sigma_max: translation std for the native Gaussian step (DiffDock's
                ``initial_noise_std_proportion=-1.0`` -> std = tr_sigma_max = 19.0).
                When None, use the legacy spherical-shell init.

        Returns:
            [B, N_lig, 3] initialized ligand positions.
        """
        if init_mode not in {"native", "coupled5a", "anchor_local"}:
            raise ValueError(f"unknown initialization mode: {init_mode}")
        B = e3_center.shape[0]
        N_lig = ligand_template.shape[0]
        device = e3_center.device

        # --- Legacy spherical-shell init -------------------------------------
        if tr_sigma_max is None:
            r = (
                self.protac_shell_min
                + torch.rand(B, device=device) * (self.protac_shell_max - self.protac_shell_min)
            )
            direction = torch.randn(B, 3, device=device)
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            translation = e3_center + r.unsqueeze(-1) * direction
            rot = torch.from_numpy(R.random(B).as_matrix()).float().to(device)
            ligand_pos = ligand_template.unsqueeze(0).repeat(B, 1, 1)
            ligand_pos = torch.bmm(ligand_pos, rot) + translation.unsqueeze(1)
            return ligand_pos

        # --- Native DiffDock init (randomize_position) -----------------------
        from .diffdock.utils.torsion import modify_conformer_torsion_angles

        ligand_pos = ligand_template.unsqueeze(0).repeat(B, 1, 1)  # [B, N_lig, 3]

        # (1) uniform-random torsion angles in [-pi, pi] (no_torsion: false).
        if lig_edge_mask is not None and bool(lig_edge_mask.any()):
            rot_edges = lig_edge_index.T[lig_edge_mask]  # [n_rot, 2]
            n_rot = rot_edges.shape[0]
            if not isinstance(lig_mask_rotate, (list, tuple)):
                lig_mask_rotate = [lig_mask_rotate] * B
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=(B, n_rot))
            for b in range(B):
                mr = lig_mask_rotate[b]
                if isinstance(mr, torch.Tensor):
                    mr = mr.detach().cpu().numpy()
                ligand_pos[b] = modify_conformer_torsion_angles(
                    ligand_pos[b].detach().cpu(), rot_edges, mr,
                    torsion_updates[b], as_numpy=False,
                ).to(device)

        # (2) rotate about the ligand's own COM, then translate onto the selected
        # anchor. ``anchor_local`` uses a small-angle rotation and a bounded
        # displacement; ``native``/``coupled5a`` retain their historical priors.
        # Match vendored DiffDock randomize_position exactly: it draws a single
        # rotation matrix, applies `@ random_rotation.T`, and then centers onto the
        # receptor COM before the Gaussian translation. Keeping the transpose here
        # is important for seed-level parity with the native initialization path.
        ligand_center = ligand_pos.mean(dim=1, keepdim=True)  # [B, 1, 3]
        if init_mode == "anchor_local":
            rot = so3_exp(
                torch.randn(B, 3, device=device, dtype=ligand_pos.dtype) * 0.25
            )
        else:
            rot = torch.from_numpy(R.random(B).as_matrix()).float().to(device)  # [B, 3, 3]
        ligand_pos = torch.bmm(ligand_pos - ligand_center, rot.transpose(-1, -2)) + e3_center.unsqueeze(1)

        # (3) Gaussian translation N(0, tr_sigma_max^2) (initial_noise_std_proportion=-1.0).
        if init_mode in {"coupled5a", "anchor_local"}:
            radius = torch.rand(B, device=device) * (self.coupled_radius if init_mode == "coupled5a" else local_radius)
            direction = torch.randn(B, 3, device=device)
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            tr_update = direction * radius.unsqueeze(-1)
            if init_mode == "anchor_local":
                if anchor_center is None:
                    raise ValueError("anchor_local requires anchor_center")
                ligand_pos = ligand_pos - e3_center.unsqueeze(1) + anchor_center.unsqueeze(1)
        else:
            tr_update = torch.randn(B, 3, device=device) * tr_sigma_max
        ligand_pos = ligand_pos + tr_update.unsqueeze(1)

        return ligand_pos

    def initialize_poi(
        self,
        e3_center: torch.Tensor,
        poi_template: torch.Tensor,
        return_rotation: bool = False,
        init_mode: str = "native",
        local_radius: float = 5.0,
    ):
        """
        Initialize POI using the selected prior. ``anchor_local`` preserves its
        current E3-frame center and only applies a bounded local perturbation.

        Args:
            e3_center: [B, 3] E3 center (typically origin)
            poi_template: [N_poi, 3] template POI geometry
        Returns:
            [B, N_poi, 3] initialized POI positions. If ``return_rotation`` is
            True, returns ``(poi_pos, rot)`` where ``rot`` [B,3,3] maps current
            E3-frame row vectors back into the POI template/body frame.
        """
        if init_mode not in {"native", "coupled5a", "anchor_local"}:
            raise ValueError(f"unknown initialization mode: {init_mode}")
        B = e3_center.shape[0]
        N_poi = poi_template.shape[0]
        device = e3_center.device

        if init_mode == "coupled5a":
            r = torch.rand(B, device=device) * self.coupled_radius
        elif init_mode == "anchor_local":
            r = torch.rand(B, device=device) * local_radius
        else:
            r = self.poi_shell_min + torch.rand(B, device=device) * (self.poi_shell_max - self.poi_shell_min)
        direction = torch.randn(B, 3, device=device)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1e-8)

        poi_pos = poi_template.unsqueeze(0).repeat(B, 1, 1)
        poi_center = poi_pos.mean(dim=1, keepdim=True)
        if init_mode == "anchor_local":
            rot = so3_exp(
                torch.randn(B, 3, device=device, dtype=poi_pos.dtype) * 0.25
            )
            translation = poi_center.squeeze(1) + r.unsqueeze(-1) * direction
        else:
            translation = e3_center + r.unsqueeze(-1) * direction
            rot = torch.from_numpy(R.random(B).as_matrix()).float().to(device)
        poi_pos = torch.bmm(poi_pos - poi_center, rot.transpose(-1, -2)) + translation.unsqueeze(1)

        if return_rotation:
            return poi_pos, rot
        return poi_pos


# ------------------------------------------------------------------
# SO(3) utilities
# ------------------------------------------------------------------

def sample_so3_noise_wrapped(
    batch_size: int, sigma: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample noise on SO(3) matching DiffDock's wrapped `so3.sample_vec` convention.

    Draws a uniform unit direction × a wrapped angle sampled from the IGSO(3)
    CDF in [0, π], returning the wrapped so(3) vector ω and its rotation matrix.
    This is the convention the frozen expert and `so3_score_vec` assume.

    The wrapped IGSO(3) distribution is required by both interface experts;
    an unwrapped R³ Gaussian emits ‖ω‖ > π for σ > 1.0, outside the score table.

    Args:
        batch_size: number of samples
        sigma: [B] noise level per sample
        device: torch device
    Returns:
        (rotation_matrices [B, 3, 3], omega [B, 3]) with ‖omega‖ ∈ [0, π]
    """
    _init_so3_cache()
    B = batch_size
    sigma_np = sigma.detach().cpu().numpy() if hasattr(sigma, 'cpu') else np.asarray(sigma)
    sigma_np = np.clip(np.atleast_1d(sigma_np).astype(float), _MIN_EPS, _MAX_EPS)
    eps_idx = (np.log10(sigma_np) - np.log10(_MIN_EPS)) / (
        np.log10(_MAX_EPS) - np.log10(_MIN_EPS)) * _N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), 0, _N_EPS - 1)

    # Inverse-CDF sampling of the wrapped angle, mirroring so3.sample(): draw
    # u ~ U[0,1) and linearly interpolate u against the CDF row → angle ∈ [0, π].
    u = torch.rand(B, device=device).cpu().numpy()
    angles_np = np.array([
        np.interp(u[i], _cdf_vals[eps_idx[i]], _omegas_array) for i in range(B)
    ])
    angles = torch.from_numpy(angles_np).float().to(device)

    direction = torch.randn(B, 3, device=device)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
    omega = direction * angles.unsqueeze(-1)  # [B, 3], ‖ω‖ ∈ [0, π]
    return so3_exp(omega), omega


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
    from pathlib import Path as _Path
    cache_dir = cache_dir or _os.environ.get(
        "TRICOOPDIFF_CACHE_DIR",
        str(_Path(__file__).resolve().parents[1]
            / "results/runtime_cache/diffdock"),
    )
    _os.makedirs(cache_dir, exist_ok=True)
    omega_file = _os.path.join(cache_dir, '.so3_omegas_array4.npy')
    score_file = _os.path.join(cache_dir, '.so3_exp_score_norms4.npy')
    score_norm_file = _os.path.join(cache_dir, '.so3_score_norms4.npy')

    if _os.path.exists(omega_file) and _os.path.exists(score_file):
        _omegas_array = np.load(omega_file)
        _exp_score_norms = np.load(score_file)
        if _os.path.exists(score_norm_file):
            _score_norms = np.load(score_norm_file)
        cdf_file = _os.path.join(cache_dir, '.so3_cdf_vals4.npy')
        if _os.path.exists(cdf_file):
            _cdf_vals = np.load(cdf_file)
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
    np.save(score_norm_file, _score_norms)
    np.save(_os.path.join(cache_dir, '.so3_cdf_vals4.npy'), _cdf_vals)


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


def so3_score_vec(sigma, vec, cache_dir=None):
    """
    SO(3) score vector — matches DiffDock utils/so3.py `score_vec(eps, vec)`.

    Given a sampled rotation vector `omega` and the noise level `sigma` that
    produced it, returns the score of the isotropic-Gaussian-on-SO(3) density,
    used as the rotation score-matching target.

    Args:
        sigma: [B] noise level (or scalar)
        vec: [B, 3] rotation vector omega
    Returns:
        [B, 3] score vector
    """
    _init_so3_cache(cache_dir)
    sigma_np = sigma.detach().cpu().numpy() if hasattr(sigma, 'cpu') else np.asarray(sigma)
    vec_np = vec.detach().cpu().numpy() if hasattr(vec, 'cpu') else np.asarray(vec)
    sigma_np = np.atleast_1d(sigma_np).astype(float)
    vec_np = np.atleast_2d(vec_np).astype(float)

    eps_idx = (np.log10(sigma_np) - np.log10(_MIN_EPS)) / (
        np.log10(_MAX_EPS) - np.log10(_MIN_EPS)) * _N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), 0, _N_EPS - 1)

    om = np.linalg.norm(vec_np, axis=-1)  # [B]
    om_safe = np.clip(om, 1e-8, None)
    score_norm = np.array([
        np.interp(om[i], _omegas_array, _score_norms[eps_idx[i]])
        for i in range(len(om))
    ])
    out = score_norm[:, None] * vec_np / om_safe[:, None]

    result = torch.from_numpy(out).float()
    if hasattr(vec, 'device'):
        result = result.to(vec.device)
    # Match input rank: scalar sigma + [3] vec -> [3]
    if not hasattr(sigma, 'shape') or len(sigma.shape) == 0:
        result = result[0]
    return result
