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

from tricoopdiff.diffdock.utils.geometry import rigid_transform_Kabsch_3D_torch_batch
from tricoopdiff.pose_anchors import assemble_interface_pair, rotation_exp as so3_exp
from tricoopdiff.diffdock.utils import so3


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

        Torsion updates are handled by the native binary sampler (applied
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
        # and the native expert's IGSO(3) convention (see sample_so3_noise_wrapped).
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
        to the clean POI. The exact sampled transforms then map the second pose
        onto the first using the same composition as initialization. Consequently
        each expert observes exactly one native DiffDock marginal, while all
        coordinates remain in one consistent global frame.
        """
        noisy_ligand, tr_e3l, rot_e3l, omega_e3l = self.add_noise_protac(
            ligand_pos, t_e3l, t_e3l, t_e3l,
        )
        _, tr_pol, rot_pol, omega_pol = self.add_noise_protac(
            ligand_pos, t_pol, t_pol, t_pol,
        )
        center = ligand_pos.mean(dim=1)
        offset_e3l = center + tr_e3l - (rot_e3l @ center.unsqueeze(-1)).squeeze(-1)
        offset_pol = center + tr_pol - (rot_pol @ center.unsqueeze(-1)).squeeze(-1)
        noisy_poi, poi_to_global = assemble_interface_pair(
            poi_pos, rot_e3l, offset_e3l, rot_pol, offset_pol,
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
        the native DiffDock conformer update — so torsions do not leak a
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
        from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

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
        source_ligand_pos: torch.Tensor,
        target_ligand_pos: torch.Tensor,
        tr_score: torch.Tensor,
        rot_score: torch.Tensor,
        t_tr: torch.Tensor,
        t_rot: torch.Tensor,
        dt_tr: torch.Tensor,
        dt_rot: torch.Tensor,
        add_noise: bool = True,
        return_rotation: bool = False,
    ):
        """Compose the POI--Ligand proposal with the E3--Ligand proposal.

        The POI--Ligand expert predicts the rigid update that DiffDock would
        apply to ``source_ligand_pos`` with POI fixed.  Both interface experts
        must start from that same old Ligand pose.  This method constructs the
        POI-side proposal, aligns it to the independently constructed E3-side
        ``target_ligand_pos``, and applies the same rigid transform to POI.
        Thus the final Ligand is the E3-side proposal while the POI--Ligand
        relative pose exactly preserves the POI-side proposal.
        """
        empty_torsion = tr_score.new_zeros((tr_score.shape[0], 0))
        proposed_ligand, _ = self.reverse_step_protac(
            source_ligand_pos,
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
        proposal_to_target_rotation, proposal_to_target_translation = (
            rigid_transform_Kabsch_3D_torch_batch(
                proposed_ligand, target_ligand_pos,
            )
        )
        updated_poi = (
            torch.bmm(
                poi_pos, proposal_to_target_rotation.transpose(-1, -2),
            )
            + proposal_to_target_translation.transpose(-1, -2)
        )
        if return_rotation:
            return updated_poi, proposal_to_target_rotation
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
# SO(3) utilities
# ------------------------------------------------------------------

def sample_so3_noise_wrapped(
    batch_size: int, sigma: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample noise on SO(3) matching DiffDock's wrapped `so3.sample_vec` convention.

    Draws a uniform unit direction × a wrapped angle sampled from the IGSO(3)
    CDF in [0, π], returning the wrapped so(3) vector ω and its rotation matrix.
    This is the convention the frozen expert assumes.

    The wrapped IGSO(3) distribution is required by both interface experts;
    an unwrapped R³ Gaussian emits ‖ω‖ > π for σ > 1.0, outside the score table.

    Args:
        batch_size: number of samples
        sigma: [B] noise level per sample
        device: torch device
    Returns:
        (rotation_matrices [B, 3, 3], omega [B, 3]) with ‖omega‖ ∈ [0, π]
    """
    B = batch_size
    sigma_np = sigma.detach().cpu().numpy() if hasattr(sigma, 'cpu') else np.asarray(sigma)
    sigma_np = np.clip(np.atleast_1d(sigma_np).astype(float), so3.MIN_EPS, so3.MAX_EPS)
    eps_idx = (np.log10(sigma_np) - np.log10(so3.MIN_EPS)) / (
        np.log10(so3.MAX_EPS) - np.log10(so3.MIN_EPS)) * so3.N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), 0, so3.N_EPS - 1)

    # Inverse-CDF sampling of the wrapped angle, mirroring so3.sample(): draw
    # u ~ U[0,1) and linearly interpolate u against the CDF row → angle ∈ [0, π].
    u = torch.rand(B, device=device).cpu().numpy()
    angles_np = np.array([
        np.interp(u[i], so3._cdf_vals[eps_idx[i]], so3._omegas_array) for i in range(B)
    ])
    angles = torch.from_numpy(angles_np).float().to(device)

    direction = torch.randn(B, 3, device=device)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-8)
    omega = direction * angles.unsqueeze(-1)  # [B, 3], ‖ω‖ ∈ [0, π]
    return so3_exp(omega), omega
