"""
Training pipeline for TriCoopDiff.

Multi-task loss: score matching (L_score) + physical consistency (L_phys) + SE3 (L_se3).

Gradient flow:
  L_score → score heads (both experts, but experts are frozen → only via FiLM)
  L_phys  → x_denoised → s_tri_residual → TriBrain (direct path, not through FiLM)
  L_se3   → TriBrain SE3 head

Two training phases:
  Phase 1: Train FiLM heads + SE3 head only (freeze judge/state heads)
  Phase 2: Unfreeze all TriBrain heads

Training strategy:
  - General ternary pretraining: 50-100 epochs, batch=32, lr=1e-4
  - PROTAC fine-tuning: 30-50 epochs, batch=8, lr=2e-5, LoRA (rank=4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import TriCoopDiff
from .diffusion import MultiRateDiffusion


class TriCoopDiffTrainer:
    """
    Trainer for TriCoopDiff model.

    Handles the multi-task loss, gradient isolation (frozen experts),
    and two-phase training strategy.
    """

    def __init__(
        self,
        model: TriCoopDiff,
        diffusion: MultiRateDiffusion,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device

        # Loss weights (from execution plan: score=1.0, phys=0.5, se3=0.3)
        self.loss_weights = loss_weights or {
            'score': 1.0,
            'phys': 0.5,
            'se3': 0.3,
        }

        # Only TriBrain is trainable; experts are frozen
        self.optimizer = AdamW(
            model.tribrain.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)

        self.global_step = 0

    def compute_loss(
        self,
        e3_data: dict,
        poi_data: dict,
        ligand_data: dict,
        t_se3_data: dict,
    ) -> dict:
        """
        Compute forward pass + losses without backward.
        Shared by train_step and validate_step.

        Returns:
            dict of loss values and intermediate tensors
        """
        self.model.train()  # still need train mode for dropout patterns consistency
        B = e3_data['coords'].shape[0]
        device = self.device

        # Sample random diffusion times
        t = torch.rand(B, device=device)
        t_se3 = torch.rand(B, device=device)

        # Add noise to PROTAC (rigid-body tr/rot DOFs only)
        noisy_ligand, tr_noise, rot_noise, rot_omega = self.diffusion.add_noise_protac(
            ligand_data['coords'].to(device), t, t, t,
        )
        # Add torsion noise (rotatable-bond angles) — the PROTAC-specific DOF
        tor_update = None
        if ligand_data.get('edge_mask') is not None and ligand_data.get('mask_rotate') is not None:
            noisy_ligand, tor_update = self.diffusion.add_torsion_noise(
                noisy_ligand, t,
                ligand_data['edge_index'], ligand_data['edge_mask'], ligand_data['mask_rotate'],
            )
        # Add noise to POI (SE3)
        noisy_poi, se3_tr_noise, se3_rot_noise, se3_rot_omega = self.diffusion.add_noise_se3(
            poi_data['coords'].to(device), t_se3, t_se3,
        )

        # Prepare noisy batch
        noisy_e3_data = {**e3_data, 'coords': e3_data['coords'].to(device)}
        noisy_poi_data = {**poi_data, 'coords': noisy_poi}
        noisy_ligand_data = {**ligand_data, 'coords': noisy_ligand}

        # Model forward
        output = self.model(noisy_e3_data, noisy_poi_data, noisy_ligand_data, t=t, t_se3=t_se3)

        # Score matching loss — matches DiffDock paper §4
        # Translation: weighted by σ² (R³ Euclidean DSM)
        # Rotation: normalized by SO(3) score_norm
        # Torsion: normalized by torus score_norm
        from .diffusion import so3_score_norm

        tr_sigma, rot_sigma, tor_sigma = self.diffusion.t_to_sigma(t, t, t)
        tr_score_target = -tr_noise / tr_sigma.unsqueeze(-1)**2
        diff_tr = output['s_q']['tr'] - tr_score_target
        loss_score_tr = (diff_tr**2 * tr_sigma.unsqueeze(-1)**2).mean()

        rot_score_target = self._compute_rot_score_target(rot_omega, rot_sigma)
        rot_norm = so3_score_norm(rot_sigma).unsqueeze(-1)
        loss_score_rot = ((output['s_q']['rot'] - rot_score_target)**2 / rot_norm**2).mean()

        if tor_update is not None and output['s_q']['tor'].numel() > 0:
            tor_score_target = self._compute_tor_score_target(tor_update, tor_sigma)
            tor_norm = self._compute_tor_score_norm(tor_sigma)  # E[score²], like so3
            loss_score_tor = ((output['s_q']['tor'] - tor_score_target)**2 / tor_norm).mean()
        else:
            loss_score_tor = torch.tensor(0.0, device=device)
        loss_score = loss_score_tr + loss_score_rot + loss_score_tor

        # SE3 score matching loss — same pattern as PROTAC
        se3_tr_sigma, se3_rot_sigma = self.diffusion.se3_t_to_sigma(t_se3, t_se3)
        se3_tr_target = -se3_tr_noise / se3_tr_sigma.unsqueeze(-1)**2
        diff_se3_tr = output['s_se3_components']['tr'] - se3_tr_target
        loss_se3_tr = (diff_se3_tr**2 * se3_tr_sigma.unsqueeze(-1)**2).mean()

        se3_rot_target = self._compute_rot_score_target(se3_rot_omega, se3_rot_sigma)
        se3_rot_norm = so3_score_norm(se3_rot_sigma).unsqueeze(-1)
        loss_se3_rot = ((output['s_se3_components']['rot'] - se3_rot_target)**2 / se3_rot_norm**2).mean()
        loss_se3 = loss_se3_tr + loss_se3_rot

        # Physical consistency loss
        denoised_ligand = self._estimate_denoised(noisy_ligand, output['s_q']['tr'], tr_sigma)
        denoised_poi = self._estimate_denoised_se3(noisy_poi, output['s_se3'], se3_tr_sigma)
        loss_phys = self._physical_loss(
            denoised_ligand, noisy_e3_data['coords'], denoised_poi
        )

        loss_total = (
            self.loss_weights['score'] * loss_score
            + self.loss_weights['phys'] * loss_phys
            + self.loss_weights['se3'] * loss_se3
        )

        return {
            'loss_total': loss_total,
            'loss_score': loss_score.item(),
            'loss_score_tr': loss_score_tr.item(),
            'loss_score_rot': loss_score_rot.item(),
            'loss_score_tor': loss_score_tor.item(),
            'loss_se3': loss_se3.item(),
            'loss_phys': loss_phys.item(),
        }

    def train_step(self, **batch) -> dict:
        """Forward + backward + optimizer step."""
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(**batch)
        loss_dict['loss_total'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.tribrain.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.global_step += 1

        result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        result['lr'] = self.optimizer.param_groups[0]['lr']
        return result

    @torch.no_grad()
    def validate_step(self, **batch) -> dict:
        """Forward + loss only, no backward."""
        loss_dict = self.compute_loss(**batch)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def _compute_rot_score_target(self, rot_omega, rot_sigma):
        """Compute SO(3) score target from the sampled so(3) rotation vector.

        Matches DiffDock `so3.score_vec(vec=omega, eps=sigma)`: the score of the
        isotropic-Gaussian-on-SO(3) density evaluated at the sampled rotation
        vector `omega`, returned in the same [B, 3] shape as the model's rot score.
        """
        from .diffusion import so3_score_vec
        return so3_score_vec(rot_sigma, rot_omega)

    def _compute_tor_score_target(self, tor_update, tor_sigma):
        """Torus (SO(2)) score target from the sampled torsion-angle noise.

        Matches DiffDock ``torus.score(torsion_updates, tor_sigma)``: the score of
        the wrapped-Gaussian-on-SO(2) density evaluated at each sampled rotation
        angle, returned in the same [B, n_rot] shape as the model's tor score.
        """
        from .diffdock.utils import torus
        B = tor_update.shape[0]
        targets = []
        for b in range(B):
            tgt = torus.score(
                tor_update[b].detach().cpu().numpy(),
                tor_sigma[b].detach().cpu().item(),
            )
            targets.append(torch.from_numpy(np.asarray(tgt, dtype=np.float32)).to(self.device))
        return torch.stack(targets, dim=0)

    def _compute_tor_score_norm(self, tor_sigma):
        """Torus score norm E[score²] for torsion-loss normalization (DiffDock)."""
        from .diffdock.utils import torus
        norm = torus.score_norm(tor_sigma.detach().cpu().numpy())
        return torch.from_numpy(np.asarray(norm, dtype=np.float32)).to(self.device).unsqueeze(-1)

    def _estimate_denoised(self, noisy_pos, score, sigma):
        """Tweedie's formula: x̂ = x_noisy + σ² * score"""
        return noisy_pos + sigma.unsqueeze(-1).unsqueeze(-1) ** 2 * score.unsqueeze(1)

    def _estimate_denoised_se3(self, noisy_poi, se3_score, sigma):
        """Estimate denoised POI positions using Tweedie's formula."""
        tr_score = se3_score[:, :3]
        # Apply to POI center
        poi_center = noisy_poi.mean(dim=1, keepdim=True)
        denoised_center = poi_center + sigma.unsqueeze(-1).unsqueeze(-1) ** 2 * tr_score.unsqueeze(1)
        return noisy_poi - poi_center + denoised_center

    def _physical_loss(self, ligand_coords, e3_coords, poi_coords):
        """
        Physical consistency loss: penalize steric clashes between
        PROTAC and both proteins.

        Per-atom VDW repulsion: Σ max(0, vdw_sum - dist)² over all
        PROTAC-protein atom pairs, using a fixed 3.4 Å contact distance.
        O(N_lig × N_prot) via torch.cdist — sufficient at MVP scale.
        """
        clash = torch.tensor(0.0, device=ligand_coords.device)
        for protein_coords in (e3_coords, poi_coords):
            if protein_coords is None:
                continue
            # [B, N_lig, N_prot] pairwise distances
            dist = torch.cdist(ligand_coords, protein_coords)
            # Repulsive term: atom pairs closer than 3.4 Å contribute
            clash = clash + (F.relu(3.4 - dist) ** 2).mean()
        return clash

    def scheduler_step(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def set_phase(self, phase: int):
        """
        Switch training phase.

        Phase 1: Train FiLM heads + SE3 head only (freeze judge/state)
        Phase 2: Unfreeze all TriBrain heads
        """
        if phase == 1:
            # Freeze judge and state heads
            for param in self.model.tribrain.judge_head.parameters():
                param.requires_grad = False
            for param in self.model.tribrain.state_head.parameters():
                param.requires_grad = False
        elif phase == 2:
            # Unfreeze everything in TriBrain
            for param in self.model.tribrain.parameters():
                param.requires_grad = True

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'tribrain_state_dict': self.model.tribrain.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint (for fine-tuning)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.tribrain.load_state_dict(ckpt['tribrain_state_dict'])
        self.global_step = ckpt['global_step']


# so3_score_norm — imported from tricoopdiff.diffusion where needed
