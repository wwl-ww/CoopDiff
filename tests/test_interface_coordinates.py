"""Checks for the two-interface coordinate diffusion formulation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tricoopdiff.diffusion import InterfaceDiffusion


def _triangle(center):
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    ).unsqueeze(0) + torch.tensor(center).view(1, 1, 3)


def test_interface_composition_preserves_poi_side_relative_geometry():
    diffusion = InterfaceDiffusion(
        temp_sampling_tr=1.0,
        temp_sampling_rot=1.0,
    )
    ligand = _triangle([2.0, -1.0, 0.5])
    poi = _triangle([-3.0, 4.0, 1.0])
    target_ligand = _triangle([7.0, 2.0, -3.0])
    tr_score = torch.tensor([[0.2, -0.3, 0.1]])
    rot_score = torch.tensor([[0.0, 0.0, 0.4]])
    time = torch.tensor([0.3])
    dt = torch.tensor([0.05])

    poi_side_proposal, _ = diffusion.reverse_step_protac(
        ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        tor_score=torch.empty(1, 0),
        t_tr=time,
        t_rot=time,
        t_tor=time,
        dt_tr=dt,
        dt_rot=dt,
        dt_tor=dt,
        add_noise=False,
    )
    updated_poi = diffusion.reverse_step_poi_ligand_interface(
        poi,
        source_ligand_pos=ligand,
        target_ligand_pos=target_ligand,
        tr_score=tr_score,
        rot_score=rot_score,
        t_tr=time,
        t_rot=time,
        dt_tr=dt,
        dt_rot=dt,
        add_noise=False,
    )
    assert updated_poi.shape == poi.shape
    assert torch.isfinite(updated_poi).all()
    assert torch.equal(ligand, _triangle([2.0, -1.0, 0.5]))
    # Applying one rigid transform to POI and the sub2 proposal preserves all
    # cross-interface distances when that proposal lands on the sub1 target.
    assert torch.allclose(
        torch.cdist(updated_poi, target_ligand),
        torch.cdist(poi, poi_side_proposal),
        atol=1e-5,
    )


def test_interface_pair_noise_reconstructs_finite_frames():
    torch.manual_seed(0)
    diffusion = InterfaceDiffusion()
    ligand = _triangle([1.0, 2.0, 3.0])
    poi = _triangle([8.0, -2.0, 1.0])
    noisy_ligand, noisy_poi, noise, global_to_poi = (
        diffusion.add_noise_interface_pair(
            ligand, poi, torch.tensor([0.4]), torch.tensor([0.4]),
        )
    )
    assert noisy_ligand.shape == ligand.shape
    assert noisy_poi.shape == poi.shape
    assert global_to_poi.shape == (1, 3, 3)
    assert torch.isfinite(noisy_ligand).all()
    assert torch.isfinite(noisy_poi).all()
    assert torch.allclose(
        torch.det(global_to_poi), torch.ones(1), atol=1e-5,
    )
    assert set(noise) == {"e3_ligand", "poi_ligand"}


if __name__ == "__main__":
    test_interface_composition_preserves_poi_side_relative_geometry()
    test_interface_pair_noise_reconstructs_finite_frames()
    print("Interface-coordinate tests passed")
