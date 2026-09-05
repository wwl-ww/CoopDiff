"""Unit checks for energy-consistent two-interface score coordination."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tricoopdiff.merging import coordinate_joint_interface_scores


def _scores(batch=2, torsions=4):
    return {
        "tr_score": torch.randn(batch, 3),
        "rot_score": torch.randn(batch, 3),
        "tor_score": torch.randn(batch, torsions),
    }


def test_zero_coupling_is_sub1_transparent():
    torch.manual_seed(0)
    scores_1, scores_2 = _scores(), _scores()
    ligand, reaction = coordinate_joint_interface_scores(
        scores_1, scores_2, torch.zeros(2, 3),
    )
    for key in ("tr_score", "rot_score", "tor_score"):
        assert torch.equal(ligand[key], scores_1[key]), key
    assert torch.equal(reaction["se3_score"], torch.zeros(2, 6))


def test_unit_coupling_adds_both_interfaces_with_reaction():
    torch.manual_seed(1)
    scores_1, scores_2 = _scores(), _scores()
    ligand, reaction = coordinate_joint_interface_scores(
        scores_1, scores_2, torch.ones(2, 3),
    )
    for key in ("tr_score", "rot_score", "tor_score"):
        assert torch.allclose(ligand[key], scores_1[key] + scores_2[key]), key
    assert torch.allclose(reaction["tr_score"], -scores_2["tr_score"])
    assert torch.allclose(reaction["rot_score"], -scores_2["rot_score"])
    assert "tor_score" not in reaction


def test_sub2_rigid_contribution_is_equal_and_opposite():
    torch.manual_seed(2)
    scores_1, scores_2 = _scores(), _scores()
    coupling = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.3, 0.9]])
    ligand, reaction = coordinate_joint_interface_scores(
        scores_1, scores_2, coupling,
    )
    assert torch.allclose(
        ligand["tr_score"] - scores_1["tr_score"] + reaction["tr_score"],
        torch.zeros_like(scores_1["tr_score"]),
        atol=2e-7,
    )
    assert torch.allclose(
        ligand["rot_score"] - scores_1["rot_score"] + reaction["rot_score"],
        torch.zeros_like(scores_1["rot_score"]),
        atol=2e-7,
    )


def test_vector_coordination_is_rotation_equivariant():
    torch.manual_seed(3)
    scores_1, scores_2 = _scores(batch=1), _scores(batch=1)
    coupling = torch.tensor([[0.4, 0.6, 0.2]])
    rotation = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ligand, reaction = coordinate_joint_interface_scores(
        scores_1, scores_2, coupling,
    )
    rotated_1 = {key: value.clone() for key, value in scores_1.items()}
    rotated_2 = {key: value.clone() for key, value in scores_2.items()}
    for key in ("tr_score", "rot_score"):
        rotated_1[key] = rotated_1[key] @ rotation
        rotated_2[key] = rotated_2[key] @ rotation
    ligand_rot, reaction_rot = coordinate_joint_interface_scores(
        rotated_1, rotated_2, coupling,
    )
    for key in ("tr_score", "rot_score"):
        assert torch.allclose(ligand_rot[key], ligand[key] @ rotation)
        assert torch.allclose(reaction_rot[key], reaction[key] @ rotation)


if __name__ == "__main__":
    test_zero_coupling_is_sub1_transparent()
    test_unit_coupling_adds_both_interfaces_with_reaction()
    test_sub2_rigid_contribution_is_equal_and_opposite()
    test_vector_coordination_is_rotation_equivariant()
    print("Joint coordination tests passed")
