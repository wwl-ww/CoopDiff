import torch

from tricoopdiff.ranking.ranking_inference import (
    PoseRanker, listwise_ranking_loss, selected_pose_metrics,
)


def test_listwise_loss_prefers_dockq_order():
    dockq = torch.tensor([0.01, 0.10, 0.40])
    correct = listwise_ranking_loss(torch.tensor([-2.0, 0.0, 2.0]), dockq)
    reversed_order = listwise_ranking_loss(torch.tensor([2.0, 0.0, -2.0]), dockq)
    assert correct < reversed_order


def test_selected_pose_metrics_distinguish_ranked_and_oracle():
    result = selected_pose_metrics(
        torch.tensor([3.0, 2.0, 1.0]), torch.tensor([0.1, 0.3, 0.8]),
    )
    assert abs(result["selected_dockq"] - 0.1) < 1e-6
    assert abs(result["oracle_dockq"] - 0.8) < 1e-6
    assert not result["selected_hit"] and result["oracle_hit"]


def test_pose_ranker_shape_contract():
    model = PoseRanker(input_dim=16, hidden_dim=8)
    assert model(torch.randn(5, 16)).shape == (5,)
