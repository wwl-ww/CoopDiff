"""Replay must preserve full-list gradients, including shared-task accumulation."""
import copy
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from test_structural_future import example
from tricoopdiff.structural_ranker import StructuralQualityRanker, decision_ranking_loss, structural_ranking_loss
from tricoopdiff.structural_ranking_data import future_utility_case
from tricoopdiff.ranking_backward import backward_list
from scripts.train_structural_ranker import score_case


def test_replay_matches_full_gradients():
    for interface in (0, 1):
        for chunk in (1, 3):
            case = example()
            model = StructuralQualityRanker(16, 16, interface_layers=interface)
            replay = copy.deepcopy(model)
            objective = lambda output: decision_ranking_loss(
                output, case['labels'], case['stages'], quality_weight=.2, top_weight=1.)
            loss = objective(score_case(model, case, 'cpu'))
            loss.backward()
            actual, _ = backward_list(replay, case, 'cpu', objective, chunk_size=chunk)
            assert torch.allclose(actual, loss)
            future = future_utility_case(case, target='union')
            objective = lambda output: structural_ranking_loss(output, future['labels'], top_weight=1.)
            (.7 * objective(score_case(model, future, 'cpu'))).backward()
            backward_list(replay, future, 'cpu', objective, chunk_size=chunk, weight=.7)
            for (name, left), (_, right) in zip(model.named_parameters(), replay.named_parameters()):
                if left.grad is None:
                    assert right.grad is None, name
                else:
                    assert torch.allclose(left.grad, right.grad, atol=2e-6, rtol=2e-4), name


if __name__ == '__main__':
    test_replay_matches_full_gradients()
    print('Gradient replay: exact shared-task gradients passed, two chunk sizes and two architectures')
