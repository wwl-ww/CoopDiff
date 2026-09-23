"""Pure regression preserves model/input contracts and exact replay gradients."""
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch
from test_structural_ranker import fixture
from test_structural_ranking_data import archive_fixture
from tricoopdiff.structural_ranker import (
    StructuralQualityRanker, dockq_regression_loss, pairwise_ranking_loss,
    regression_pairwise_loss, hard_top_pairwise_loss,
    top_tier_classification_loss, top_focused_ranking_loss,
)
from tricoopdiff.structural_ranking_data import file_digest
from tricoopdiff.ranking_backward import backward_list
from scripts.train_structural_ranker import main
from scripts.rank_structural_archive import rank_archive


def test_regression_replay_and_target_detach():
    graph = fixture(); graph.pop('stage'); graph.pop('source')
    model = StructuralQualityRanker(16, 16)
    replay = copy.deepcopy(model)
    case = {'graphs': [graph] * 2, 'stages': ['initial', 'final'], 'sources': ['sdpi'] * 2}
    labels = torch.tensor([.1, .7], requires_grad=True)
    loss = lambda out: dockq_regression_loss(out, labels)
    loss(model.forward_list(case['graphs'], stages=case['stages'], sources=case['sources'])).backward()
    backward_list(replay, case, 'cpu', loss, chunk_size=1)
    assert labels.grad is None
    assert model.rank_head.weight.grad.abs().sum() > 0
    assert model.encoder.protein_feature_proj.weight.grad.abs().sum() > 0
    for (name, a), (_, b) in zip(model.named_parameters(), replay.named_parameters()):
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=2e-4), name


def test_pairwise_and_combined_objectives_reward_correct_order():
    labels = torch.tensor([.1, .8, .4])
    correct = {'score': torch.tensor([-2., 2., 0.], requires_grad=True)}
    reversed_order = {'score': torch.tensor([2., -2., 0.], requires_grad=True)}
    assert pairwise_ranking_loss(correct, labels) < pairwise_ranking_loss(reversed_order, labels)
    combined = regression_pairwise_loss(correct, labels)
    assert torch.isfinite(combined)
    combined.backward()
    assert correct['score'].grad.abs().sum() > 0


def test_top_focused_objectives_reward_top_order_and_tier_calibration():
    labels = torch.tensor([.1, .8, .4, .05])
    correct = {'score': torch.tensor([-2., 3., 1., -3.], requires_grad=True),
               'p23': torch.tensor([.1, .9, .8, .1], requires_grad=True)}
    wrong = {'score': -correct['score'].detach(),
             'p23': 1 - correct['p23'].detach()}
    assert hard_top_pairwise_loss(correct['score'], labels, negatives=2) < hard_top_pairwise_loss(wrong['score'], labels, negatives=2)
    assert top_tier_classification_loss(correct, labels) < top_tier_classification_loss(wrong, labels)
    loss = top_focused_ranking_loss(correct, labels, hard_weight=.5, tier_weight=.5)
    loss.backward()
    assert correct['score'].grad.abs().sum() > 0
    assert correct['p23'].grad.abs().sum() > 0


def test_regression_training_and_inference():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary); arguments = []
        for split, name in [('train', '1ABC_A_B_L'), ('val', '2ABC_A_B_L')]:
            archive, labels = archive_fixture()
            archive.update(case=name, group=split, split=split)
            labels['case'] = name
            manifest = dict(cases=[name], checkpoint_sha256='test', steps=20, pool={}, selection={})
            archive['manifest'] = manifest
            poses, targets = root / split, root / (split + '_labels')
            poses.mkdir(); targets.mkdir()
            (poses / 'manifest.json').write_text(json.dumps(manifest))
            torch.save(archive, poses / f'{name}.pt')
            digest = file_digest(poses / f'{name}.pt')
            (poses / f'{name}.complete.json').write_text(json.dumps({'archive_sha256': digest}))
            labels['archive_sha256'] = digest
            (targets / f'{name}.json').write_text(json.dumps(labels))
            (targets / 'manifest.json').write_text(json.dumps({'archive_manifest_sha256': file_digest(poses / 'manifest.json')}))
            arguments += [f'--{split}-archives', str(poses), f'--{split}-labels', str(targets)]
        out = root / 'run'
        main(arguments + ['--out', str(out), '--objective', 'regression', '--top-weight', '0',
                          '--epochs', '1', '--hidden-dim', '16', '--shuffle-repeats', '0',
                          '--backward-chunk-size', '1'])
        checkpoint = torch.load(out / 'best.pt', weights_only=False)
        assert checkpoint['metadata']['arguments']['objective'] == 'regression'
        assert checkpoint['head_supervision'] == {'rank': True, 'quality_aux': False, 'future': False}
        assert len(rank_archive(out / 'best.pt', root / 'val/2ABC_A_B_L.pt')['rankings']['union']) == 2
