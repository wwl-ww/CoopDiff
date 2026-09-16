"""Absolute quality, ordered thresholds and frozen-generator boundaries."""

from pathlib import Path
import json
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.pose_ranking import StateQualityRanker, state_quality_loss
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluation_common import sha256_file
from train_state_ranker import combine_training_parts, evaluate, load_split, quality_class_support
from train_state_ranker import main as train_main
from tricoopdiff.ranking_inference import GENERATOR_KEYS


def test_quality_outputs_are_bounded_and_thresholds_ordered():
    model = StateQualityRanker(12, 16)
    output = model(torch.randn(8, 12))
    assert ((output['q_hat'] > 0) & (output['q_hat'] < 1)).all()
    assert (output['p49'] <= output['p23']).all()
    assert ((output['p23'] >= 0) & (output['p23'] <= 1)).all()


def test_quality_loss_updates_only_ranker_and_rejects_invalid_truth():
    torch.manual_seed(7)
    features = torch.randn(8, 12, requires_grad=True)
    model = StateQualityRanker(12, 16)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    labels = torch.linspace(0., 1., 8)
    loss = state_quality_loss(model(features), labels)
    loss.backward()
    assert features.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.head.weight)
    try:
        state_quality_loss(model(features), labels * 2)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid quality labels must be rejected')


def test_archive_loader_keeps_failed_probe_and_separates_labels(tmp_path):
    archives, labels = tmp_path / 'archives', tmp_path / 'labels'
    archives.mkdir()
    labels.mkdir()
    manifest = {'cases': ['TEST_A_B_LIG']}
    (archives / 'manifest.json').write_text(json.dumps(manifest))
    (labels / 'manifest.json').write_text(json.dumps({
        'archive_manifest_sha256': sha256_file(archives / 'manifest.json')}))
    archive_path = archives / 'TEST_A_B_LIG.pt'
    torch.save({'case': 'TEST_A_B_LIG', 'manifest': manifest, 'pool': [{'source': 'sdpi'}],
                'split': 'train', 'probe_feature_dim': 2, 'pairs': [{
        'pool_index': 0, 'source': 'sdpi', 'valid': False,
        'initial_features': {'geometry': [1., 2.], 'frozen': None, 'probe_valid': False},
        'final_features': None,
    }]}, archive_path)
    (archives / 'TEST_A_B_LIG.complete.json').write_text(json.dumps({
        'archive_sha256': sha256_file(archive_path)}))
    label_path = labels / 'TEST_A_B_LIG.json'
    payload = {'case': 'TEST_A_B_LIG', 'archive_sha256': sha256_file(archive_path), 'labels': [{
        'pool_index': 0, 'initial': {'dockq': .2}, 'final': None,
    }]}
    label_path.write_text(json.dumps(payload))
    _, before = load_split(archives, labels, 'train')
    assert before[0]['stages'] == ['initial'], 'failed final must not erase its initial state'
    assert before[0]['features'].shape == (1, 5)
    assert torch.equal(before[0]['features'][0, :2], torch.zeros(2))
    assert before[0]['features'][0, -1] == 0
    payload['labels'][0]['initial']['dockq'] = .8
    label_path.write_text(json.dumps(payload))
    _, after = load_split(archives, labels, 'train')
    assert torch.equal(before[0]['features'], after[0]['features']), 'labels must not affect features'
    assert not torch.equal(before[0]['dockq'], after[0]['dockq'])
    payload['labels'].append(dict(payload['labels'][0]))
    label_path.write_text(json.dumps(payload))
    try:
        load_split(archives, labels, 'train')
    except ValueError as error:
        assert 'duplicate' in str(error)
    else:
        raise AssertionError('duplicate labels must not silently overwrite each other')
    payload['labels'].pop()
    payload['archive_sha256'] = 'wrong'
    label_path.write_text(json.dumps(payload))
    try:
        load_split(archives, labels, 'train')
    except ValueError as error:
        assert 'identity mismatch' in str(error)
    else:
        raise AssertionError('mismatched archive labels must be rejected')


def test_ranker_evaluation_retains_missing_final_in_denominator():
    model = StateQualityRanker(2, 4)
    cases = [{'case': 'test', 'features': torch.ones(1, 2),
              'dockq': torch.tensor([.5]), 'stages': ['initial']}]
    result = evaluate(model, cases, torch.zeros(2), torch.ones(2), 'cpu')['leaderboards']
    assert result['final']['cases'] == result['final']['missing'] == 1
    assert result['final']['hits_0.23'] == result['final']['mean_top1'] == 0
    assert result['union']['mean_top1'] == .5
    support = quality_class_support(cases)
    assert support['class_counts_lt23_23to49_ge49'] == [0, 0, 1]
    assert support['classes_with_no_labels'] == [0, 1]


def test_training_diagnostic_evaluation_preserves_rng_and_weights():
    model = StateQualityRanker(2, 4).train()
    cases = [{'case': 'test', 'features': torch.ones(2, 2),
              'dockq': torch.tensor([.1, .3]), 'stages': ['initial', 'final']}]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    result = evaluate(model, cases, torch.zeros(2), torch.ones(2), 'cpu')
    assert torch.equal(rng, torch.random.get_rng_state()), 'diagnostics must not change training RNG'
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert result['leaderboards']['union']['cases'] == 1


def test_training_shards_keep_order_and_reject_overlap_or_generator_drift():
    keys = (*GENERATOR_KEYS, 'walkers', 'selection', 'proposal_seed', 'selection_seed', 'diffusion_seed')
    manifest = dict.fromkeys(keys, 'same')
    first, second = [{'case': 'AAAA_A_B_LIG'}], [{'case': 'BBBB_A_B_LIG'}]
    _, combined = combine_training_parts([(manifest, first), (manifest, second)])
    assert combined == first + second
    for other_manifest, other_rows in (({**manifest, 'steps': 'changed'}, second),
                                       (manifest, [{'case': 'AAAA_C_D_OTHER'}])):
        try:
            combine_training_parts([(manifest, first), (other_manifest, other_rows)])
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible/overlapping archive shards accepted')


def test_multishard_training_cli_normalizes_only_the_selected_prefix(tmp_path):
    paths = []
    keys = (*GENERATOR_KEYS, 'walkers', 'selection', 'proposal_seed', 'selection_seed', 'diffusion_seed')
    for index, (case, split, value) in enumerate((('AAAA_A_B_LIG', 'train', 1.),
                                                ('BBBB_A_B_LIG', 'train', 99.),
                                                ('CCCC_A_B_LIG', 'val', 3.))):
        archives, labels = tmp_path / f'archives{index}', tmp_path / f'labels{index}'
        archives.mkdir(); labels.mkdir()
        manifest = {**dict.fromkeys(keys, 'same'), 'cases': [case]}
        (archives / 'manifest.json').write_text(json.dumps(manifest))
        (labels / 'manifest.json').write_text(json.dumps({
            'archive_manifest_sha256': sha256_file(archives / 'manifest.json')}))
        feature = {'geometry': [value] * 14, 'frozen': [value, value], 'probe_valid': True}
        archive = {'case': case, 'split': split, 'manifest': manifest, 'probe_feature_dim': 2,
                   'pool': [{'source': 'sdpi'}], 'pairs': [{'pool_index': 0, 'source': 'sdpi', 'valid': True,
                                                         'initial_features': feature, 'final_features': feature}]}
        path = archives / f'{case}.pt'
        torch.save(archive, path)
        digest = sha256_file(path)
        (archives / f'{case}.complete.json').write_text(json.dumps({'archive_sha256': digest}))
        (labels / f'{case}.json').write_text(json.dumps({'case': case, 'archive_sha256': digest,
            'labels': [{'pool_index': 0, 'initial': {'dockq': .1}, 'final': {'dockq': .2}}]}))
        paths.append((archives, labels))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'train_cases': ['AAAA_A_B_LIG', 'BBBB_A_B_LIG'],
                                 'val_cases': ['CCCC_A_B_LIG'], 'benchmark_pdbs_excluded': []}))
    out = tmp_path / 'trained'
    argv = ['train', '--train-archives', str(paths[0][0]), '--train-labels', str(paths[0][1]),
            '--additional-train-archives', str(paths[1][0]), '--additional-train-labels', str(paths[1][1]),
            '--val-archives', str(paths[2][0]), '--val-labels', str(paths[2][1]),
            '--split-audit', str(audit), '--out', str(out), '--epochs', '1', '--train-limit', '1',
            '--geometry-only', '--diagnostics']
    with patch('sys.argv', argv):
        train_main()
    payload = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    assert len(payload['training_shards_sha256']) == 2
    assert payload['training_case_ids'] == ['AAAA_A_B_LIG']
    assert torch.allclose(payload['mean'], torch.full((14,), 2.).log()), 'unused cases leaked into normalization'
    report = json.loads((out / 'report.json').read_text())
    assert report['train_cases'] == report['optimizer_updates'] == 1
