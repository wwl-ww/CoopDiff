"""Deployment without labels, exact random baselines, and source/failed-state accounting."""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import sha256_file
from train_pose_ranker import load_cases as load_pose_cases
from tricoopdiff.ranking.ranking_inference import (
    PoseRanker, listwise_ranking_loss, selected_pose_metrics,
)
from tricoopdiff.ranking.ranking_inference import StateQualityRanker
from tricoopdiff.ranking.ranking_inference import (export_quality_checkpoint, feature_vector,
                                         load_quality_checkpoint, predict_archive)
from tricoopdiff.ranking.ranking_audit import case_metrics, join_labels, label_support, random_hit_probability, summarize_cases
from tricoopdiff.ranking.ranking_audit import paired_case_comparison


def test_pose_ranker_and_loss_preserve_score_order():
    model = PoseRanker(input_dim=16, hidden_dim=8)
    assert model(torch.randn(5, 16)).shape == (5,)
    dockq = torch.tensor([0.01, 0.10, 0.40])
    correct = listwise_ranking_loss(torch.tensor([-2.0, 0.0, 2.0]), dockq)
    reversed_order = listwise_ranking_loss(torch.tensor([2.0, 0.0, -2.0]), dockq)
    assert correct < reversed_order
    result = selected_pose_metrics(
        torch.tensor([3.0, 2.0, 1.0]), torch.tensor([0.1, 0.3, 0.8]),
    )
    assert abs(result['selected_dockq'] - 0.1) < 1e-6
    assert abs(result['oracle_dockq'] - 0.8) < 1e-6
    assert not result['selected_hit'] and result['oracle_hit']


def test_load_pose_cases_preserves_recorded_rank_features(tmp_path):
    payload = {
        'case': '1ABC_A_B_LIG',
        'run_config': {'rank_features_saved': True},
        'walkers': [
            {'dockq': 0.1 + index / 100, 'rank_features': [index, index + 1]}
            for index in range(8)
        ],
    }
    (tmp_path / 'case.json').write_text(json.dumps(payload))
    cases = load_pose_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0]['features'].shape == (8, 2)


def paired_report(values, label='label'):
    cases = {}
    files = {}
    for index, value in enumerate(values):
        case = f'case{index}'
        cases[case] = {'count': 2, 'top1': value, 'oracle': .8}
        files[case] = {'label': label, 'prediction': f'prediction{index}'}
    return {'labels_manifest_sha256': 'manifest', 'files_sha256': files,
            'cases': {'all/union': cases}}


def test_paired_comparison_averages_seeds_before_resampling_cases():
    left = [paired_report([.6, .2]), paired_report([.4, .4])]
    right = [paired_report([.2, .2]), paired_report([.2, .2])]
    result = paired_case_comparison(left, right, 'all/union', repeats=100, seed=7)
    assert abs(result['mean_delta_top1'] - .2) < 1e-12
    assert result['case_wins'] == 2 and result['case_losses'] == result['case_ties'] == 0
    assert result['seeds_per_method'] == 2 and result['cases'] == 2
    assert result['mean_seed_hit_count_delta']['0.23'] == 1.5


def test_paired_comparison_rejects_label_or_candidate_drift():
    original = paired_report([.4, .3])
    for changed in (paired_report([.4, .3], label='other'), paired_report([.4, .3])):
        if changed['files_sha256']['case0']['label'] == 'label':
            changed['cases']['all/union']['case0']['oracle'] = .7
        try:
            paired_case_comparison([original], [changed], 'all/union')
        except ValueError:
            pass
        else:
            raise AssertionError('paired audit accepted label/candidate drift')


def example_archive():
    features = {'geometry': [1.] * 14, 'probe_valid': True, 'frozen': [2., 3.]}
    return {'case': 'demo', 'probe_feature_dim': 2, 'pool': [{'source': 'sdpi'}, {'source': 'bitseed'}],
            'pairs': [{'pool_index': 0, 'source': 'sdpi', 'valid': False,
                       'initial_features': features, 'final_features': None},
                      {'pool_index': 1, 'source': 'bitseed', 'valid': True,
                       'initial_features': features, 'final_features': features}]}


def test_deploy_features_ignore_truth_metadata_and_zero_failed_probes():
    features = example_archive()['pairs'][0]['initial_features']
    original = feature_vector(features, 2)
    poisoned = {**features, 'dockq': 1., 'final_dockq': 1., 'source': 'bitseed', 'stage': 'final'}
    assert torch.equal(original, feature_vector(poisoned, 2))
    failed = feature_vector({**features, 'probe_valid': False, 'frozen': None}, 2)
    assert torch.equal(failed[:2], torch.zeros(2)) and failed[-1] == 0


def test_ranking_ties_and_failed_final_keep_initial_without_extra_pool_states():
    archive = example_archive()
    archive['pool'].append({'source': 'sdpi'})  # Unsampled candidate must not enter output.
    before = predict_archive(archive)
    assert len(before['rows']) == 3
    assert [(r['pool_index'], r['stage']) for r in before['selected_top5']['union']] == [
        (0, 'initial'), (1, 'initial'), (1, 'final')]
    archive['pairs'].reverse()
    assert predict_archive(archive) == before, 'ties must not depend on serialized pair order'
    assert before['selected_top5']['final'][0]['source'] == 'bitseed'


def test_checkpoint_contract_and_real_model_prediction():
    model = StateQualityRanker(17, 4)
    manifest = {k: 'test' for k in ('protocol', 'checkpoint_sha256', 'probe', 'runtime_sha256',
                                   'pool', 'steps', 'expert_assets')}
    checkpoint = {'architecture': 'state_quality_ranker_v1', 'model_config': model.config,
                  'state_dict': model.state_dict(), 'generator': manifest, 'geometry_only': False,
                  'mean': torch.zeros(17), 'scale': torch.ones(17)}
    exported = export_quality_checkpoint({**checkpoint, 'metrics': {'secret_dockq': .9},
                                          'generator': {**manifest, 'cases': ['hidden_id']}})
    assert 'metrics' not in exported and 'cases' not in exported['generator']
    loaded = load_quality_checkpoint(checkpoint, manifest)
    result = predict_archive(example_archive(), loaded, checkpoint)
    assert all(0 <= r['p49'] <= r['p23'] <= 1 for r in result['rows'])
    try:
        load_quality_checkpoint(checkpoint, {**manifest, 'checkpoint_sha256': 'different'})
    except ValueError:
        pass
    else:
        raise AssertionError('generator mismatch must fail before inference')


def test_label_join_rejects_omitted_state_and_wrong_archive():
    prediction = {'case': 'demo', 'archive_sha256': 'abc', **predict_archive(example_archive())}
    label = {'case': 'demo', 'archive_sha256': 'abc', 'labels': [
        {'pool_index': 0, 'initial': {'dockq': .1}, 'final': None},
        {'pool_index': 1, 'initial': {'dockq': .6}, 'final': {'dockq': .3}}]}
    rows = join_labels(prediction, label)
    assert case_metrics(rows)['top1'] == .1
    assert case_metrics(rows)['oracle'] == .6
    for invalid in ({**prediction, 'archive_sha256': 'wrong'},
                    {**prediction, 'rows': prediction['rows'][:-1]}):
        try:
            join_labels(invalid, label)
        except ValueError:
            pass
        else:
            raise AssertionError('incomplete/mismatched ranking must not be scored')


def test_random_expectation_zero_oracle_and_absent_source_denominators():
    assert abs(random_hit_probability(10, 1, 5) - .5) < 1e-12
    row = {'pool_index': 0, 'stage': 'initial', 'source': 'sdpi', 'score': 0., 'dockq': .5,
           'p23': .8, 'p49': .6}
    result = summarize_cases({'a': [row], 'b': [{**row, 'dockq': 0.}]})['summary']
    assert result['sdpi/initial']['mean_top1'] == .25
    assert result['sdpi/initial']['retention_defined_cases'] == 1
    assert result['sdpi/initial']['zero_oracle_cases'] == 1
    assert result['bitseed/union']['cases'] == result['bitseed/union']['missing'] == 2
    assert result['all/final']['missing'] == 2
    assert result['all/initial']['thresholds']['0.23']['random_top1_hit'] == 1.
    assert abs(result['all/initial']['brier_case_mean']['0.23'] - .34) < 1e-6


def test_class_support_counts_cases_not_correlated_states():
    positive = {'stage': 'initial', 'source': 'bitseed', 'dockq': .3}
    negative = {'stage': 'initial', 'source': 'sdpi', 'dockq': .1}
    report = label_support({'positive': [positive, {**positive, 'stage': 'final'}], 'negative': [negative]})
    assert report['all/union']['class_counts_lt23_23to49_ge49'] == [1, 2, 0]
    assert report['all/union']['cases_with_ge23'] == 1
    assert report['all/union']['classes_with_no_labels'] == [2]
    assert report['sdpi/final']['states'] == 0 and report['sdpi/final']['cases'] == 2


def test_native_free_cli_with_no_structure_or_label_files(tmp_path):
    archive = example_archive()
    manifest = {'cases': ['demo']}
    archive['manifest'] = manifest
    directory = tmp_path / 'archives'
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    path = directory / 'demo.pt'
    torch.save(archive, path)
    (directory / 'demo.complete.json').write_text(json.dumps({'archive_sha256': sha256_file(path)}))
    out = tmp_path / 'ranks'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    subprocess.run([sys.executable, str(ROOT / 'scripts/rank_pose_archive.py'),
                    '--archives', str(directory), '--out', str(out)],
                   cwd=tmp_path, env=environment, check=True, capture_output=True, timeout=30)
    assert json.loads((out / 'complete.json').read_text())['cases'] == ['demo']
    assert json.loads((out / 'manifest.json').read_text())['native_or_labels_read'] is False
    assert len(json.loads((out / 'demo.json').read_text())['rows']) == 3


def test_complete_comparison_exports_weights_and_runs_three_cpu_methods(tmp_path):
    manifest = {k: 'fixture' for k in ('protocol', 'checkpoint_sha256', 'probe', 'runtime_sha256',
                                      'pool', 'steps', 'expert_assets')}
    manifest['cases'] = ['demo']
    archive = {**example_archive(), 'manifest': manifest}
    archives, labels = tmp_path / 'val_archives', tmp_path / 'val_labels'
    archives.mkdir()
    labels.mkdir()
    (archives / 'manifest.json').write_text(json.dumps(manifest))
    torch.save(archive, archives / 'demo.pt')
    digest = sha256_file(archives / 'demo.pt')
    (archives / 'demo.complete.json').write_text(json.dumps({'archive_sha256': digest}))
    (labels / 'manifest.json').write_text(json.dumps({
        'archive_manifest_sha256': sha256_file(archives / 'manifest.json')}))
    (labels / 'demo.json').write_text(json.dumps({'case': 'demo', 'archive_sha256': digest, 'labels': [
        {'pool_index': 0, 'initial': {'dockq': .1}, 'final': None},
        {'pool_index': 1, 'initial': {'dockq': .6}, 'final': {'dockq': .3}}]}))
    (tmp_path / 'pipeline_status.json').write_text(json.dumps({'stage': 'complete'}))
    for mode, dimension, geometry_only in (('ranker_geometry', 14, True), ('ranker_frozen_geometry', 17, False)):
        destination = tmp_path / mode
        destination.mkdir()
        model = StateQualityRanker(dimension, 4)
        torch.save({'architecture': 'state_quality_ranker_v1', 'model_config': model.config,
                    'state_dict': model.state_dict(), 'generator': manifest, 'geometry_only': geometry_only,
                    'mean': torch.zeros(dimension), 'scale': torch.ones(dimension),
                    'metrics': {'labels_must_not_enter_inference': True}}, destination / 'best.pt')
    output = tmp_path / 'comparison'
    subprocess.run([sys.executable, str(ROOT / 'scripts/compare_state_rankers.py'), '--run-dir', str(tmp_path),
                    '--out', str(output)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                   capture_output=True, check=True, timeout=60)
    result = json.loads((output / 'comparison.json').read_text())
    assert len(result['reports']) == 3
    assert (output / 'comparison.md').is_file()
    for mode in ('ranker_geometry', 'ranker_frozen_geometry'):
        exported = torch.load(output / f'{mode}_inference.pt', map_location='cpu', weights_only=False)
        assert 'metrics' not in exported
        assert result['reports'][mode]['all/final']['cases'] == 1
