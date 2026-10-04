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
from tricoopdiff.ranking.ranking_inference import StateQualityRanker
from tricoopdiff.ranking.ranking_inference import (export_quality_checkpoint, feature_vector,
                                         load_quality_checkpoint, predict_archive)
from tricoopdiff.ranking.ranking_audit import case_metrics, join_labels, label_support, random_hit_probability, summarize_cases


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
