"""Full-pool selector features and retention accounting."""

from pathlib import Path
import importlib.util
import json
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tricoopdiff.initial_selector import (future_selector_retention,
                                          future_selector_retention_at_indices,
                                          initial_pool_features,
                                          diversity_aware_indices, exact_random_selection,
                                          selector_retention,
                                          selector_retention_at_indices)
from tricoopdiff.ranking_archive import state_hash

SPEC = importlib.util.spec_from_file_location(
    'train_initial_selector', ROOT / 'scripts/train_initial_selector.py')
train_selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_selector)
from scripts.evaluation_common import sha256_file


def archive():
    pool = []
    for index in range(256):
        state = {'ligand': torch.tensor([[float(index), 0., 0.]]),
                 'poi': torch.tensor([[0., 1., 0.]]), 'body_frame': torch.eye(3)}
        pool.append({'candidate_id': ['bitseed' if index < 192 else 'sdpi', index],
                     'source': 'bitseed' if index < 192 else 'sdpi', 'state': state,
                     'state_sha256': state_hash(state)})
    return {'protocol': 'initial_final_pose_archive_v1', 'pool': pool,
            'input_graphs': {'e3_data': {'coords': torch.zeros(1, 1, 3)}}}


def test_all_pool_features_are_aligned_and_checksummed():
    value = archive()
    rows = initial_pool_features(value)
    assert len(rows) == 256 and rows[17]['pool_index'] == 17
    assert rows[0]['geometry'].shape == (14,)
    value['pool'][2]['state']['ligand'][0, 0] += 1
    try:
        initial_pool_features(value)
    except ValueError as error:
        assert 'checksum' in str(error)
    else:
        raise AssertionError('mutated pool state accepted')


def test_retention_tracks_oracle_and_source_budget():
    scores = torch.arange(256).float()
    dockq = torch.zeros(256); dockq[250] = .3; dockq[10] = .8
    sources = ['bitseed'] * 192 + ['sdpi'] * 64
    result = selector_retention(scores, dockq, sources)
    assert abs(result['selected_best'] - .3) < 1e-6
    assert abs(result['full_pool_oracle'] - .8) < 1e-6
    assert result['selected_hit'] == result['full_pool_hit'] == 1
    assert result['oracle_index_retained'] == 0
    assert result['selected_source_counts'] == {'bitseed': 0, 'sdpi': 64}


def test_diversity_selection_has_exact_topk_control_and_removes_redundancy():
    scores = torch.tensor([1., .99, .8, .7])
    descriptors = torch.tensor([[0., 0.], [0., 0.], [10., 0.], [0., 10.]])
    assert diversity_aware_indices(scores, descriptors, keep=2, penalty=0).tolist() == [0, 1]
    selected = diversity_aware_indices(scores, descriptors, keep=2, penalty=.5)
    assert selected.tolist() == [0, 2]
    result = selector_retention_at_indices(selected, torch.tensor([.1, .9, .8, .2]),
                                           ['bitseed'] * 2 + ['sdpi'] * 2)
    assert result['keep'] == 2 and abs(result['selected_best'] - .8) < 1e-6
    assert result['selected_source_counts'] == {'bitseed': 1, 'sdpi': 1}


def test_future_retention_separates_initial_final_and_rescued_candidates():
    scores = torch.arange(256).float()
    initial = torch.zeros(256); final = torch.zeros(256)
    initial[10] = .8
    final[250] = .6
    final[20] = .3
    sources = ['bitseed'] * 192 + ['sdpi'] * 64
    result = future_selector_retention(scores, initial, final, sources)
    assert result['initial']['oracle_index_retained'] == 0
    assert result['final']['oracle_index_retained'] == 1
    assert abs(result['union']['full_pool_oracle'] - .8) < 1e-6
    assert result['rescued_candidates'] == 2
    assert result['rescued_candidates_retained'] == 1
    assert result['rescued_hit_retained'] == 1
    explicit = future_selector_retention_at_indices(
        torch.arange(192, 256), initial, final, sources)
    assert explicit == result


def test_exact_random_selection_matches_enumerated_source_quota():
    import itertools
    dockq = torch.tensor([.9, .4, .3, .1])
    sources = ['bitseed', 'bitseed', 'sdpi', 'sdpi']
    result = exact_random_selection(dockq, sources, {'bitseed': 1, 'sdpi': 1})
    maxima = [max(float(dockq[i]), float(dockq[j]))
              for i, j in itertools.product((0, 1), (2, 3))]
    assert abs(result['expected_best'] - sum(maxima) / len(maxima)) < 1e-7
    assert abs(result['hit_probability_0.23'] - 1.) < 1e-10


def test_selector_training_loader_requires_complete_ordered_pool():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); archives = root / 'archives'; labels = root / 'labels'
        archives.mkdir(); labels.mkdir()
        (archives / 'manifest.json').write_text(json.dumps({'cases': ['1ABC_A_B_LIG']}))
        pool = [{'source': 'bitseed' if i < 192 else 'sdpi'} for i in range(256)]
        torch.save({'split': 'train', 'pool': pool}, archives / '1ABC_A_B_LIG.pt')
        (labels / 'manifest.json').write_text(json.dumps({
            'protocol': 'exact_initial_pool_labels_v1',
            'archive_manifest_sha256': sha256_file(archives / 'manifest.json')}))
        rows = [{'pool_index': i, 'source': pool[i]['source'], 'geometry': [float(i % 5)] * 14,
                 'dockq': {'dockq': float(i) / 255}} for i in range(256)]
        (labels / '1ABC_A_B_LIG.json').write_text(json.dumps({
            'archive_sha256': sha256_file(archives / '1ABC_A_B_LIG.pt'), 'rows': rows}))
        cases, manifests = train_selector.load_cases([archives], [labels], 'train')
    assert len(cases) == len(manifests) == 1
    assert cases[0]['features'].shape == (256, 14)
    assert float(cases[0]['dockq'][-1]) == 1.


if __name__ == '__main__':
    test_all_pool_features_are_aligned_and_checksummed()
    test_retention_tracks_oracle_and_source_budget()
    test_diversity_selection_has_exact_topk_control_and_removes_redundancy()
    test_future_retention_separates_initial_final_and_rescued_candidates()
    test_exact_random_selection_matches_enumerated_source_quota()
    test_selector_training_loader_requires_complete_ordered_pool()
