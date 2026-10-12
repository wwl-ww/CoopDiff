"""Ranking regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import random
from types import SimpleNamespace

import numpy as np
import torch

from tricoopdiff.ranking.initial_selector import (
    diversity_aware_indices,
    exact_random_selection,
    future_selector_retention,
    future_selector_retention_at_indices,
    initial_pool_features,
    selector_retention,
    selector_retention_at_indices,
)
from tricoopdiff.ranking.ranking_archive import (
    geometry_features,
    probe_state,
    select_archive_candidates,
    state_hash,
)
from tricoopdiff.ranking.ranking_audit import (
    case_metrics,
    join_labels,
    label_support,
    random_hit_probability,
    summarize_cases,
)
from tricoopdiff.ranking.ranking_inference import (
    StateQualityRanker,
    export_quality_checkpoint,
    feature_vector,
    load_quality_checkpoint,
    predict_archive,
    state_quality_loss,
    state_ranking_only_loss,
)


# Initial selector


def initial_selector_archive():
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
    value = initial_selector_archive()
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


# Ranking archive


def test_archive_sampling_tracks_actual_sources_and_probabilities():
    candidates = [{'candidate_id': [s, i]} for s, n in [('bitseed', 192), ('sdpi', 64)] for i in range(n)]
    selected, probability = select_archive_candidates(candidates, 32, 7)
    assert selected == select_archive_candidates(candidates, 32, 7)[0]
    assert len(set(selected)) == 32
    assert sum(candidates[i]['candidate_id'][0] == 'bitseed' for i in selected) == 16
    assert probability == {'bitseed': 16/192, 'sdpi': 16/64}
    fallback = [{'candidate_id': ['sdpi', i]} for i in range(256)]
    ids, probability = select_archive_candidates(fallback, 32, 7)
    assert len(ids) == 32 and probability == {'bitseed': 0., 'sdpi': .125}


def test_geometry_is_rigid_invariant_and_hash_tracks_frame():
    torch.manual_seed(17)
    coords = [torch.randn(n, 3) for n in (8, 9, 4)]
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert np.allclose(geometry_features(*coords),
                       geometry_features(*(c @ rotation + 3 for c in coords)), atol=1e-5)
    state = {'ligand': coords[2], 'poi': coords[1], 'body_frame': torch.eye(3)}
    assert state_hash(state) != state_hash({**state, 'body_frame': rotation})


def test_probe_uses_saved_coordinates_fixed_clock_and_preserves_rng():
    class Model:
        training = False
        expert = SimpleNamespace(model=SimpleNamespace(dynamic_max_cross=False))

        def __call__(self, e3, poi, ligand, **kwargs):
            assert torch.allclose(ligand['coords'], torch.ones(1, 4, 3) * 7)
            assert kwargs['source'] == 1
            assert abs(float(kwargs['t_e3l']) - .05) < 1e-6
            random.random(); np.random.rand(); torch.rand(1)
            return {'rank_features': torch.ones(1, 8)}

    class Flow:
        def t_to_sigma(self, *args):
            return (torch.ones(1),) * 3

        def conditioning(self, t, dt, e3, poi):
            assert float(dt) == 0
            return torch.zeros(1, 9)

    inputs = {'coords': torch.randn(1, 4, 3)}
    state = {'ligand': torch.ones(4, 3) * 7, 'poi': torch.randn(4, 3), 'body_frame': torch.eye(3)}
    before = state_hash(state)
    torch_state, py_state, np_state = torch.random.get_rng_state(), random.getstate(), np.random.get_state()
    output = probe_state(Model(), Flow(), inputs, inputs, inputs, state, 'bitseed')
    assert output['probe_valid'] and len(output['frozen']) == 8
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert py_state == random.getstate() and np.array_equal(np_state[1], np.random.get_state()[1])
    assert before == state_hash(state)


# Ranking inference


def ranking_inference_example_archive():
    features = {'geometry': [1.] * 14, 'probe_valid': True, 'frozen': [2., 3.]}
    return {'case': 'demo', 'probe_feature_dim': 2, 'pool': [{'source': 'sdpi'}, {'source': 'bitseed'}],
            'pairs': [{'pool_index': 0, 'source': 'sdpi', 'valid': False,
                       'initial_features': features, 'final_features': None},
                      {'pool_index': 1, 'source': 'bitseed', 'valid': True,
                       'initial_features': features, 'final_features': features}]}


def test_deploy_features_ignore_truth_metadata_and_zero_failed_probes():
    features = ranking_inference_example_archive()['pairs'][0]['initial_features']
    original = feature_vector(features, 2)
    poisoned = {**features, 'dockq': 1., 'final_dockq': 1., 'source': 'bitseed', 'stage': 'final'}
    assert torch.equal(original, feature_vector(poisoned, 2))
    failed = feature_vector({**features, 'probe_valid': False, 'frozen': None}, 2)
    assert torch.equal(failed[:2], torch.zeros(2)) and failed[-1] == 0


def test_ranking_ties_and_failed_final_keep_initial_without_extra_pool_states():
    archive = ranking_inference_example_archive()
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
    result = predict_archive(ranking_inference_example_archive(), loaded, checkpoint)
    assert all(0 <= r['p49'] <= r['p23'] <= 1 for r in result['rows'])
    try:
        load_quality_checkpoint(checkpoint, {**manifest, 'checkpoint_sha256': 'different'})
    except ValueError:
        pass
    else:
        raise AssertionError('generator mismatch must fail before inference')


def test_label_join_rejects_omitted_state_and_wrong_archive():
    prediction = {'case': 'demo', 'archive_sha256': 'abc', **predict_archive(ranking_inference_example_archive())}
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


# State quality ranker

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


def test_ranking_only_loss_is_temperature_sensitive_and_ignores_class_heads():
    model = StateQualityRanker(3, 4)
    output = model(torch.randn(5, 3))
    labels = torch.tensor([0., .01, .02, .1, .3])
    broad = state_ranking_only_loss(output, labels, .05)
    sharp = state_ranking_only_loss(output, labels, .01)
    assert torch.isfinite(broad) and torch.isfinite(sharp) and broad != sharp
    broad.backward()
    assert model.head.weight.grad[0].abs().sum() > 0
    assert model.head.weight.grad[1:].abs().sum() == 0
