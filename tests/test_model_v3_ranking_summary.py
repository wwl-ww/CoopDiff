"""Schema test for the all-or-nothing ranking summary."""

from pathlib import Path
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    'summarize_model_v3_ranking', ROOT / 'scripts/summarize_model_v3_ranking.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_ranking_summary_keeps_all_metric_types_separate(tmp_path=None):
    import tempfile
    context = tempfile.TemporaryDirectory() if tmp_path is None else None
    root = Path(context.name) if context is not None else tmp_path
    inputs = {name: root / name for name in (
        'current', 'random', 'future', 'expanded', 'current_diversity',
        'future_diversity', 'objective', 'old')}
    base = {'mean_selected_best': .2, 'mean_case_oracle_retention': .8,
            'selected_hits_0.23': 3}
    write(inputs['current'] / 'summary.json', {'runs': [
        {'size': 160, **base}]})
    write(inputs['random'] / 'random_baselines/summary.json', {'baselines': {
        'uniform_64': {'mean_expected_best': .1,
                       'mean_expected_oracle_retention': .4,
                       'expected_hits_0.23': 1.5}}})
    future_metric = {'mean_selected_best': .2,
                     'mean_case_oracle_retention': .8}
    write(inputs['future'] / 'summary.json', {'runs': [
        {'target': target, 'final': future_metric, 'union': future_metric,
         'rescued_cases_retained': 2, 'rescued_cases_available': 4}
        for target in ('final', 'union')]})
    leaderboard = {stage: {'mean_top1': .05} for stage in
                   ('initial', 'final', 'union')}
    write(inputs['expanded'] / 'summary.json', {'runs': [leaderboard]})
    diversity_row = {'penalty': 0., **base}
    write(inputs['current_diversity'] / 'audit/report.json', {'runs': [
        {'run': 'n160_s7315', 'summary': [diversity_row]}]})
    future_diversity_row = {'penalty': 0., 'final': future_metric,
                            'union': future_metric, 'rescued_cases_retained': 2}
    write(inputs['future_diversity'] / 'audit/report.json', {'runs': [
        {'target': target, 'summary': [future_diversity_row]}
        for target in ('final', 'union')]})
    write(inputs['objective'] / 'summary.json', {'runs': [
        {'temperature': .05, **leaderboard},
        {'temperature': .01, **leaderboard}]})
    old_runs = []
    for features in ('geometry', 'frozen_geometry'):
        old_runs.append({'features': features, 'train_cases': 160,
                         'metrics': {f'all/{stage}': {'mean_top1': .04}
                                     for stage in ('initial', 'final', 'union')}})
    write(inputs['old'] / 'summary.json', {'runs': old_runs})
    payload = summary.summarize(inputs)
    text = summary.markdown(payload)
    assert payload['current_quality_selection'][0]['mean_selected_best'] == .2
    assert payload['postranking_top1'][-1]['union_top1'] == .05
    assert len(payload['postranking_top1']) == 5
    assert 'oracle-within-the-selected-64' in text and 'Post-ranker top-1' in text
    if context is not None:
        context.cleanup()


if __name__ == '__main__':
    test_ranking_summary_keeps_all_metric_types_separate()
