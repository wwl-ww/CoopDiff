"""Case-paired ranking comparisons keep labels and candidate sets fixed."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.ranking.ranking_audit import paired_case_comparison


def report(values, label='label'):
    cases = {}
    files = {}
    for index, value in enumerate(values):
        case = f'case{index}'
        cases[case] = {'count': 2, 'top1': value, 'oracle': .8}
        files[case] = {'label': label, 'prediction': f'prediction{index}'}
    return {'labels_manifest_sha256': 'manifest', 'files_sha256': files,
            'cases': {'all/union': cases}}


def test_paired_comparison_averages_seeds_before_resampling_cases():
    left = [report([.6, .2]), report([.4, .4])]
    right = [report([.2, .2]), report([.2, .2])]
    result = paired_case_comparison(left, right, 'all/union', repeats=100, seed=7)
    assert abs(result['mean_delta_top1'] - .2) < 1e-12
    assert result['case_wins'] == 2 and result['case_losses'] == result['case_ties'] == 0
    assert result['seeds_per_method'] == 2 and result['cases'] == 2
    assert result['mean_seed_hit_count_delta']['0.23'] == 1.5


def test_paired_comparison_rejects_label_or_candidate_drift():
    original = report([.4, .3])
    for changed in (report([.4, .3], label='other'), report([.4, .3])):
        if changed['files_sha256']['case0']['label'] == 'label':
            changed['cases']['all/union']['case0']['oracle'] = .7
        try:
            paired_case_comparison([original], [changed], 'all/union')
        except ValueError:
            pass
        else:
            raise AssertionError('paired audit accepted label/candidate drift')
