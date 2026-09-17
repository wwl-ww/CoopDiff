import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_conditioned_fullpool import audit


def write_run(root, name, offsets):
    path = root / name
    path.mkdir()
    cases = ['A', 'B']
    summary = {'requested_cases': 2, 'completed_cases': cases, 'failures': [],
               'leaderboards': {'hybrid_56_8': {}}}
    (path / 'summary.json').write_text(json.dumps(summary))
    for case_index, case in enumerate(cases):
        refinements = []
        for index, source in enumerate(('bitseed', 'sdpi')):
            initial = .1 * (index + 1) + .01 * case_index
            refined = initial + offsets[index]
            refinements.append({
                'candidate_id': [source, index], 'pool_index': index,
                'source': source, 'initial_state_sha256': f'{case}-{index}',
                'diffusion_seed': 100 + index, 'initial_dockq': initial,
                'refined_dockq': refined, 'safe_dockq': max(initial, refined),
                'valid': True,
            })
        best = lambda key: max(row[key] for row in refinements)
        stages = {'initial': {'best': best('initial_dockq')},
                  'refined': {'best': best('refined_dockq')},
                  'safe_union': {'best': best('safe_dockq')}}
        report = {
            'status': 'complete', 'proposal_pool_sha256': f'pool-{case}',
            'union_evaluated_count': 2, 'refinements': refinements,
            'arms': {'hybrid_56_8': stages},
        }
        (path / f'{case}.json').write_text(json.dumps(report))
    return path


def test_fullpool_audit_is_candidate_paired(tmp_path):
    pocket = write_run(tmp_path, 'pocket', (.03, -.02))
    absent = write_run(tmp_path, 'absent', (.01, .01))
    control = write_run(tmp_path, 'control', (-.01, .02))
    parent = write_run(tmp_path, 'parent', (0., 0.))
    result = audit({
        'pocket': pocket, 'absent': absent, 'control': control,
        'parent': parent,
    }, 2)
    assert result['identity_checks']['candidate_initial_state_seed_score_equal']
    effect = result['comparisons']['pocket_minus_absent']['candidate']['refined']
    assert effect['count'] == 4 and effect['wins'] == effect['losses'] == 2
    assert abs(effect['mean'] + .005) < 1e-12
    control_effect = result['comparisons']['absent_minus_control'][
        'candidate']['refined']
    assert control_effect['wins'] == control_effect['losses'] == 2
    assert result['comparisons']['absent_minus_parent']['candidate']['refined']['wins'] == 4


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='fullpool-audit-test-') as directory:
        test_fullpool_audit_is_candidate_paired(Path(directory))
