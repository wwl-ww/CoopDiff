import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_conditioned_checkpoint_sweep import audit_pair


def write_run(root, name, offsets):
    path = root / name
    path.mkdir()
    cases = ['A', 'B']
    (path / 'summary.json').write_text(json.dumps({
        'requested_cases': 2, 'completed_cases': cases, 'failures': []}))
    for case_index, case in enumerate(cases):
        refinements = []
        for index in range(2):
            initial = .1 + .1 * index + .01 * case_index
            refined = initial + offsets[index]
            refinements.append({
                'candidate_id': ['bitseed', index], 'initial_state_sha256': f'{case}-{index}',
                'diffusion_seed': 100 + index, 'initial_dockq': initial,
            })
        stages = {
            'initial': {'best': max(row['initial_dockq'] for row in refinements)},
            'refined': {'best': max(row['initial_dockq'] + offsets[index]
                                    for index, row in enumerate(refinements))},
            'safe_union': {'best': max(
                row['initial_dockq'] + max(0., offsets[index])
                for index, row in enumerate(refinements))},
        }
        (path / f'{case}.json').write_text(json.dumps({
            'status': 'complete', 'proposal_pool_sha256': f'pool-{case}',
            'refinements': refinements,
            'arms': {'hybrid_56_8': {
                'selected_candidate_ids': [['bitseed', 0], ['bitseed', 1]],
                **stages}},
        }))
    return path


def test_checkpoint_sweep_is_strictly_paired(tmp_path):
    pocket = write_run(tmp_path, 'pocket', (.03, -.01))
    absent = write_run(tmp_path, 'absent', (.01, .02))
    result = audit_pair(pocket, absent, 'hybrid_56_8')
    assert result['identity_checks']['initial_state_seed_score_equal']
    delta = result['pocket_minus_absent']['refined']
    assert delta['count'] == 2 and delta['losses'] == 2
    assert abs(delta['mean'] + .03) < 1e-12


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='conditioned-sweep-test-') as directory:
        test_checkpoint_sweep_is_strictly_paired(Path(directory))
