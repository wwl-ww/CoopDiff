import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from audit_conditioned_protac_results import RUN_NAMES, audit, render


def _run(path, shift):
    path.mkdir()
    case = 'fixture'
    summary = {'requested_cases': 1, 'completed_cases': [case], 'failures': []}
    (path / 'summary.json').write_text(json.dumps(summary))
    values = {'initial': .2, 'refined': .3 + shift, 'safe_union': .3 + shift}
    report = {
        'status': 'complete', 'proposal_pool_sha256': 'abc',
        'arms': {'hybrid_56_8': {
            'selected_candidate_ids': [['bitseed', 0]],
            **{stage: {'best': value} for stage, value in values.items()}}},
        'refinements': [{
            'candidate_id': ['bitseed', 0], 'initial_state_sha256': 'state',
            'diffusion_seed': 17, 'initial_dockq': .2,
        }],
    }
    (path / f'{case}.json').write_text(json.dumps(report))


def test_conditioned_protac_audit_requires_paired_identity(tmp_path):
    paths = {name: tmp_path / name for name in RUN_NAMES}
    for name, shift in zip(RUN_NAMES, (.1, .0, -.1)):
        _run(paths[name], shift)
    summary = audit(paths)
    assert summary['leaderboards']['default_pocket']['refined']['mean'] == .4
    delta = summary['paired_comparisons']['pocket_input_effect']['refined']
    assert delta['mean'] > 0 and delta['wins'] == 1
    assert 'not post-ranker top1' in render(summary)
    report = json.loads((paths['control_absent'] / 'fixture.json').read_text())
    report['proposal_pool_sha256'] = 'different'
    (paths['control_absent'] / 'fixture.json').write_text(json.dumps(report))
    try:
        audit(paths)
    except ValueError as error:
        assert 'proposal pools differ' in str(error)
    else:
        raise AssertionError('unpaired proposal pool accepted')


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='conditioned-protac-audit-') as directory:
        test_conditioned_protac_audit_requires_paired_identity(Path(directory))
