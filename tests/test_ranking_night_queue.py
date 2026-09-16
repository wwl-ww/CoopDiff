"""Overnight queue admission: immutable panel reuse and safe upstream handoff."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_ranking_smoke import prepare_extension
from run_ranking_night_queue import upstream_state, validate_panel


def panel_fixture(tmp_path):
    base, run = tmp_path / 'base', tmp_path / 'extension'
    train = [dict(case=f'T{i:03d}_A_B_LIG', group=str(i), split='train', protein1='a', protein2='b', ligand='c')
             for i in range(160)]
    val = [dict(case=f'V{i:03d}_A_B_LIG', group=str(i), split='val', protein1='a', protein2='b', ligand='c')
           for i in range(8)]
    for folder, cases in ((base, train[:32]), (run, train)):
        indexes = folder / 'indexes'
        indexes.mkdir(parents=True)
        audit = {'seed': 7315, 'source_sha256': {}, 'benchmark_pdbs_excluded': [],
                 'train_cases': [e['case'] for e in cases], 'val_cases': [e['case'] for e in val]}
        (indexes / 'audit.json').write_text(json.dumps(audit))
        for split, entries in (('train', cases), ('val', val)):
            for kind in ('inference', 'label_refs'):
                payload = entries if kind == 'inference' else [{**e, 'native_complex_pdb': 'native'} for e in entries]
                (indexes / f'{split}_{kind}.json').write_text(json.dumps({'entries': payload}))
    return base, run


def test_night_extension_is_disjoint_suffix_and_validation_is_immutable(tmp_path):
    base, run = panel_fixture(tmp_path)
    assert prepare_extension(base / 'indexes', run / 'indexes') == 128
    plan = validate_panel(base, run)
    assert plan['new_trajectories'] == 8192 and plan['ranker_fits'] == 18
    path = run / 'indexes/extra_train_inference.json'
    original = path.read_text()
    payload = json.loads(original)
    payload['entries'][0]['case'] = 'T000_A_B_LIG'
    path.write_text(json.dumps(payload))
    try:
        validate_panel(base, run)
    except ValueError:
        pass
    else:
        raise AssertionError('queue accepted a changed suffix')
    path.write_text(original)
    path = run / 'indexes/val_inference.json'
    payload = json.loads(path.read_text())
    payload['entries'][0]['protein1'] = 'different'
    path.write_text(json.dumps(payload))
    try:
        validate_panel(base, run)
    except ValueError:
        pass
    else:
        raise AssertionError('queue accepted a changed validation input')


def test_queue_waits_for_completed_upstream_audit_and_rejects_failure(tmp_path):
    status = tmp_path / 'pipeline_status.json'
    for stage in ('collecting', 'scoring', 'auditing_labels', 'training_rankers', 'complete'):
        status.write_text(json.dumps({'stage': stage}))
        assert not upstream_state(tmp_path), 'must wait for the final upstream comparison too'
    (tmp_path / 'validation_ranking_audit').mkdir()
    (tmp_path / 'validation_ranking_audit/comparison.md').write_text('completed')
    assert upstream_state(tmp_path)
    status.write_text(json.dumps({'stage': 'blocked_incomplete_data'}))
    try:
        upstream_state(tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError('failed upstream must not start dependent GPU work')
