"""Label-blind panel expansion and declared resource budgets."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_ranking_smoke import select_groups
from run_ranking_pipeline import main as run_pipeline, pipeline_budget


def test_group_expansion_preserves_prefix_and_excludes_entire_clusters():
    entries = [{'case': f'{i:04d}_A_B_LIG'} for i in range(12)]
    clusters = [{'items': [e['case']]} for e in entries]
    clusters[0]['items'].append('EXCL_X_Y_LIG')
    payload = {'entries': entries, 'clusters': clusters}
    small = select_groups(payload, 4, {'EXCL'}, 7315)
    large = select_groups(payload, 8, {'EXCL'}, 7315)
    assert small == large[:4], 'expansion must not redraw the earlier panel'
    assert all(group != 0 for group, _ in large), 'excluded cluster members exclude the whole cluster'
    poisoned = {'entries': [{**e, 'dockq': 1.} for e in entries], 'clusters': clusters}
    assert [e['case'] for _, e in select_groups(poisoned, 8, {'EXCL'}, 7315)] == [e['case'] for _, e in large]


def test_budget_uses_actual_indexes_and_rejects_wrong_split_or_native(tmp_path):
    indexes = tmp_path / 'indexes'
    indexes.mkdir()
    audit = {'benchmark_pdbs_excluded': ['TEST']}
    for split, count, prefix in (('train', 3, 'TR'), ('val', 2, 'VA')):
        entries = [dict(case=f'{prefix}{i}_A_B_LIG', group=str(i), split=split,
                        protein1='a', protein2='b', ligand='c') for i in range(count)]
        audit[f'{split}_cases'] = [e['case'] for e in entries]
        (indexes / f'{split}_inference.json').write_text(json.dumps({'entries': entries}))
    (indexes / 'audit.json').write_text(json.dumps(audit))
    budget = pipeline_budget(tmp_path, 64, 256)
    assert budget['train_cases'] == 3 and budget['val_cases'] == 2
    assert budget['total_trajectories'] == 704
    for key, value in (('split', 'train'), ('native_complex_pdb', 'secret')):
        path = indexes / 'val_inference.json'
        payload = json.loads(path.read_text())
        payload['entries'][0][key] = value
        path.write_text(json.dumps(payload))
        try:
            pipeline_budget(tmp_path, 64, 256)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid inference panel accepted')
        payload['entries'][0].pop('native_complex_pdb', None)
        payload['entries'][0]['split'] = 'val'
        path.write_text(json.dumps(payload))


def test_pipeline_gates_training_on_complete_collection_and_label_audit(tmp_path):
    for failing_script in (None, 'collect_ranking_archive.py', 'audit_ranking_label_balance.py'):
        run = tmp_path / (failing_script or 'success')
        indexes = run / 'indexes'
        indexes.mkdir(parents=True)
        audit = {'benchmark_pdbs_excluded': []}
        for split, case in (('train', 'TR00_A_B_LIG'), ('val', 'VA00_A_B_LIG')):
            entry = dict(case=case, split=split, group=split, protein1='a', protein2='b', ligand='c')
            (indexes / f'{split}_inference.json').write_text(json.dumps({'entries': [entry]}))
            audit[f'{split}_cases'] = [case]
        (indexes / 'audit.json').write_text(json.dumps(audit))
        calls = []

        def execute(command, **kwargs):
            name = Path(command[2]).name
            calls.append((name, kwargs['env']['CUDA_VISIBLE_DEVICES']))
            return SimpleNamespace(returncode=int(name == failing_script))

        argv = ['pipeline', '--run-dir', str(run), '--runtime', str(tmp_path),
                '--project-root', str(tmp_path), '--gpus', '5', '7', '--train-walkers', '64']
        with patch('sys.argv', argv), patch('run_ranking_pipeline.subprocess.run', side_effect=execute):
            try:
                result = run_pipeline()
            except SystemExit as error:
                result = error.code
        status = json.loads((run / 'pipeline_status.json').read_text())
        trained = [gpu for name, gpu in calls if name == 'train_state_ranker.py']
        assert bool(result) == bool(failing_script)
        if failing_script:
            assert not trained, 'failed collection/audit must prevent either ranker from fitting'
            assert status['stage'].startswith('blocked_')
        else:
            assert sorted(trained) == ['5', '7'] and status['stage'] == 'complete'
        assert all(gpu == '' for name, gpu in calls if name in (
            'score_ranking_archive.py', 'audit_ranking_label_balance.py'))


def test_reference_control_reports_bad_or_nonfinite_cases_without_dropping_them(tmp_path):
    import audit_ranking_references as control

    references = tmp_path / 'references.json'
    references.write_text(json.dumps({'entries': [dict(case=str(i), protein1='a', protein2='b',
        ligand='c', native_complex_pdb='native') for i in range(3)]}))
    output = tmp_path / 'report.json'
    argv = ['control', '--label-refs', str(references), '--out', str(output), '--expect-native-frame']
    values = [dict(dockq=q, fnat=1., irms=0., lrms=0.) for q in (1., .5, float('nan'))]
    with patch('sys.argv', argv), patch.object(control, 'evaluation_input_signature', return_value={}), \
            patch.object(control, '_ligand_records', return_value=(None, None, [])), \
            patch.object(control, '_featurized_ca_coords', return_value=[]), \
            patch.object(control, 'compute_fullatom_dockq', side_effect=values):
        assert control.main() == 1
    report = json.loads(output.read_text())
    assert report['cases'] == 3 and report['passed'] == 1
    assert 'nonfinite' in report['rows'][2]['error']
