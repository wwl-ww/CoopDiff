"""Keep split decontamination checks independent of retired GPU orchestration."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_validation_index import prepare_training_index


def test_filter_validation_clusters_without_changing_training(tmp_path):
    config = {}
    inputs = {
        'train_index': {'entries': [{'case': 'AAAA_A_B_L'}],
                        'clusters': [{'items': ['AAAA_A_B_L']}]},
        'val_index': {'entries': [{'case': c} for c in
                                 ['AAAA_X_Y_L', 'BBBB_A_B_L', 'CCCC_A_B_L']],
                      'clusters': [{'items': ['AAAA_X_Y_L', 'BBBB_A_B_L']},
                                   {'items': ['CCCC_A_B_L']}]},
        'protac_index': {'entries': [{'case': 'DDDD_A_B_L'}]},
        'mgd_index': {'entries': [{'case': 'EEEE_A_B_L'}]},
    }
    for key, payload in inputs.items():
        path = tmp_path / (key + '.json')
        path.write_text(json.dumps(payload))
        config[key] = str(path)
    output = tmp_path / 'derived'
    output.mkdir()
    prepared, report = prepare_training_index(config, output)
    assert json.loads(Path(prepared['train_index']).read_text()) == inputs['train_index']
    validation = json.loads(Path(prepared['val_index']).read_text())
    assert [e['case'] for e in validation['entries']] == ['CCCC_A_B_L']
    assert report['removed_cases'] == ['AAAA_X_Y_L', 'BBBB_A_B_L']
    for key, payload in inputs.items():
        assert json.loads(Path(config[key]).read_text()) == payload
    original_outputs = {path.name: path.read_bytes() for path in output.iterdir()}
    try:
        prepare_training_index(config, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError('existing split outputs must not be overwritten')
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original_outputs


def test_filter_without_clusters_excludes_benchmarks_and_rejects_empty(tmp_path):
    config = {}
    for name, cases in (
        ('train', ['AAAA_A_B_L']), ('val', ['aaaa_X_Y_L', 'BBBB_A_B_L', 'CCCC_A_B_L']),
        ('protac', ['BBBB_X_Y_L']), ('mgd', ['DDDD_A_B_L']),
    ):
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps({'entries': [{'case': case} for case in cases]}))
        config[f'{name}_index'] = str(path)
    prepared, _ = prepare_training_index(config, tmp_path / 'derived')
    assert json.loads(Path(prepared['val_index']).read_text())['entries'] == [
        {'case': 'CCCC_A_B_L'}]
    Path(config['val_index']).write_text(json.dumps({'entries': [{'case': 'BBBB_A_B_L'}]}))
    try:
        prepare_training_index(config, tmp_path / 'empty')
    except ValueError as error:
        assert 'no validation samples' in str(error)
    else:
        raise AssertionError('empty validation must not be accepted')
    assert not (tmp_path / 'empty').exists(), 'failed preparation must not write outputs'


if __name__ == '__main__':
    for function in (test_filter_validation_clusters_without_changing_training,
                     test_filter_without_clusters_excludes_benchmarks_and_rejects_empty):
        with tempfile.TemporaryDirectory(prefix='validation-split-test-') as path:
            function(Path(path))
    print('Validation split checks passed')
