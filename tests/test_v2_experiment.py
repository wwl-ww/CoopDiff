"""No GPU or model download: v2 workflow contracts and split decontamination."""

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_overnight import V2_STAGES, validate_plan
from v2_experiment import prepare_training_index


def test_v2_manifest_is_complete():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'scripts/overnight_v2.json').read_text())
    assert validate_plan(manifest) == 36000
    assert [s['name'] for s in manifest['stages']] == list(V2_STAGES)
    for stage in manifest['stages']:
        assert stage['name'] in stage['argv']
        assert 'v2_experiment.py' in stage['argv'][1]


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


if __name__ == '__main__':
    test_v2_manifest_is_complete()
    with tempfile.TemporaryDirectory(prefix='v2-split-test-') as path:
        test_filter_validation_clusters_without_changing_training(Path(path))
    print('V2 workflow checks passed')
