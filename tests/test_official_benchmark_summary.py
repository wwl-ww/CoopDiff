"""Keep failed/missing benchmark cases in success-rate denominators."""

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluation_common import sha256_file
from summarize_official_benchmark import main


def test_summary_accounts_for_missing_cases_and_seeds():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'model.pt'
        checkpoint.write_bytes(b'checkpoint')
        cases = root / 'cases.txt'
        cases.write_text('case1\ncase2\n')
        reports = root / 'reports'
        reports.mkdir()
        report = {
            'case': 'case1',
            'run_config': dict(checkpoint_sha256=sha256_file(checkpoint),
                               trifuse_checkpoint=str(checkpoint), n_walkers=2,
                               steps=20, seed=0, init_mode='sdpi', freeze_poi=False),
            'ensemble': dict(n_walkers=2, n_valid_walkers=2, n_nan_walkers=0,
                             n_exact_dockq_walkers=2, invalid_seeds=[], selection_metric='dockq'),
            'structure': {'dockq_best': .5, 'dockq_mean': .4},
            'walkers': [{'seed': 0, 'dockq': .5}, {'seed': 1, 'dockq': .3}],
        }
        (reports / 'case1.json').write_text(json.dumps(report))
        destination = root / 'summary.json'
        argv = ['summary', '--reports-root', str(reports), '--cases-file', str(cases),
                '--checkpoint', str(checkpoint), '--n-walkers', '2', '--out', str(destination)]

        def run(expected_exit):
            with patch.object(sys, 'argv', argv), patch('builtins.print'):
                try:
                    main()
                except SystemExit as error:
                    assert error.code == expected_exit
                else:
                    raise AssertionError('summary must report completeness via exit status')
            return json.loads(destination.read_text())['summary']

        summary = run(1)
        assert summary['expected_cases'] == 2
        assert summary['case_success_rates_including_missing']['0.23']['rate'] == .5
        report['case'] = 'case2'
        report['walkers'] = [{'seed': 0, 'dockq': .5}, {'seed': 1, 'exact_dockq_error': 'failed'}]
        report['ensemble']['n_exact_dockq_walkers'] = 1
        (reports / 'case2.json').write_text(json.dumps(report))
        summary = run(1)
        assert summary['exact_scored_walkers'] == 3
        assert not summary['complete_all_cases_and_seeds']


if __name__ == '__main__':
    test_summary_accounts_for_missing_cases_and_seeds()
    print('Official benchmark summary test passed')
