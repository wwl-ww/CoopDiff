"""Run all pytest domains in isolated, offline CPU subprocesses."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHILD = r'''
import json
import sys

import pytest


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' and report.passed:
            self.passed += 1
        elif report.failed:
            self.failed.append(report.nodeid + '\n' + str(report.longrepr))

    def pytest_collectreport(self, report):
        if report.failed:
            self.failed.append(report.nodeid + '\n' + str(report.longrepr))


results = Results()
status = pytest.main([sys.argv[1], '-q', '--tb=short'], plugins=[results])
print('CPU_TEST_RESULT=' + json.dumps({'passed': results.passed, 'failed': results.failed}))
sys.exit(status)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=90, help='Seconds allowed per test file')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                       PYTHONOPTIMIZE='0')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        environment[key] = '2'
    environment.setdefault('TRICOOPDIFF_CACHE_DIR', str(ROOT / 'results/runtime_cache/diffdock'))
    cache = Path(environment['TRICOOPDIFF_CACHE_DIR'])
    required = ('.p.npy', '.score.npy', '.so3_omegas_array4.npy', '.so3_cdf_vals4.npy',
                '.so3_score_norms4.npy', '.so3_exp_score_norms4.npy')
    if any(not (cache / name).is_file() for name in required):
        parser.error(f'provide existing DiffDock numerical caches at {cache}; no auto-generation')
    passed = failures = 0
    files = sorted((ROOT / 'tests').glob('test_*.py'))
    if not files:
        parser.error('no test files found')
    for path in files:
        try:
            result = subprocess.run([sys.executable, '-c', CHILD, str(path)], cwd=ROOT,
                                    env=environment, capture_output=True, text=True,
                                    timeout=args.timeout)
            records = [line.removeprefix('CPU_TEST_RESULT=') for line in result.stdout.splitlines()
                       if line.startswith('CPU_TEST_RESULT=')]
            if not records:
                raise RuntimeError(result.stdout + result.stderr)
            report = json.loads(records[-1])
            passed += report['passed']
            failures += len(report['failed']) or int(result.returncode != 0)
            print(f"{path.name}: {report['passed']} passed, {len(report['failed'])} failed", flush=True)
            for error in report['failed']:
                print(error, flush=True)
            if result.returncode and not report['failed']:
                print(result.stderr, flush=True)
        except (subprocess.TimeoutExpired, RuntimeError, ValueError) as error:
            failures += 1
            print(f'{path.name}: RUNNER ERROR: {error}', flush=True)
    print(f'TOTAL: {len(files)} files, {passed} passed, {failures} failures', flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
