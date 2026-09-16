"""Wait for every model-v3 ranking arm and emit one all-or-nothing report."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('current', 'random', 'future', 'expanded', 'current_diversity',
         'future_diversity', 'objective')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in NAMES:
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=48.)
    args = parser.parse_args(argv)
    inputs = {name: getattr(args, name).resolve() for name in NAMES}
    old, out = args.old.resolve(), args.out.resolve()
    if out.exists() or not 0 <= args.wait_hours <= 72:
        parser.error('use a new output and wait-hours in [0,72]')
    out.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'stage': 'waiting',
              'started_unix': time.time(), 'updated_unix': time.time()}

    def update(stage, **details):
        status.update(stage=stage, updated_unix=time.time(), **details)
        temporary = out / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    try:
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            states = {name: json.loads((root / 'status.json').read_text()).get('stage')
                      for name, root in inputs.items()}
            update('waiting', upstream=states)
            failures = {name: stage for name, stage in states.items()
                        if stage == 'failed'}
            if failures:
                raise RuntimeError(f'upstream failure: {failures}')
            if all(stage == 'complete' for stage in states.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'ranking summary deadline exceeded: {states}')
            time.sleep(30)
        update('summarizing')
        command = [sys.executable, '-u', str(ROOT / 'scripts/summarize_model_v3_ranking.py')]
        for name, root in inputs.items():
            command += ['--' + name.replace('_', '-'), str(root)]
        command += ['--old', str(old), '--out', str(out / 'report')]
        with (out / 'summary.log').open('x') as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f'ranking summarizer exited {code}')
        update('complete', report=str(out / 'report/summary.md'))
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
