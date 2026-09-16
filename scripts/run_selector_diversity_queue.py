"""Wait for the selector grid, then run the predeclared diversity audit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selector-run', type=Path, required=True)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=24.)
    args = parser.parse_args(argv)
    selector, base, out = (path.resolve() for path in
                           (args.selector_run, args.base_run, args.out))
    if out.exists() or not 0 <= args.wait_hours <= 48:
        parser.error('use a new output and wait-hours in [0,48]')
    out.mkdir(parents=True, exist_ok=False)
    status = {'pid': os.getpid(), 'stage': 'waiting_selector',
              'started_unix': time.time(), 'updated_unix': time.time()}

    def update(stage, **extra):
        status.update(stage=stage, updated_unix=time.time(), **extra)
        temporary = out / 'status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(out / 'status.json')

    try:
        update('waiting_selector')
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            upstream = json.loads((selector / 'status.json').read_text())
            if upstream.get('stage') == 'complete':
                break
            if upstream.get('stage') == 'failed':
                raise RuntimeError('selector upstream failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('selector upstream missed diversity-audit deadline')
            time.sleep(30)
        update('auditing')
        command = [
            sys.executable, '-u', str(ROOT / 'scripts/audit_initial_selector_diversity.py'),
            '--rankers', str(selector / 'rankers'),
            '--val-archives', str(base / 'val_archives'),
            '--val-labels', str(selector / 'val8_labels'),
            '--out', str(out / 'audit'),
        ]
        with (out / 'audit.log').open('x') as handle:
            code = subprocess.run(command, cwd=ROOT, stdout=handle,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f'diversity audit exited {code}')
        update('complete')
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
