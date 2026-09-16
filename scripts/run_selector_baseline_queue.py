"""Wait for full-pool selector labels, then compute exact random baselines."""

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
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=24.)
    args = parser.parse_args(argv)
    if args.out.exists() or not 0 <= args.wait_hours <= 48:
        parser.error('use a new output directory and wait-hours in [0,48]')
    args.out.mkdir(parents=True)
    status_path = args.out / 'status.json'

    def update(stage, **details):
        payload = {'pid': os.getpid(), 'stage': stage, 'updated_unix': time.time(), **details}
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n'); temporary.replace(status_path)

    try:
        update('waiting_selector')
        deadline = time.monotonic() + args.wait_hours * 3600
        while True:
            upstream = json.loads((args.selector_run / 'status.json').read_text())
            if upstream['stage'] == 'complete':
                break
            if upstream['stage'] == 'failed':
                raise RuntimeError('selector upstream failed')
            if time.monotonic() >= deadline:
                raise TimeoutError('selector baseline wait expired')
            time.sleep(30)
        update('auditing')
        destination = args.out / 'random_baselines'
        command = [sys.executable, str(ROOT / 'scripts/audit_selector_random_baselines.py'),
                   '--labels', str(args.selector_run / 'val8_labels'),
                   '--out', str(destination)]
        code = subprocess.run(command, cwd=ROOT).returncode
        if code:
            raise RuntimeError(f'random baseline audit exited {code}')
        update('complete', report=str(destination / 'summary.md'))
    except Exception as error:
        update('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
