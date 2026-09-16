"""Compute exact random and source-quota baselines for full-pool selector labels."""

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tricoopdiff.initial_selector import POOL_LABEL_VERSION, exact_random_selection


QUOTAS = {
    'uniform_64': {'all': 64},
    'balanced_32_32': {'bitseed': 32, 'sdpi': 32},
    'hybrid_48_16': {'bitseed': 48, 'sdpi': 16},
    'hybrid_56_8': {'bitseed': 56, 'sdpi': 8},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((args.labels / 'manifest.json').read_text())
    if manifest.get('protocol') != POOL_LABEL_VERSION:
        raise ValueError('expected exact full-pool initial labels')
    rows = []
    for path in sorted(args.labels.glob('*.json')):
        if path.name == 'manifest.json' or path.name.startswith('worker_'):
            continue
        payload = json.loads(path.read_text())
        labels = payload['rows']
        if len(labels) != 256:
            raise ValueError(f'{path.stem}: incomplete full pool')
        dockq = [row['dockq']['dockq'] for row in labels]
        sources = [row['source'] for row in labels]
        case = {'case': path.stem}
        for name, quotas in QUOTAS.items():
            if name == 'uniform_64':
                result = exact_random_selection(dockq, ['all'] * 256, quotas)
            else:
                result = exact_random_selection(dockq, sources, quotas)
            case[name] = result
        rows.append(case)
    if not rows:
        raise ValueError('no complete selector labels')
    summary = {'protocol': 'exact_selector_random_baselines_v1', 'cases': len(rows),
               'rows': rows, 'baselines': {}}
    for name in QUOTAS:
        summary['baselines'][name] = {
            'mean_expected_best': statistics.mean(row[name]['expected_best'] for row in rows),
            'mean_expected_oracle_retention': statistics.mean(
                row[name]['expected_oracle_retention'] for row in rows
                if row[name]['expected_oracle_retention'] is not None),
            'expected_hits_0.23': sum(row[name]['hit_probability_0.23'] for row in rows),
        }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Exact selector random baselines', '',
             '| Sampling rule | Expected best DockQ | Oracle retention | Expected ≥0.23 cases |',
             '| --- | ---: | ---: | ---: |']
    for name, values in summary['baselines'].items():
        lines.append(f'| {name} | {values["mean_expected_best"]:.4f} | '
                     f'{values["mean_expected_oracle_retention"]:.4f} | '
                     f'{values["expected_hits_0.23"]:.2f}/{len(rows)} |')
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
