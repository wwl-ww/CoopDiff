"""Summarize diagnostic loss scales by case, time and torsion count."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median


def summarize(rows):
    result = {'count': len(rows), 'mean_loss': mean(r['loss'] for r in rows),
              'components': {k: mean(r['components'][k] for r in rows)
                             for k in rows[0]['components']}}
    updates = [r for r in rows if 'preclip_norm' in r]
    if updates:
        result.update(
            clipped_fraction=mean(r['preclip_norm'] > 1 for r in updates),
            median_preclip_norm=median(r['preclip_norm'] for r in updates),
            median_clip_factor=median(r['clip_factor'] for r in updates),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / 'steps.jsonl').read_text().splitlines()]
    report = {'overall': summarize(rows)}
    for label, key in (
        ('case', lambda r: r.get('case', 'fixed')),
        ('time_bin', lambda r: min(int(r.get('time', .5) * 5), 4)),
        ('torsion_count', lambda r: r.get('n_torsions', 'fixed')),
    ):
        groups = defaultdict(list)
        for row in rows:
            groups[str(key(row))].append(row)
        report[label] = {name: summarize(group) for name, group in groups.items()}
    summary_path = args.run / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if 'initial_validation' in summary:
            report['heldout_path_mean_loss'] = {
                phase: mean(r['loss'] for r in summary[phase + '_validation'])
                for phase in ('initial', 'final')}
        if 'condition_ablation' in summary:
            report['condition_ablation'] = summary['condition_ablation']
    destination = args.run / 'breakdown.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report.get('heldout_path_mean_loss', report['overall']), indent=2))
    print(destination)


if __name__ == '__main__':
    main()
