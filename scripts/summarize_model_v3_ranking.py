"""Create an all-or-nothing summary of the predeclared model-v3 ranking studies."""

import argparse
import json
from pathlib import Path
import statistics


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def load(path):
    return json.loads(path.read_text())


def summarize(inputs):
    """Join completed reports without selecting a preferred seed or penalty."""
    current = load(inputs['current'] / 'summary.json')['runs']
    random = load(inputs['random'] / 'random_baselines/summary.json')
    future = load(inputs['future'] / 'summary.json')['runs']
    expanded = load(inputs['expanded'] / 'summary.json')['runs']
    current_diversity = load(inputs['current_diversity'] / 'audit/report.json')['runs']
    future_diversity = load(inputs['future_diversity'] / 'audit/report.json')['runs']
    old = load(inputs['old'] / 'summary.json')['runs']

    current_table = []
    for size in sorted({row['size'] for row in current}):
        rows = [row for row in current if row['size'] == size]
        current_table.append({'method': f'learned_top64_n{size}',
                              'mean_selected_best': mean(rows, 'mean_selected_best'),
                              'mean_oracle_retention': mean(rows, 'mean_case_oracle_retention'),
                              'hit_range': [min(r['selected_hits_0.23'] for r in rows),
                                            max(r['selected_hits_0.23'] for r in rows)]})
    for name, row in random['baselines'].items():
        current_table.append({'method': name,
                              'mean_selected_best': row['mean_expected_best'],
                              'mean_oracle_retention': row['mean_expected_oracle_retention'],
                              'expected_hits_0.23': row['expected_hits_0.23']})

    current_diversity_table = []
    groups = {}
    for run in current_diversity:
        size = int(run['run'].split('_')[0][1:])
        for row in run['summary']:
            groups.setdefault((size, row['penalty']), []).append(row)
    for (size, penalty), rows in sorted(groups.items()):
        current_diversity_table.append({
            'train_cases': size, 'penalty': penalty,
            'mean_selected_best': mean(rows, 'mean_selected_best'),
            'mean_oracle_retention': mean(rows, 'mean_case_oracle_retention'),
            'hit_range': [min(r['selected_hits_0.23'] for r in rows),
                          max(r['selected_hits_0.23'] for r in rows)],
        })

    future_table = []
    for target in ('final', 'union'):
        rows = [row for row in future if row['target'] == target]
        future_table.append({
            'target': target,
            'final_selected_best': mean([row['final'] for row in rows],
                                        'mean_selected_best'),
            'final_retention': mean([row['final'] for row in rows],
                                    'mean_case_oracle_retention'),
            'union_selected_best': mean([row['union'] for row in rows],
                                        'mean_selected_best'),
            'union_retention': mean([row['union'] for row in rows],
                                    'mean_case_oracle_retention'),
            'rescued_range': [min(row['rescued_cases_retained'] for row in rows),
                              max(row['rescued_cases_retained'] for row in rows)],
            'rescued_available': rows[0]['rescued_cases_available'],
        })

    future_diversity_table = []
    groups = {}
    for run in future_diversity:
        for row in run['summary']:
            groups.setdefault((run['target'], row['penalty']), []).append(row)
    for (target, penalty), rows in sorted(groups.items()):
        future_diversity_table.append({
            'target': target, 'penalty': penalty,
            'final_selected_best': mean([row['final'] for row in rows],
                                        'mean_selected_best'),
            'final_retention': mean([row['final'] for row in rows],
                                    'mean_case_oracle_retention'),
            'union_selected_best': mean([row['union'] for row in rows],
                                        'mean_selected_best'),
            'union_retention': mean([row['union'] for row in rows],
                                    'mean_case_oracle_retention'),
            'rescued_range': [min(row['rescued_cases_retained'] for row in rows),
                              max(row['rescued_cases_retained'] for row in rows)],
        })

    postranker_table = []
    for mode in ('geometry', 'frozen_geometry'):
        rows = [row for row in old
                if row['features'] == mode and row['train_cases'] == 160]
        postranker_table.append({'features': mode, **{
            f'{stage}_top1': mean(
                [row['metrics'][f'all/{stage}'] for row in rows], 'mean_top1')
            for stage in ('initial', 'final', 'union')}})
    postranker_table.append({'features': 'expanded_geometry', **{
        f'{stage}_top1': mean([row[stage] for row in expanded], 'mean_top1')
        for stage in ('initial', 'final', 'union')}})
    return {
        'scope': ('internal validation n=8; validation reused for epoch selection and '
                  'penalty diagnostics; all seeds and penalties retained'),
        'current_quality_selection': current_table,
        'current_quality_diversity': current_diversity_table,
        'future_value_selection': future_table,
        'future_value_diversity': future_diversity_table,
        'postranking_top1': postranker_table,
    }


def markdown(payload):
    lines = ['# Model-v3 ranking decision table', '', payload['scope'], '',
             'Selected-best tables are oracle-within-the-selected-64 allocation metrics; '
             'post-ranker values are deployable top-1 metrics. They are not interchangeable.', '',
             '## Current-quality 256→64 allocation', '',
             '| Method | Selected-best | Retention | ≥0.23 |', '| --- | ---: | ---: | ---: |']
    for row in payload['current_quality_selection']:
        hits = (f"{row['hit_range'][0]}–{row['hit_range'][1]}/8" if 'hit_range' in row
                else f"{row['expected_hits_0.23']:.2f}/8 expected")
        lines.append(f"| {row['method']} | {row['mean_selected_best']:.4f} | "
                     f"{row['mean_oracle_retention']:.4f} | {hits} |")
    lines += ['', '## Current-quality diversity diagnostic', '',
              '| Train cases | Penalty | Selected-best | Retention | ≥0.23 |',
              '| ---: | ---: | ---: | ---: | ---: |']
    for row in payload['current_quality_diversity']:
        lines.append(f"| {row['train_cases']} | {row['penalty']:.3f} | "
                     f"{row['mean_selected_best']:.4f} | {row['mean_oracle_retention']:.4f} | "
                     f"{row['hit_range'][0]}–{row['hit_range'][1]}/8 |")
    lines += ['', '## Future-value 256→64 allocation', '',
              '| Target | Final best/retention | Union best/retention | Rescued cases |',
              '| --- | --- | --- | ---: |']
    for row in payload['future_value_selection']:
        lines.append(f"| {row['target']} | {row['final_selected_best']:.4f} / "
                     f"{row['final_retention']:.4f} | {row['union_selected_best']:.4f} / "
                     f"{row['union_retention']:.4f} | {row['rescued_range'][0]}–"
                     f"{row['rescued_range'][1]}/{row['rescued_available']} |")
    lines += ['', '## Future-value diversity diagnostic', '',
              '| Target | Penalty | Final best/retention | Union best/retention | Rescued range |',
              '| --- | ---: | --- | --- | ---: |']
    for row in payload['future_value_diversity']:
        lines.append(f"| {row['target']} | {row['penalty']:.3f} | "
                     f"{row['final_selected_best']:.4f} / {row['final_retention']:.4f} | "
                     f"{row['union_selected_best']:.4f} / {row['union_retention']:.4f} | "
                     f"{row['rescued_range'][0]}–{row['rescued_range'][1]}/8 |")
    lines += ['', '## Post-ranker top-1', '',
              '| Features | Initial | Final | Union |', '| --- | ---: | ---: | ---: |']
    for row in payload['postranking_top1']:
        lines.append(f"| {row['features']} | {row['initial_top1']:.4f} | "
                     f"{row['final_top1']:.4f} | {row['union_top1']:.4f} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('current', 'random', 'future', 'expanded', 'current_diversity',
                 'future_diversity', 'old'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('output must not exist')
    inputs = {name: getattr(args, name) for name in (
        'current', 'random', 'future', 'expanded', 'current_diversity',
        'future_diversity', 'old')}
    payload = summarize(inputs)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    (args.out / 'summary.md').write_text(markdown(payload))


if __name__ == '__main__':
    main()
