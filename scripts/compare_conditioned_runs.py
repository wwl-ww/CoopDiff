"""Compare paired PROTAC-conditioned training runs without structure labels.

This audit is intentionally limited to the four validation-loss cells emitted
by ``train_conditioned_entry.py``.  It does not claim a DockQ improvement;
structure-level evaluation remains a separate experiment.
"""

import argparse
import json
from pathlib import Path
import time


VALIDATION_CELLS = (
    'sdpi/full',
    'sdpi/absent',
    'bitseed_endpoint/full',
    'bitseed_endpoint/absent',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditioned-run', required=True)
    parser.add_argument('--absent-control-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--wait-hours', type=float, default=0.)
    parser.add_argument('--poll-seconds', type=float, default=60.)
    args = parser.parse_args(argv)
    if args.wait_hours < 0 or args.poll_seconds <= 0:
        parser.error('--wait-hours must be non-negative and --poll-seconds positive')
    return args


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _validations(run):
    rows = {}
    with (Path(run) / 'metrics.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('event') != 'validation':
                continue
            cells = row.get('by_cell', {})
            if set(cells) != set(VALIDATION_CELLS):
                raise RuntimeError(
                    f'incomplete validation grid at epoch {row.get("epoch")}: '
                    f'{sorted(cells)}')
            rows[int(row['epoch'])] = row
    return rows


def _assert_paired_configs(conditioned, control):
    left = _read_json(Path(conditioned) / 'config.json')
    right = _read_json(Path(control) / 'config.json')
    # Separate cache directories may contain the same deterministic endpoint
    # protocol; their contents are independently audited by the trainers.
    allowed = {
        'output_dir', 'condition_dropout', 'bitseed_lazy_cache_dir',
        # Strict paired continuation deliberately uses distinct epoch-1
        # checkpoints and makes only the control arm a cache follower.
        'resume', 'paired_cache_follower',
        'condition_drop_all', 'condition_drop_side', 'condition_drop_pocket',
        'condition_drop_ligand_role', 'condition_drop_binary',
    }
    mismatches = {
        key: [left.get(key), right.get(key)]
        for key in sorted(set(left) | set(right))
        if key not in allowed and left.get(key) != right.get(key)
    }
    if mismatches:
        raise RuntimeError(f'paired configuration mismatch: {mismatches}')
    def drop_all(config):
        values = config['condition_dropout']
        return float(values.get('drop_all_probability', values.get('drop_all')))

    if drop_all(left) >= 1.:
        raise RuntimeError('conditioned run unexpectedly drops all conditions')
    if drop_all(right) != 1.:
        raise RuntimeError('absent control must use drop_all=1')
    return left, right


def compare(conditioned, control, require_complete=True):
    conditioned, control = Path(conditioned), Path(control)
    config, _ = _assert_paired_configs(conditioned, control)
    left, right = _validations(conditioned), _validations(control)
    epochs = sorted(set(left) & set(right))
    if not epochs:
        raise RuntimeError('paired runs have no common validation epoch')
    target = int(config['epochs'])
    if require_complete and (target not in left or target not in right):
        raise RuntimeError(f'paired runs are incomplete: target epoch={target}')
    rows = []
    for epoch in epochs:
        conditioned_row, control_row = left[epoch], right[epoch]
        cell_rows = {}
        for cell in VALIDATION_CELLS:
            full_loss = float(conditioned_row['by_cell'][cell])
            control_loss = float(control_row['by_cell'][cell])
            cell_rows[cell] = {
                'conditioned_loss': full_loss,
                'absent_control_loss': control_loss,
                'conditioned_minus_control': full_loss - control_loss,
            }
        rows.append({
            'epoch': epoch,
            'conditioned_selection_loss': float(conditioned_row['selection_loss']),
            'absent_control_selection_loss': float(control_row['selection_loss']),
            'selection_loss_delta': (float(conditioned_row['selection_loss'])
                                     - float(control_row['selection_loss'])),
            'raw_condition_scale': float(conditioned_row['raw_condition_scale']),
            'ema_condition_scale': float(conditioned_row['ema_condition_scale']),
            'cells': cell_rows,
            'within_conditioned_full_minus_absent': {
                source: (float(conditioned_row['by_cell'][f'{source}/full'])
                         - float(conditioned_row['by_cell'][f'{source}/absent']))
                for source in ('sdpi', 'bitseed_endpoint')
            },
        })
    best_conditioned = min(rows, key=lambda row: row['conditioned_selection_loss'])
    best_control = min(rows, key=lambda row: row['absent_control_selection_loss'])
    return {
        'schema': 'conditioned_training_comparison_v1',
        'interpretation': (
            'Negative loss deltas favor explicit conditions. These are paired '
            'validation losses, not DockQ or structure-quality results.'),
        'conditioned_run': str(conditioned.resolve()),
        'absent_control_run': str(control.resolve()),
        'parent_checkpoint_sha256': config['parent_checkpoint_sha256'],
        'paired_epochs': epochs,
        'target_epochs': target,
        'complete': target in left and target in right,
        'best_conditioned': best_conditioned,
        'best_absent_control': best_control,
        'epochs': rows,
    }


def render_markdown(summary):
    conditioned = summary['best_conditioned']
    control = summary['best_absent_control']
    final = summary['epochs'][-1]
    lines = [
        '# Conditioned Training Comparison',
        '',
        ('This report compares paired validation losses only. It does **not** '
         'establish a DockQ or structure-quality gain.'),
        '',
        '| View | Epoch | Conditioned loss | Absent-control loss | Delta |',
        '| --- | ---: | ---: | ---: | ---: |',
        (f'| Best conditioned checkpoint | {conditioned["epoch"]} | '
         f'{conditioned["conditioned_selection_loss"]:.6f} | '
         f'{conditioned["absent_control_selection_loss"]:.6f} | '
         f'{conditioned["selection_loss_delta"]:+.6f} |'),
        (f'| Best absent-control checkpoint | {control["epoch"]} | '
         f'{control["conditioned_selection_loss"]:.6f} | '
         f'{control["absent_control_selection_loss"]:.6f} | '
         f'{control["selection_loss_delta"]:+.6f} |'),
        (f'| Final paired epoch | {final["epoch"]} | '
         f'{final["conditioned_selection_loss"]:.6f} | '
         f'{final["absent_control_selection_loss"]:.6f} | '
         f'{final["selection_loss_delta"]:+.6f} |'),
        '',
        'Negative deltas favor the explicitly conditioned run.',
        '',
        '## Per-epoch audit',
        '',
        '| Epoch | Selection delta | SDPI full−absent | BiTSeed full−absent | '
        'Raw condition scale | EMA condition scale |',
        '| ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary['epochs']:
        within = row['within_conditioned_full_minus_absent']
        lines.append(
            f'| {row["epoch"]} | {row["selection_loss_delta"]:+.6f} | '
            f'{within["sdpi"]:+.6f} | {within["bitseed_endpoint"]:+.6f} | '
            f'{row["raw_condition_scale"]:+.6g} | '
            f'{row["ema_condition_scale"]:+.6g} |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = parse_args(argv)
    conditioned, control = Path(args.conditioned_run), Path(args.absent_control_run)
    deadline = time.time() + args.wait_hours * 3600
    while True:
        complete = ((conditioned / 'final_model.pt').exists()
                    and (control / 'final_model.pt').exists())
        if complete or time.time() >= deadline:
            break
        print('waiting for paired conditioned training runs', flush=True)
        time.sleep(args.poll_seconds)
    summary = compare(conditioned, control, require_complete=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'REPORT.md').write_text(render_markdown(summary))
    print(json.dumps({'out': str(output), 'complete': summary['complete'],
                      'paired_epochs': len(summary['paired_epochs'])}))


if __name__ == '__main__':
    main()
