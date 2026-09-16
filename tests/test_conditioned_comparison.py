import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from compare_conditioned_runs import compare, render_markdown


def _write_run(path, drop_all, losses):
    path.mkdir()
    config = {
        'output_dir': str(path),
        'condition_dropout': {'drop_all': drop_all},
        'epochs': 2,
        'seed': 7315,
        'parent_checkpoint_sha256': 'abc123',
    }
    (path / 'config.json').write_text(json.dumps(config))
    with (path / 'metrics.jsonl').open('w') as handle:
        for epoch, selection in enumerate(losses, 1):
            cells = {
                'sdpi/full': selection - .03,
                'sdpi/absent': selection + .01,
                'bitseed_endpoint/full': selection - .02,
                'bitseed_endpoint/absent': selection + .04,
            }
            row = {
                'event': 'validation',
                'epoch': epoch,
                'selection_loss': selection,
                'by_cell': cells,
                'raw_condition_scale': epoch / 10,
                'ema_condition_scale': epoch / 100,
            }
            handle.write(json.dumps(row) + '\n')


def test_paired_conditioned_comparison(tmp_path):
    conditioned, control = tmp_path / 'conditioned', tmp_path / 'control'
    _write_run(conditioned, .1, [2., 1.5])
    _write_run(control, 1., [2.1, 1.8])
    summary = compare(conditioned, control)
    assert summary['complete']
    assert summary['best_conditioned']['epoch'] == 2
    assert summary['best_absent_control']['epoch'] == 2
    assert summary['epochs'][-1]['selection_loss_delta'] < 0
    assert summary['epochs'][-1]['within_conditioned_full_minus_absent']['sdpi'] < 0
    report = render_markdown(summary)
    assert 'does **not** establish a DockQ' in report
    assert 'Final paired epoch' in report


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory(prefix='conditioned-comparison-test-') as directory:
        test_paired_conditioned_comparison(Path(directory))
