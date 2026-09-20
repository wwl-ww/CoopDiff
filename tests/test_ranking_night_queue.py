"""Fast orchestration test: useful continuation after early training completion."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_ranking_night as night


def test_early_completion_continues_but_long_training_does_not_duplicate_work():
    for training_seconds, expected_batches in ((1000., 2), (5000., 0)):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / 'run'
            archives = root / 'results/ranking_fullpool_train32_20260917'
            labels = Path(str(archives) + '_labels')
            archives.mkdir(parents=True); labels.mkdir()
            (archives / 'manifest.json').write_text(json.dumps({'cases': ['train0', 'train1']}))
            (archives / 'train0.complete.json').write_text('{}')
            for name in ('train0', 'train1'):
                (labels / f'{name}.json').write_text('{}')
            extra = root / 'results/model_v3_ranking_night160_20260915/extra_train_archives'
            extra.mkdir(parents=True)
            (extra / 'manifest.json').write_text(json.dumps({'cases': ['extra0', 'extra1']}))
            now, calls = [100.], []

            def execute(command, **kwargs):
                script = Path(command[2]).name
                calls.append(script)
                if script == 'train_structural_ranker.py':
                    out = Path(command[command.index('--out') + 1])
                    out.mkdir()
                    (out / 'history.jsonl').write_text('{"epoch": 20}\n')
                    now[0] += training_seconds
                elif script == 'extend_ranking_archive.py':
                    assert command[command.index('--max-new-pairs') + 1] == '64'
                    now[0] += 2000.

            with patch.object(night, '__file__', str(root / 'scripts/run_ranking_night.py')), \
                 patch.object(night, 'process_token', return_value=None), \
                 patch.object(night.time, 'time', side_effect=lambda: now[0]), \
                 patch.object(night.subprocess, 'run', side_effect=execute), \
                 patch.object(sys, 'argv', ['queue', '--gpu', '5', '--run-root', str(run),
                                           '--collector-pid', '123', '--hours', str(4000. / 3600)]):
                night.main()
            status = json.loads((run / 'gpu5_status.json').read_text())
            assert status['stage'] == 'complete'
            assert status['continuation_batches'] == expected_batches
            assert status['elapsed_hours'] >= 4000. / 3600
            assert calls.count('train_structural_ranker.py') == 1
            assert calls.count('extend_ranking_archive.py') == expected_batches


if __name__ == '__main__':
    test_early_completion_continues_but_long_training_does_not_duplicate_work()
    print('Night queue: early/late completion tests passed')
