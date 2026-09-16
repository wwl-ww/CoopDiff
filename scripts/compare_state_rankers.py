"""CPU-only R0/R1 comparison after the complete frozen collection/training pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-hours', type=float, default=0.)
    args = parser.parse_args()
    if not 0 <= args.wait_hours <= 24:
        parser.error('wait budget must be between zero and 24 hours')
    if args.out.exists():
        raise FileExistsError('use a new comparison directory')
    run = args.run_dir.resolve()
    deadline = time.monotonic() + args.wait_hours * 3600
    previous_stage = None
    while True:
        status = json.loads((run / 'pipeline_status.json').read_text())
        stage = status['stage']
        if stage != previous_stage:
            print(f'UPSTREAM: {stage}', flush=True)
            previous_stage = stage
        if stage == 'complete':
            break
        if stage not in ('collecting', 'scoring', 'auditing_labels', 'training_rankers'):
            raise RuntimeError(f'upstream pipeline did not complete: {stage}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'upstream still {stage}; no partial-case ranking comparison')
        time.sleep(min(30., max(0., deadline - time.monotonic())))
    args.out.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    reports = {}
    import torch
    from tricoopdiff.ranking_inference import export_quality_checkpoint
    for mode in ('r0_fixed_geometry', 'ranker_geometry', 'ranker_frozen_geometry'):
        predictions = args.out / mode
        command = [sys.executable, str(ROOT / 'scripts/rank_pose_archive.py'),
                   '--archives', str(run / 'val_archives'), '--out', str(predictions)]
        if mode != 'r0_fixed_geometry':
            # Export outside the native-free subprocess; old frozen trainers save metadata in best.pt.
            checkpoint = torch.load(run / mode / 'best.pt', map_location='cpu', weights_only=False)
            inference_path = args.out / f'{mode}_inference.pt'
            torch.save(export_quality_checkpoint(checkpoint), inference_path)
            command += ['--ranker', str(inference_path)]
        with (args.out / f'{mode}.log').open('x') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            report = args.out / f'{mode}_audit.json'
            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_state_ranking.py'),
                            '--predictions', str(predictions), '--labels', str(run / 'val_labels'),
                            '--out', str(report)], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[mode] = json.loads(report.read_text())['summary']
    result = {'scope': 'internal validation; same data selected R1 checkpoints, not holdout',
              'ranking_scope': 'full archived initial/final/union; not a 256-to-64 selector experiment',
              'reports': reports}
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Internal R0/R1 comparison', '', result['scope'], '', result['ranking_scope'], '',
             '| Method | Source / states | Top1 | Oracle | ≥0.23 | ≥0.49 | Random top1 expectation |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for mode, strata in reports.items():
        for name, row in strata.items():
            lines.append(f'| {mode} | {name} | {row["mean_top1"]:.4f} | {row["mean_oracle"]:.4f} | '
                         f'{row["thresholds"]["0.23"]["top1_hit"]}/{row["cases"]} | '
                         f'{row["thresholds"]["0.49"]["top1_hit"]}/{row["cases"]} | '
                         f'{row["mean_random_top1_mean"]:.4f} |')
    (args.out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(f'COMPLETE: {args.out / "comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
