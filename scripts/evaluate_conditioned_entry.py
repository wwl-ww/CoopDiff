"""Explicit one-case evaluator for conditioned checkpoints and manifests."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import load_index_entry, resolve_entry_paths
from evaluation_runner import EvaluationRunner, write_report_atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--condition-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model-dir', default='checkpoints/score_model')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--walkers', type=int, default=32)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--init-mode', choices=('sdpi', 'expert_anchor'), default='expert_anchor')
    args = parser.parse_args(argv)
    if args.walkers < 1 or args.steps < 1:
        parser.error('walkers and steps must be positive')
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        parser.error('use a new output path')
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index
    entry = load_index_entry(index, args.case)
    paths = resolve_entry_paths(entry, ROOT)
    case_dir = paths['ligand'].parent
    native = next((path for path in (case_dir / 'gt_complex.pdb', case_dir / 'complex.pdb')
                   if path.is_file()), None)
    runner = EvaluationRunner(
        ROOT, args.checkpoint, device=args.device, model_dir=args.model_dir,
        allow_conditioned_checkpoint=True)
    report = runner.evaluate_case(
        args.case, paths, cache_dir=ROOT / args.cache_dir,
        native_complex_pdb=native, n_walkers=args.walkers, steps=args.steps,
        seed=args.seed, init_mode=args.init_mode,
        condition_manifest=args.condition_manifest)
    write_report_atomic(report, out)
    print(json.dumps({'out': str(out), 'dockq_best': report['structure']['dockq_best'],
                      'valid_walkers': report['ensemble']['n_valid_walkers']}, indent=2))


if __name__ == '__main__':
    main()
