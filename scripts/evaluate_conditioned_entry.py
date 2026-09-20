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
    parser.add_argument('--project-root', default=str(ROOT),
                        help='Root containing indexed data and model assets')
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
    parser.add_argument('--init-mode', choices=('sdpi', 'expert_anchor', 'binary_pose',
                                                'deepternary_pose'),
                        default='expert_anchor')
    parser.add_argument('--t-anchor', type=float)
    parser.add_argument('--binary-pose-candidates', type=int, default=24)
    parser.add_argument('--deepternary-pose')
    parser.add_argument('--condition-guidance', choices=('full', 'residual'), default='full')
    parser.add_argument('--condition-guidance-scale', type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.walkers < 1 or args.steps < 1 or args.binary_pose_candidates < 1:
        parser.error('walkers, steps, and binary-pose-candidates must be positive')
    if not 0.0 <= args.condition_guidance_scale <= 1.0:
        parser.error('condition-guidance-scale must be in [0, 1]')
    if args.init_mode == 'deepternary_pose' and args.t_anchor is None:
        parser.error('deepternary-pose refinement requires an explicit --t-anchor')
    project_root = Path(args.project_root).resolve()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        parser.error('use a new output path')
    index = Path(args.index)
    if not index.is_absolute():
        index = project_root / index
    entry = load_index_entry(index, args.case)
    paths = resolve_entry_paths(entry, project_root)
    case_dir = paths['ligand'].parent
    native = next((path for path in (case_dir / 'gt_complex.pdb', case_dir / 'complex.pdb')
                   if path.is_file()), None)
    runner = EvaluationRunner(
        project_root, args.checkpoint, device=args.device, model_dir=args.model_dir,
        allow_conditioned_checkpoint=True)
    report = runner.evaluate_case(
        args.case, paths, cache_dir=project_root / args.cache_dir,
        native_complex_pdb=native, n_walkers=args.walkers, steps=args.steps,
        seed=args.seed, init_mode=args.init_mode,
        t_anchor=args.t_anchor,
        condition_manifest=args.condition_manifest,
        binary_pose_candidates=args.binary_pose_candidates,
        deepternary_pose=args.deepternary_pose,
        condition_guidance=args.condition_guidance,
        condition_guidance_scale=args.condition_guidance_scale)
    write_report_atomic(report, out)
    print(json.dumps({'out': str(out), 'dockq_oracle_best': report['structure']['dockq_best'],
                      'selection_protocol': 'native_metric_oracle; not model top-1',
                      'valid_walkers': report['ensemble']['n_valid_walkers']}, indent=2))


if __name__ == '__main__':
    main()
