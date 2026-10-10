#!/usr/bin/env python
"""Run resumable case-wise evaluation for a ternary JSON index."""

import argparse
import json
from pathlib import Path

from evaluation_common import (
    completed_report, evaluation_input_signature, resolve_entry_paths, sha256_file,
)
from tricoopdiff.models.assembly import DIRECT_RANK_FEATURE_STATE, LEGACY_RANK_FEATURE_STATE


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_entries(index_path, start=0, limit=None):
    """Load a deterministic slice from an evaluation index."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = payload["entries"] if isinstance(payload, dict) else payload
    cases = [entry['case'] for entry in entries]
    if len(set(cases)) != len(cases):
        raise ValueError('evaluation index contains duplicate case names')
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("start must be non-negative and limit must be positive")
    return entries[start:] if limit is None else entries[start:start + limit]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/ternary_test.json")
    parser.add_argument("--trifuse-ckpt", required=True)
    parser.add_argument("--out-dir", default="results/deepternary_test")
    parser.add_argument("--native-root", default="data/TernaryDB_release/pdbs")
    parser.add_argument("--metric-template-root",
                        help="Prealigned bound protein1/2 templates for DeepTernary unbound scoring only")
    parser.add_argument("--deepternary-root", default="DeepTernary",
                        help="Local DeepTernary installation with executable DockQ/fnat")
    parser.add_argument("--cache-dir", default="results/featurize_cache_deepternary")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--init-mode", choices=["sdpi", "expert_anchor"],
                        default="sdpi")
    parser.add_argument("--binary-anchor-steps", type=int, default=5)
    parser.add_argument("--anchor-proposal-attempts", type=int, default=4)
    parser.add_argument("--anchor-selection", choices=["geometry", "consensus"],
                        default="geometry")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true",
                        help="Explicit legacy diagnostic control; never mix with deterministic results")
    parser.add_argument(
        "--allow-experimental-checkpoint", action="store_true",
        help="Explicitly evaluate a validated frozen factorial-v1 checkpoint with EMA weights",
    )
    parser.add_argument(
        "--save-rank-features", action="store_true",
        help="Persist invariant final-step contexts for supervised ranking on non-benchmark data",
    )
    parser.add_argument("--conformer-policy",
                        choices=["etkdgv3_random", "diffdock_v1_1"],
                        default="etkdgv3_random")
    return parser


def main():
    import torch
    from evaluation_runner import EvaluationRunner, write_report_atomic

    args = build_parser().parse_args()
    index = Path(args.index).resolve()
    checkpoint_path = Path(args.trifuse_ckpt)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _PROJECT_ROOT / checkpoint_path
    checkpoint = str(checkpoint_path.resolve())
    entries = select_entries(index, args.start, args.limit)
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "index": str(index),
        "index_sha256": sha256_file(index),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "cases": [entry["case"] for entry in entries],
        "init_mode": args.init_mode,
        "initialization_version": (
            ("dual_diffdock_rigid_anchor_consensus_v1"
             if args.anchor_selection == "consensus"
             else "dual_diffdock_rigid_anchor_v1")
            if args.init_mode == "expert_anchor" else "sdpi_v1"),
        "binary_anchor_steps": (args.binary_anchor_steps
                                if args.init_mode == "expert_anchor" else None),
        "anchor_proposal_attempts": (args.anchor_proposal_attempts
                                     if args.init_mode == "expert_anchor" else None),
        "anchor_selection": (args.anchor_selection
                             if args.init_mode == "expert_anchor" else None),
        "generative_process": "sdpi_flow_v1",
        "seed": args.seed,
        "n_walkers": args.n_walkers,
        "steps": args.steps,
        "native_root": str((_PROJECT_ROOT / args.native_root).resolve()),
        "metric_template_root": args.metric_template_root,
        "deterministic_algorithms": not args.allow_nondeterministic,
        "conformer_policy": args.conformer_policy,
        "allow_experimental_checkpoint": args.allow_experimental_checkpoint,
        "rank_features_saved": args.save_rank_features,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    runner = EvaluationRunner(
        _PROJECT_ROOT, checkpoint=checkpoint, device=args.device,
        deepternary_root=args.deepternary_root,
        deterministic=not args.allow_nondeterministic,
        conformer_policy=args.conformer_policy,
        allow_experimental_checkpoint=args.allow_experimental_checkpoint,
    )
    failures = []
    for position, entry in enumerate(entries, start=1):
        case = entry["case"]
        report_path = output / f"{case}.json"
        try:
            paths = resolve_entry_paths(entry, _PROJECT_ROOT)
            if args.metric_template_root:
                template_root = _PROJECT_ROOT / args.metric_template_root / case
                for name in ('protein1', 'protein2'):
                    template = template_root / f'{name}.pdb'
                    if not template.is_file():
                        raise FileNotFoundError(template)
                    paths['metric_' + name] = template.resolve()
            native = _PROJECT_ROOT / args.native_root / case / "gt_complex.pdb"
            native = native if native.is_file() else None
            expected_config = {
                **runner.reproducibility,
                "init_mode": args.init_mode,
                "initialization_version": (
                    ("dual_diffdock_rigid_anchor_consensus_v1"
                     if args.anchor_selection == "consensus"
                     else "dual_diffdock_rigid_anchor_v1")
                    if args.init_mode == "expert_anchor" else "sdpi_v1"),
                "binary_anchor_steps": (args.binary_anchor_steps
                                        if args.init_mode == "expert_anchor" else None),
                "anchor_proposal_attempts": (args.anchor_proposal_attempts
                                             if args.init_mode == "expert_anchor" else None),
                "anchor_selection": (args.anchor_selection
                                     if args.init_mode == "expert_anchor" else None),
                "generative_process": "sdpi_flow_v1",
                "seed": args.seed,
                "n_walkers": args.n_walkers,
                "steps": args.steps,
                "checkpoint_sha256": runner.checkpoint_sha256,
                "expert_checkpoint_sha256": runner.expert.provenance["score_checkpoint_sha256"],
                "expert_config_sha256": runner.expert_config_sha256,
                "input_signature": evaluation_input_signature(paths, native),
                "freeze_poi": False,
                "activation_clamp_mode": runner.activation_clamp_mode,
                "conformer_policy": args.conformer_policy,
                "t_anchor": 1.0,
                "dockq_source_sha256": sha256_file(runner.deepternary_root / 'DockQ/dockq_util.py'),
                "dockq_binary_sha256": sha256_file(runner.deepternary_root / 'DockQ/fnat'),
                "rank_features_saved": args.save_rank_features,
                "rank_feature_state": (
                    DIRECT_RANK_FEATURE_STATE
                    if args.save_rank_features and runner.checkpoint['model_config'].get('direct_ternary_motion', False)
                    else LEGACY_RANK_FEATURE_STATE if args.save_rank_features else None
                ),
            }
            if not args.no_resume and completed_report(
                    report_path, case, checkpoint, expected_config,
                    require_exact_dockq=True):
                print(f"[{position}/{len(entries)}] SKIP completed {case}", flush=True)
                continue
            print(f"[{position}/{len(entries)}] RUN {case}", flush=True)
            report = runner.evaluate_case(
                case, paths, cache_dir=_PROJECT_ROOT / args.cache_dir,
                native_complex_pdb=native,
                n_walkers=args.n_walkers, steps=args.steps, seed=args.seed,
                init_mode=args.init_mode,
                binary_anchor_steps=args.binary_anchor_steps,
                anchor_proposal_attempts=args.anchor_proposal_attempts,
                anchor_selection=args.anchor_selection,
                return_rank_features=args.save_rank_features,
            )
            write_report_atomic(report, report_path)
            ensemble = report['ensemble']
            if (ensemble['n_valid_walkers'] == 0
                    or ensemble['n_exact_dockq_walkers'] != ensemble['n_valid_walkers']):
                raise RuntimeError('incomplete full-atom DockQ; see saved per-seed report')
        except Exception as error:
            failures.append({
                "case": case,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"[{position}/{len(entries)}] FAIL {case}: {error}", flush=True)
            if args.device == "cuda":
                torch.cuda.empty_cache()

    status = {
        "requested_cases": len(entries),
        "failed_cases": failures,
        "complete": not failures,
    }
    (output / "batch_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8",
    )
    print(json.dumps(status, indent=2), flush=True)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
