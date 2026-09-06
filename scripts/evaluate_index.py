#!/usr/bin/env python
"""Run resumable case-wise evaluation for a ternary JSON index."""

import argparse
import json
from pathlib import Path

from evaluation_common import (
    completed_report, evaluation_input_signature, resolve_entry_paths, sha256_file,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def select_entries(index_path, start=0, limit=None):
    """Load a deterministic slice from an evaluation index."""
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload)
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("start must be non-negative and limit must be positive")
    return entries[start:] if limit is None else entries[start:start + limit]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/ternary_test.json")
    parser.add_argument("--trifuse-ckpt", required=True)
    parser.add_argument("--out-dir", default="results/deepternary_test")
    parser.add_argument("--native-root", default="data/TernaryDB_release/pdbs")
    parser.add_argument("--cache-dir", default="results/featurize_cache_deepternary")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
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
        "init_mode": "sdpi",
        "initialization_version": "sdpi_v1",
        "generative_process": "sdpi_flow_v1",
        "seed": args.seed,
        "n_walkers": args.n_walkers,
        "steps": args.steps,
        "native_root": str((_PROJECT_ROOT / args.native_root).resolve()),
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    runner = EvaluationRunner(
        _PROJECT_ROOT, checkpoint=checkpoint, device=args.device,
    )
    failures = []
    for position, entry in enumerate(entries, start=1):
        case = entry["case"]
        report_path = output / f"{case}.json"
        try:
            paths = resolve_entry_paths(entry, _PROJECT_ROOT)
            native = _PROJECT_ROOT / args.native_root / case / "gt_complex.pdb"
            native = native if native.is_file() else None
            expected_config = {
                "init_mode": "sdpi",
                "initialization_version": "sdpi_v1",
                "generative_process": "sdpi_flow_v1",
                "seed": args.seed,
                "n_walkers": args.n_walkers,
                "steps": args.steps,
                "checkpoint_sha256": runner.checkpoint_sha256,
                "expert_checkpoint_sha256": runner.expert.provenance["score_checkpoint_sha256"],
                "expert_config_sha256": runner.expert_config_sha256,
                "input_signature": evaluation_input_signature(paths, native),
                "freeze_poi": False,
                "t_anchor": 1.0,
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
            )
            write_report_atomic(report, report_path)
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
