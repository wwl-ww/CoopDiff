#!/usr/bin/env python
"""Run the step-0 native/inline parity probe for both expert branches.

This is intentionally a direct Python job, not a SLURM workload. It writes one
JSON report containing the E3-L and POI-L results and exits non-zero when either
branch fails or the underlying probe cannot run.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE = PROJECT_ROOT / "scripts" / "probe_native_inline_step0.py"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="5T35_D_A_759")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    parser.add_argument("--model-dir", default="checkpoints/score_model")
    parser.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    parser.add_argument("--cache-dir", default="results/featurize_cache")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--nan-trace", action="store_true")
    parser.add_argument("--nan-clamp", default="none")
    parser.add_argument("--init-mode", choices=["native", "coupled5a", "anchor_local"], default="coupled5a")
    parser.add_argument("--t-anchor", type=float, default=None)
    parser.add_argument("--out", default="results/sampling_parity_regression.json")
    return parser


def run_branch(args, branch, output_path):
    command = [
        sys.executable,
        str(PROBE),
        "--case", args.case,
        "--device", args.device,
        "--seed", str(args.seed),
        "--steps", str(args.steps),
        "--protac-dir", args.protac_dir,
        "--model-dir", args.model_dir,
        "--ckpt", args.ckpt,
        "--cache-dir", args.cache_dir,
        "--branch", branch,
        "--atol", str(args.atol),
        "--rtol", str(args.rtol),
        "--nan-clamp", str(args.nan_clamp),
        "--init-mode", args.init_mode,
        "--t-anchor", str(args.t_anchor) if args.t_anchor is not None else ("0.7" if args.init_mode == "anchor_local" else "1.0"),
        "--out", str(output_path),
    ]
    if args.nan_trace:
        command.append("--nan-trace")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if not output_path.exists():
        return {
            "parity_pass": False,
            "probe_exit_code": completed.returncode,
            "error": "probe did not produce its JSON report",
        }
    with output_path.open() as handle:
        result = json.load(handle)
    result["probe_exit_code"] = completed.returncode
    result["parity_pass"] = bool(result.get("parity_pass")) and completed.returncode == 0
    return result


def main():
    args = build_parser().parse_args()
    output_path = PROJECT_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tricoopdiff-parity-") as temp_dir:
        temp_dir = Path(temp_dir)
        branches = {
            branch: run_branch(args, branch, temp_dir / f"{branch}.json")
            for branch in ("sub1", "sub2")
        }

    passed = all(result.get("parity_pass", False) for result in branches.values())
    report = {
        "case": args.case,
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "init_mode": args.init_mode,
        "nan_clamp": args.nan_clamp,
        "t_anchor": args.t_anchor if args.t_anchor is not None else (0.7 if args.init_mode == "anchor_local" else 1.0),
        "atol": args.atol,
        "rtol": args.rtol,
        "parity_pass": passed,
        "branches": branches,
    }
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
