#!/usr/bin/env python3
"""
Compare SOTA audit results against the official DeepTernary baseline.

Reads the official results from result_20240515193524.txt and compares
per-case and aggregate metrics against an audit results directory.

Usage:
    python scripts/compare_audit_results.py --audit-dir results/audit_sota/
    python scripts/compare_audit_results.py --audit-dir results/audit_sota/ --output comparison.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# ── Official results (hardcoded from result_20240515193524.txt) ──
# Order matches the 22 PROTAC test cases
# Values: [pred_dockq, best_dockq, best_rank, best_smrmsd, sm_rank, rmsd10, hit_count, hit_rate]
OFFICIAL_DATA = [
    [0.8617, 0.9516, 1, 1.2334, 21, 0.7750, "1/9/4", 0.3500],
    [0.5834, 0.8881, 1, 1.3457, 11, 0.8000, "0/16/3", 0.4750],
    [0.4301, 0.7018, 1, 1.2626, 5, 0.6000, "16/9/0", 0.6250],
    [0.0429, 0.7108, 37, 1.6262, 37, 0.0750, "0/1/0", 0.0250],
    [0.4413, 0.6776, 1, 1.4340, 6, 0.9250, "5/4/0", 0.2250],
    [0.2020, 0.7623, 2, 1.5412, 2, 0.9250, "5/8/0", 0.3250],
    [0.5375, 0.7023, 1, 1.3960, 3, 0.9750, "10/3/1", 0.3500],
    [0.3538, 0.6517, 1, 1.6719, 19, 0.9750, "8/5/0", 0.3250],
    [0.6502, 0.7253, 1, 1.6847, 11, 1.0000, "17/7/0", 0.6000],
    [0.6219, 0.7782, 1, 1.2239, 16, 1.0000, "9/6/1", 0.4000],
    [0.5619, 0.8832, 1, 1.3435, 5, 0.6500, "5/13/5", 0.5750],
    [0.5413, 0.8107, 1, 1.2335, 7, 0.7500, "2/12/6", 0.5000],
    [0.4241, 0.8765, 1, 0.9438, 18, 0.3000, "7/8/3", 0.4500],
    [0.4819, 0.9053, 1, 0.9595, 7, 0.4000, "5/14/1", 0.5000],
    [0.3188, 0.6417, 1, 1.3075, 3, 0.2250, "0/4/0", 0.1000],
    [0.7486, 0.8533, 1, 1.1343, 4, 0.7500, "2/27/3", 0.8000],
    [0.4972, 0.6549, 1, 1.6887, 28, 0.7250, "1/0/0", 0.0250],
    [0.2484, 0.2615, 1, 5.2142, 18, 1.0000, "0/0/0", 0.0000],
    [0.1642, 0.5990, 2, 2.2444, 22, 0.3250, "10/9/0", 0.4750],
    [0.1818, 0.5889, 2, 1.0556, 5, 0.5250, "12/22/0", 0.8500],
    [0.3650, 0.4792, 1, 2.5270, 23, 0.1750, "4/3/0", 0.1750],
    [0.1728, 0.4316, 2, 1.5287, 2, 0.1750, "22/10/0", 0.8000],
]

OFFICIAL_MEANS = {
    "pred_dockq": 0.4287,
    "best_dockq": 0.7062,
    "best_rank": 2.8182,
    "best_smrmsd": 1.6182,
    "sm_rank": 12.4091,
    "rmsd10": 0.6386,
    "hit_rate": 0.4068,
}

COLUMNS = [
    "pred_dockq", "best_dockq", "best_rank", "best_smrmsd",
    "sm_rank", "rmsd10", "hit_count", "hit_rate",
]


def parse_protac22_order(protac_dir: str = "DeepTernary/output/protac22") -> list:
    """Get sorted list of case names matching protac22.txt order."""
    protac22_path = Path(protac_dir) / ".." / ".." / "data" / "PROTAC" / "protac22.txt"
    # Fall back to directory listing
    if not protac22_path.exists():
        case_dirs = sorted(d for d in os.listdir(protac_dir)
                          if os.path.isdir(Path(protac_dir) / d))
        return case_dirs

    with open(protac22_path) as f:
        cases = [line.strip() for line in f if line.strip()]
    return cases


def load_audit_csv(audit_dir: str, case_name: str) -> dict:
    """Load per-case summary CSV from audit output."""
    case_dir = Path(audit_dir) / case_name
    if not case_dir.exists():
        return None
    # Support both naming conventions: summary.csv and summary_<case>.csv
    candidates = [
        case_dir / "summary.csv",
        case_dir / f"summary_{case_name}.csv",
    ] + sorted(case_dir.glob("summary*.csv"))
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        return None

    with open(csv_path) as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None

    # Parse CSV: seed,dockq,irmsd,lrmsd,fnat,lig_rmsd,...
    header = lines[0].strip().split(",")
    dockq_idx = header.index("dockq") if "dockq" in header else 1

    dockqs = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) > dockq_idx:
            try:
                dockqs.append(float(parts[dockq_idx]))
            except ValueError:
                continue

    if not dockqs:
        return None

    return {
        "pred_dockq": dockqs[0],  # sorted by p2_rmsd, first = best rank
        "best_dockq": max(dockqs),
        "best_rank": dockqs.index(max(dockqs)) + 1,
        "mean_dockq": np.mean(dockqs),
        "median_dockq": np.median(dockqs),
        "std_dockq": np.std(dockqs),
        "n_seeds": len(dockqs),
        "all_dockqs": dockqs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare audit results against official DeepTernary baseline")
    parser.add_argument("--audit-dir", required=True,
                        help="Directory containing per-case audit output")
    parser.add_argument("--output", "-o", default=None,
                        help="Save comparison JSON to this file")
    parser.add_argument("--protac-dir", default="DeepTernary/output/protac22",
                        help="Directory with PROTAC benchmark cases")
    args = parser.parse_args()

    audit_dir = Path(args.audit_dir)
    cases = parse_protac22_order(args.protac_dir)

    print(f"{'Case':<22s} {'Audit Best':>10s} {'Official Best':>14s} {'Delta':>8s} {'Status':>8s}")
    print("-" * 70)

    audit_means = {k: [] for k in COLUMNS}
    per_case = {}
    n_matched = 0
    n_missing = 0

    for i, case in enumerate(cases):
        result = load_audit_csv(str(audit_dir), case)
        if result is None:
            print(f"{case:<22s} {'N/A':>10s} {OFFICIAL_DATA[i][1]:>14.4f} {'---':>8s} {'MISSING':>8s}")
            n_missing += 1
            continue

        official_best = OFFICIAL_DATA[i][1]
        delta = result["best_dockq"] - official_best
        status = "✓" if abs(delta) < 0.02 else ("↑" if delta > 0 else "↓")

        print(f"{case:<22s} {result['best_dockq']:>10.4f} {official_best:>14.4f} {delta:>+8.4f} {status:>8s}")

        per_case[case] = {
            "audit_best_dockq": result["best_dockq"],
            "official_best_dockq": official_best,
            "delta": delta,
            "audit_pred_dockq": result["pred_dockq"],
            "official_pred_dockq": OFFICIAL_DATA[i][0],
            "n_seeds": result["n_seeds"],
            "audit_mean": result["mean_dockq"],
            "audit_std": result["std_dockq"],
        }

        for k in ["pred_dockq", "best_dockq"]:
            audit_means[k].append(result[k] if k == "pred_dockq" else result["best_dockq"])
        n_matched += 1

    print("-" * 70)

    if n_matched > 0:
        audit_avg_best = np.mean(audit_means["best_dockq"])
        audit_avg_pred = np.mean(audit_means["pred_dockq"])
        delta_best = audit_avg_best - OFFICIAL_MEANS["best_dockq"]
        delta_pred = audit_avg_pred - OFFICIAL_MEANS["pred_dockq"]

        print(f"{'MEAN':<22s} {audit_avg_best:>10.4f} {OFFICIAL_MEANS['best_dockq']:>14.4f} {delta_best:>+8.4f}")
        print(f"{'  pred_dockq':<22s} {audit_avg_pred:>10.4f} {OFFICIAL_MEANS['pred_dockq']:>14.4f} {delta_pred:>+8.4f}")
        print(f"\nMatched: {n_matched}/{len(cases)}, Missing: {n_missing}")
        print(f"Target: mean best DockQ = 0.7062")

        if abs(delta_best) < 0.01:
            print("✓ MATCH — audit reproduces official baseline within 0.01 DockQ")
        elif abs(delta_best) < 0.02:
            print("~ CLOSE — within 0.02 DockQ, minor variance")
        else:
            print("✗ GAP — significant deviation from official baseline")

    # Save JSON
    summary = {
        "audit_mean_best_dockq": float(audit_avg_best) if n_matched > 0 else None,
        "official_mean_best_dockq": OFFICIAL_MEANS["best_dockq"],
        "delta_best": float(delta_best) if n_matched > 0 else None,
        "n_cases_matched": n_matched,
        "n_cases_total": len(cases),
        "per_case": per_case,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {args.output}")

    # Also print as JSON for easy parsing
    print("\n--- JSON ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
