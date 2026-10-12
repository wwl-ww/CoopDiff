#!/usr/bin/env python
"""Analyze a saved fixed-receptor-frame conformational ensemble."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tricoopdiff.metrics.ensemble import analyze_ensemble  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True,
                        help=".npy array shaped [walkers, atoms, 3]")
    parser.add_argument("--energies", help="Optional .npy per-walker energies")
    parser.add_argument("--rmsd-cutoff", type=float, default=2.0)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    poses = np.load(args.poses, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[-1] != 3:
        raise SystemExit(f"FATAL: poses must have shape [walkers, atoms, 3], got {poses.shape}")
    energies = np.load(args.energies, allow_pickle=False) if args.energies else None
    report = analyze_ensemble(
        list(poses), rmsd_cutoff=args.rmsd_cutoff,
        temperature_k=args.temperature_k, energies_kcal_mol=energies,
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
