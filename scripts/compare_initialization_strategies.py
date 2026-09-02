#!/usr/bin/env python
"""Compare current and DeepTernary-style initialization distance priors.

This CPU-only Monte Carlo diagnostic does not import either model. It measures
how often independently initialized ligand and POI centers are close enough to
interact under representative cross-edge cutoffs.
"""

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffdock-sigma", type=float, default=19.0)
    parser.add_argument("--poi-shell-min", type=float, default=30.0)
    parser.add_argument("--poi-shell-max", type=float, default=60.0)
    parser.add_argument("--deepternary-radius", type=float, default=5.0)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[20.0, 40.0, 77.0])
    parser.add_argument("--out", default="results/initialization_strategy_comparison.json")
    return parser


def random_directions(rng, count):
    directions = rng.normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def uniform_ball_offsets(rng, count, radius):
    # Radius^(1/3) produces uniform volume density rather than a center-heavy ball.
    radii = radius * np.cbrt(rng.uniform(size=count))
    return random_directions(rng, count) * radii[:, None]


def summarize(distances, cutoffs):
    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "quantiles": {
            str(q): float(np.quantile(distances, q)) for q in (0.05, 0.5, 0.95, 0.99)
        },
        "fraction_within_cutoff": {
            str(cutoff): float(np.mean(distances <= cutoff)) for cutoff in cutoffs
        },
    }


def main():
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    rng = np.random.default_rng(args.seed)

    # Current TriCoopDiff: native DiffDock Gaussian ligand center plus an
    # independently sampled POI shell around the fixed E3 center.
    ligand_current = rng.normal(scale=args.diffdock_sigma, size=(args.samples, 3))
    poi_radii = rng.uniform(args.poi_shell_min, args.poi_shell_max, size=args.samples)
    poi_current = random_directions(rng, args.samples) * poi_radii[:, None]

    # DeepTernary prediction scripts center each movable body independently and
    # apply a random translation whose magnitude is uniform in [0, 5] Å.
    ligand_deepternary = (
        random_directions(rng, args.samples)
        * rng.uniform(0.0, args.deepternary_radius, size=(args.samples, 1))
    )
    poi_deepternary = (
        random_directions(rng, args.samples)
        * rng.uniform(0.0, args.deepternary_radius, size=(args.samples, 1))
    )

    report = {
        "samples": args.samples,
        "seed": args.seed,
        "note": (
            "This compares center-distance priors only. DeepTernary is a keypoint "
            "regressor, so its 5A prior must not replace DiffDock initialization "
            "without downstream score-field validation."
        ),
        "current_tricoopdiff": summarize(
            np.linalg.norm(ligand_current - poi_current, axis=1), args.cutoffs
        ),
        "deepternary_style": summarize(
            np.linalg.norm(ligand_deepternary - poi_deepternary, axis=1), args.cutoffs
        ),
    }
    output_path = PROJECT_ROOT / args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
