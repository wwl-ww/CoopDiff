#!/usr/bin/env python
"""Evaluate calibrated predicted ΔΔG against experimental cooperativity alpha."""

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tricoopdiff.ensemble_analysis import (  # noqa: E402
    cooperativity_from_ddg,
    cooperativity_ranking,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="CSV with case,predicted_ddg,experimental_alpha")
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case", "predicted_ddg", "experimental_alpha"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"FATAL: input must contain columns {sorted(required)}")
    predicted_ddg = [float(row["predicted_ddg"]) for row in rows]
    experimental_alpha = [float(row["experimental_alpha"]) for row in rows]
    metrics = cooperativity_ranking(
        predicted_ddg, experimental_alpha,
        temperature_k=args.temperature_k, top_fraction=args.top_fraction,
    )
    predictions = [
        {
            "case": row["case"],
            "predicted_ddg_kcal_mol": ddg,
            "predicted_alpha": cooperativity_from_ddg(ddg, args.temperature_k),
            "experimental_alpha": alpha,
        }
        for row, ddg, alpha in zip(rows, predicted_ddg, experimental_alpha)
    ]
    report = {
        "definition": "alpha=exp(-(DeltaG_ternary-DeltaG_binary)/(RT))",
        "temperature_k": args.temperature_k,
        "metrics": metrics,
        "predictions": predictions,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
