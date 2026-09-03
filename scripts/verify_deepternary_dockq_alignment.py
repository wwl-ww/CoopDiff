#!/usr/bin/env python
"""Verify the full-atom DeepTernary DockQ bridge on a native pose."""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]

from featurize_common import featurize_complex  # noqa: E402
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="5T35_D_A_759")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/deepternary_dockq_alignment.json")
    args = parser.parse_args()

    case_dir = PROJECT_ROOT / "DeepTernary" / "output" / "protac22" / args.case
    model_dir = PROJECT_ROOT / "checkpoints" / "score_model"
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))
    ligand_input = case_dir / "ligand.sdf"
    e3, poi, ligand, _ = featurize_complex(
        args.case,
        str((case_dir / "protein1.pdb").resolve()),
        str((case_dir / "protein2.pdb").resolve()),
        str(ligand_input.resolve()),
        model_args,
        args.device,
        cache_dir=str(PROJECT_ROOT / "results" / "featurize_cache"),
    )
    result = compute_fullatom_dockq(
        pred_ligand=ligand["coords"][0].detach().cpu().numpy(),
        pred_poi_ca=poi["coords"][0].detach().cpu().numpy(),
        featurized_e3_ca=e3["coords"][0].detach().cpu().numpy(),
        protein1_pdb=case_dir / "protein1.pdb",
        protein2_pdb=case_dir / "protein2.pdb",
        ligand_pdb=case_dir / "ligand.pdb",
        native_complex_pdb=case_dir / "gt_complex.pdb",
        deepternary_root=PROJECT_ROOT / "DeepTernary",
    )
    result["case"] = args.case
    result["pass"] = result["dockq"] >= 0.999
    output = PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
