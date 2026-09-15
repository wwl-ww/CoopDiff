#!/usr/bin/env python
"""Prepare a predeclared Molecular Glue diagnostic panel and v7 caches.

The five cases are fixed by evenly spaced positions in the official MGD94
ordering before any model scores are inspected.  They are a route-selection
panel, not the final MGD94 benchmark; the remaining cases stay untouched.
"""

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_ESM_SNAPSHOTS = (ROOT / "hf_cache/models--facebook--esm2_t33_650M_UR50D/snapshots")
_LOCAL_ESM = sorted(
    path for path in _ESM_SNAPSHOTS.glob("*")
    if (path / "config.json").is_file()
)
if len(_LOCAL_ESM) == 1:
    os.environ.setdefault("TRICOOPDIFF_ESM_MODEL", str(_LOCAL_ESM[0]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from featurize_common import featurize_complex  # noqa: E402


PANEL_POSITIONS = (0, 23, 46, 69, 93)


def select_panel(cases):
    """Select fixed, approximately quartile-spaced cases from MGD94."""
    if len(cases) != 94 or len(set(cases)) != 94:
        raise ValueError("expected 94 unique official MGD cases")
    return [cases[position] for position in PANEL_POSITIONS]


def build_entries(cases, ternary_root, restored_root, allow_pdb_fallback=False):
    """Resolve the bound protein structures and restored CCD ligands."""
    entries, fallbacks = [], []
    for case in cases:
        case_root = ternary_root / case
        restored_ligand = restored_root / case / "ligand.sdf"
        if restored_ligand.is_file():
            ligand = restored_ligand
            chemistry = "ccd_restored"
        elif allow_pdb_fallback:
            ligand = case_root / "ligand.pdb"
            chemistry = "pdb_inferred_fallback"
            fallbacks.append(case)
        else:
            ligand = restored_ligand
        paths = {
            "protein1": case_root / "protein1.pdb",
            "protein2": case_root / "protein2.pdb",
            "ligand": ligand,
        }
        required = [*paths.values(), case_root / "ligand.pdb",
                    case_root / "gt_complex.pdb"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{case}: missing inputs: {missing}")
        entries.append({
            "case": case,
            **{key: str(path.resolve()) for key, path in paths.items()},
            "source": "DeepTernary/MolecularGlue-MGD94",
            "ligand_chemistry": chemistry,
        })
    return entries, fallbacks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-list", type=Path,
        default=ROOT / "DeepTernary/data/MolecularGlue/test_all.txt",
    )
    parser.add_argument(
        "--ternary-root", type=Path,
        default=ROOT / "DeepTernary/data/TernaryDB/pdbs",
    )
    parser.add_argument(
        "--restored-root", type=Path,
        default=ROOT / "results/handoff_artifacts/data/ccd_restored_v1/mgd94",
    )
    parser.add_argument(
        "--score-model", type=Path,
        default=ROOT / "results/handoff_artifacts/checkpoints/score_model",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/mgd_gate/diagnostic_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument(
        "--full", action="store_true",
        help="Prepare all MGD94 cases; explicitly allow PDB fallback when CCD restoration is impossible",
    )
    args = parser.parse_args()

    official_cases = [
        line.strip().split(",", 1)[0]
        for line in args.test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    panel_cases = official_cases if args.full else select_panel(official_cases)
    entries, fallbacks = build_entries(
        panel_cases, args.ternary_root, args.restored_root,
        allow_pdb_fallback=args.full,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / (
        "mgd94_full_v1.json" if args.full else "mgd94_diagnostic_v1.json"
    )
    index = {
        "entries": entries,
        "rejected": [],
        "metadata": {
            "protocol": (
                "mgd94_bound_structure_sdpi_randomized_placement_full_v1"
                if args.full else
                "mgd94_bound_structure_sdpi_randomized_placement_diagnostic_v1"
            ),
            "split": ("test" if args.full else
                      "diagnostic_route_selection_not_final_test"),
            "selection": {
                "rule": ("all_official_cases_in_published_order" if args.full else
                         "fixed_evenly_spaced_positions_in_official_order"),
                "one_based_positions": (
                    list(range(1, 95)) if args.full else
                    [position + 1 for position in PANEL_POSITIONS]
                ),
                "selected_cases": panel_cases,
                "selected_before_model_scoring": True,
            },
            "official_cases": 94,
            "diagnostic_cases": len(panel_cases),
            "reserved_final_cases": 94 - len(panel_cases),
            "input_geometry": "bound protein conformations and native ligand conformer",
            "sampling_geometry": "SDPI t=1 randomized component placement",
            "warning": (
                "Oracle-best over 32 walkers is not model-ranked top-1."
                if args.full else
                "Do not report this five-case panel as the final MGD94 benchmark."
            ),
            "ligand_chemistry": {
                "ccd_restored_cases": len(entries) - len(fallbacks),
                "pdb_inferred_fallback_cases": fallbacks,
                "fallback_reason": (
                    "3IQV deposited FSC is missing six CCD heavy atoms"
                    if fallbacks else None
                ),
            },
        },
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"index: {index_path}", flush=True)
    print(f"cases: {' '.join(panel_cases)}", flush=True)

    if args.index_only:
        return
    score_args = Namespace(**yaml.full_load(
        (args.score_model / "model_parameters.yml").read_text(encoding="utf-8")
    ))
    cache_dir = args.out / "cache_ccd_v7"
    for number, entry in enumerate(entries, start=1):
        print(f"[{number}/{len(entries)}] featurizing {entry['case']}", flush=True)
        featurize_complex(
            entry["case"], entry["protein1"], entry["protein2"],
            entry["ligand"], score_args, args.device, cache_dir=cache_dir,
        )
    print(f"cache: {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
