#!/usr/bin/env python
"""
Build the PROTAC training index for TriCoopDiff.

Scans ``DeepTernary/output/protac22/`` for ternary complexes (each a directory
with ``protein1.pdb`` [E3], ``protein2.pdb`` [POI] and ``ligand.sdf``/``ligand.pdb``
[PROTAC]) and writes a JSON index consumed by ``scripts/train_entry.py``.

Output format::
    {"entries": [
        {"case": "5T35_D_A_759",
         "protein1": "/abs/path/protein1.pdb",
         "protein2": "/abs/path/protein2.pdb",
         "ligand": "/abs/path/ligand.sdf"},
        ...
    ]}

Paths are written ABSOLUTE so the loader is independent of the process CWD.
The receptor feature dim (1281 for DiffDock-L) is NOT baked in here — it is
derived at runtime from the first featurized sample (see train_entry.py).
"""

import argparse
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTAC_DIR = _PROJECT_ROOT / "DeepTernary" / "output" / "protac22"
DEFAULT_OUT = _PROJECT_ROOT / "data" / "protac_index.json"


def _resolve_ligand(case_dir: Path):
    """Prefer ligand.sdf (processed 3D conformer); fall back to ligand.pdb."""
    for name in ("ligand.sdf", "ligand.pdb"):
        p = case_dir / name
        if p.exists():
            return str(p)
    return None


def build_index(protac_dir: Path):
    entries = []
    skipped = []
    for case_dir in sorted(protac_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        protein1 = case_dir / "protein1.pdb"
        protein2 = case_dir / "protein2.pdb"
        ligand = _resolve_ligand(case_dir)
        if not (protein1.exists() and protein2.exists() and ligand):
            skipped.append(case_dir.name)
            continue
        entries.append({
            "case": case_dir.name,
            "protein1": str(protein1),
            "protein2": str(protein2),
            "ligand": ligand,
        })
    return entries, skipped


def main():
    p = argparse.ArgumentParser(description="Build PROTAC training index")
    p.add_argument("--protac-dir", default=str(DEFAULT_PROTAC_DIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    protac_dir = Path(args.protac_dir)
    if not protac_dir.is_dir():
        raise SystemExit(f"FATAL: protac dir not found: {protac_dir}")

    entries, skipped = build_index(protac_dir)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"entries": entries}, f, indent=2)

    print(f"Wrote {len(entries)} entries to {out}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete dirs: {skipped}")


if __name__ == "__main__":
    main()
