"""Compare DeepTernary test IDs with LigandPro/Bento training annotations."""

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bento-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    datasets = args.bento_root / "datasets"
    training_paths = read_tsv(datasets / "path_train.tsv")
    pdbbind_uids = {
        row["uid"] for row in training_paths if row["dataset"] == "PDBBind_train"
    }
    moad_uids = {
        row["uid"] for row in training_paths if row["dataset"] == "MOAD_train"
    }
    all_train_pdbs = {row["uid"][:4].lower() for row in training_paths}
    annotated = [
        ("PDBBind_train", row)
        for row in read_tsv(datasets / "pdbbind_train.tsv")
        if row["uid"] in pdbbind_uids
    ]
    annotated.extend(
        ("MOAD_train", row)
        for row in read_tsv(datasets / "moad.tsv")
        if row["uid"] in moad_uids
    )

    by_pdb = defaultdict(list)
    by_ligand = defaultdict(set)
    for source, row in annotated:
        pdb_id = row["pdb"].lower()
        ligands = ast.literal_eval(row["ligand"])
        by_pdb[pdb_id].append((source, row["uid"], ligands))
        for ligand in ligands:
            by_ligand[ligand.upper()].add(pdb_id)

    root = Path(__file__).resolve().parents[1]
    rows = []
    for benchmark, filename in (("MGD", "test.txt"), ("PROTAC", "test.csv")):
        path = root / "DeepTernary" / "data" / "TernaryDB" / benchmark / filename
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            case = fields[0]
            pdb_id = case[:4].lower()
            ligand = case.rsplit("_", 1)[-1].upper()
            same_pdb = by_pdb[pdb_id]
            exact = [
                f"{source}:{uid}"
                for source, uid, ligands in same_pdb
                if ligand in (value.upper() for value in ligands)
            ]
            other_pdbs = sorted(by_ligand[ligand] - {pdb_id})
            unbound_ids = [field.split("-", 1)[0].lower() for field in fields[1:3]]
            rows.append({
                "benchmark": benchmark,
                "case": case,
                "pdb_id": pdb_id,
                "ligand_ccd": ligand,
                "same_pdb_train": pdb_id in all_train_pdbs,
                "same_pdb_ligand_train": bool(exact),
                "exact_training_rows": ";".join(exact),
                "same_ligand_other_pdb_count": len(other_pdbs),
                "same_ligand_other_pdb_examples": ";".join(other_pdbs[:10]),
                "binary_source_1_in_train": bool(unbound_ids) and unbound_ids[0] in all_train_pdbs,
                "binary_source_2_in_train": len(unbound_ids) > 1 and unbound_ids[1] in all_train_pdbs,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for benchmark in ("MGD", "PROTAC"):
        subset = [row for row in rows if row["benchmark"] == benchmark]
        exact = sum(row["same_pdb_ligand_train"] for row in subset)
        same_ligand = sum(row["same_ligand_other_pdb_count"] > 0 for row in subset)
        print(f"{benchmark}: {len(subset)} cases; exact PDB+ligand {exact}; "
              f"same ligand in another PDB {same_ligand}")


if __name__ == "__main__":
    main()
