"""Score archived MGD Final poses with Smina/Vina's fixed classical score."""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import Model, PDBIO, PDBParser, Structure
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
OBABEL = "/home/wl/micromamba/envs/ob2/bin/obabel"
SMINA = "/home/wl/micromamba/envs/smina/bin/smina"


def score_case(task):
    archive_path, entry, out_path = task
    archive_path, out_path = Path(archive_path), Path(out_path)
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    parser = PDBParser(QUIET=True)
    protein1 = parser.get_structure("p1", entry["protein1"])
    ligand_path = Path(entry["ligand"])
    if ligand_path.suffix.lower() == ".sdf":
        ligand_template = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
    else:
        ligand_template = Chem.MolFromPDBFile(str(ligand_path), removeHs=False, sanitize=False)
    if ligand_template is None:
        raise ValueError(f"cannot parse ligand template: {ligand_path}")
    results = []
    for pair in archive["pairs"]:
        if not pair["valid"]:
            results.append({"pool_index": pair["pool_index"], "valid": False, "score": None})
            continue
        with tempfile.TemporaryDirectory(prefix="tricoopdiff_smina_") as temp:
            temp = Path(temp)
            protein2 = parser.get_structure("p2", entry["protein2"])
            model2 = protein2[0]
            ca = np.asarray([atom.coord for atom in model2.get_atoms() if atom.get_name() == "CA"])
            target = pair["final"]["poi"].numpy()
            source_center, target_center = ca.mean(0), target.mean(0)
            u, _, v = np.linalg.svd((ca - source_center).T @ (target - target_center))
            determinant = np.linalg.det(u @ v)
            rotation = u @ np.diag([1.0, 1.0, determinant]) @ v
            translation = target_center - source_center @ rotation
            for atom in model2.get_atoms():
                atom.coord = atom.coord @ rotation + translation

            complex_structure = Structure.Structure("complex")
            model = Model.Model(0)
            complex_structure.add(model)
            for structure, chain_id in ((protein1[0], "A"), (model2, "B")):
                for chain in structure:
                    chain.id = chain_id
                    model.add(chain.copy())
            receptor_pdb = temp / "receptor.pdb"
            writer = PDBIO()
            writer.set_structure(complex_structure)
            writer.save(str(receptor_pdb))

            ligand = Chem.Mol(ligand_template)
            conformer = ligand.GetConformer()
            coordinates = pair["final"]["ligand"].numpy()
            for index in range(ligand.GetNumAtoms()):
                conformer.SetAtomPosition(index, tuple(float(x) for x in coordinates[index]))
            ligand_pdb = temp / "ligand.pdb"
            Chem.MolToPDBFile(ligand, str(ligand_pdb), flavor=4)
            receptor_pdbqt, ligand_pdbqt = temp / "receptor.pdbqt", temp / "ligand.pdbqt"
            subprocess.run([OBABEL, str(receptor_pdb), "-O", str(receptor_pdbqt), "-xr"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run([OBABEL, str(ligand_pdb), "-O", str(ligand_pdbqt), "-h"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            completed = subprocess.run(
                [SMINA, "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt),
                 "--score_only", "--cpu", "1"], capture_output=True, text=True, check=True)
            match = re.search(r"Affinity:\s+([-+0-9.eE]+)", completed.stdout)
            if match is None:
                raise RuntimeError(f"Smina did not report affinity for {archive['case']}")
            results.append({"pool_index": pair["pool_index"], "valid": True,
                            "score": float(match.group(1))})
    out_path.write_text(json.dumps({"case": archive["case"], "archive_sha256": hashlib.sha256(
        archive_path.read_bytes()).hexdigest(), "scores": results}, indent=2) + "\n")
    return archive["case"], len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    entries = {item["case"]: item for item in json.loads(args.labels.read_text())["entries"]}
    tasks = []
    for archive_path in sorted(args.archives.glob("*.pt")):
        case = archive_path.stem
        if case not in entries:
            raise ValueError(f"missing reference entry: {case}")
        tasks.append((str(archive_path), entries[case], str(args.out / f"{case}.json")))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, task) for task in tasks]
        for future in as_completed(futures):
            print("SCORED", *future.result(), flush=True)
    (args.out / "complete.json").write_text(json.dumps({"cases": len(tasks)}) + "\n")


if __name__ == "__main__":
    main()
