"""Score archived MGD Final poses with Rosetta ref2015 total energy."""

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


ROSETTA_PY = "/home/wl/micromamba/envs/roshambo/bin/python"
PARAMS = "/mnt/disk/wl/tools/rosetta/source/scripts/python/public/molfile_to_params.py"
SCORE = "/home/wl/micromamba/envs/rosetta_bin/bin/rosetta_scripts.cxx11threadserialization.linuxclangrelease"
PROTOCOL = str(Path(__file__).resolve().with_name("rosetta_interface_mgd.xml"))


def rewrite_ligand(template, coordinates, destination):
    lines = template.read_text().splitlines(True)
    atom_lines = [i for i, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    if len(atom_lines) != len(coordinates):
        raise ValueError(f"ligand atom count mismatch: {len(atom_lines)} != {len(coordinates)}")
    for line_index, xyz in zip(atom_lines, coordinates):
        line = lines[line_index]
        lines[line_index] = f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"
    destination.write_text("".join(lines))


def score_case(task):
    archive_path, entry, out_path = Path(task[0]), task[1], Path(task[2])
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    parser = PDBParser(QUIET=True)
    protein1 = parser.get_structure("p1", entry["protein1"])
    ligand_path = Path(entry["ligand"])
    if ligand_path.suffix.lower() == ".sdf":
        ligand = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
    else:
        ligand = Chem.MolFromPDBFile(str(ligand_path), removeHs=False, sanitize=False)
    if ligand is None:
        ligand = Chem.RemoveHs(Chem.Mol(archive["input_graphs"]["ligand_data"]["mol"]))
    ligand = Chem.RemoveHs(ligand)
    results = []
    with tempfile.TemporaryDirectory(prefix="tricoopdiff_rosetta_case_") as root:
        root = Path(root)
        ligand_sdf = root / "ligand.sdf"
        writer = Chem.SDWriter(str(ligand_sdf)); writer.write(ligand); writer.close()
        subprocess.run([ROSETTA_PY, PARAMS, "-n", "LIG", "-p", str(root / "LIG"), str(ligand_sdf)],
                       cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ligand_template = root / "LIG_0001.pdb"
        params = root / "LIG.params"
        for pair in archive["pairs"]:
            if not pair["valid"]:
                results.append({"pool_index": pair["pool_index"], "valid": False, "score": None})
                continue
            protein2 = parser.get_structure("p2", entry["protein2"])[0]
            ca = np.asarray([atom.coord for atom in protein2.get_atoms() if atom.get_name() == "CA"])
            target = pair["final"]["poi"].numpy()
            source_center, target_center = ca.mean(0), target.mean(0)
            u, _, v = np.linalg.svd((ca - source_center).T @ (target - target_center))
            rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ v)]) @ v
            translation = target_center - source_center @ rotation
            for atom in protein2.get_atoms():
                atom.coord = atom.coord @ rotation + translation
            complex_structure = Structure.Structure("complex")
            model = Model.Model(0); complex_structure.add(model)
            for structure, chain_id in ((protein1[0], "A"), (protein2, "B")):
                for chain in structure:
                    chain.id = chain_id; model.add(chain.copy())
            protein_pdb = root / "protein.pdb"; io = PDBIO(); io.set_structure(complex_structure); io.save(str(protein_pdb))
            ligand_pdb = root / "ligand_pose.pdb"
            rewrite_ligand(ligand_template, pair["final"]["ligand"].numpy(), ligand_pdb)
            complex_pdb = root / "complex.pdb"
            complex_pdb.write_text(protein_pdb.read_text() + ligand_pdb.read_text())
            scorefile = root / "score.sc"
            completed = subprocess.run([SCORE, "-s", str(complex_pdb), "-extra_res_fa", str(params),
                                       "-parser:protocol", PROTOCOL, "-out:file:scorefile", str(scorefile), "-overwrite",
                                       "-ignore_unrecognized_res"], cwd=root, capture_output=True, text=True, check=True)
            score_lines = [line for line in scorefile.read_text().splitlines() if line.startswith("SCORE:")]
            header = score_lines[0].split(); values = score_lines[-1].split()
            metrics = {}
            for index, name in enumerate(header):
                if index >= len(values) or name in ("SCORE:", "description"):
                    continue
                try:
                    metrics[name] = float(values[index])
                except ValueError:
                    continue
            if "dG_separated" not in metrics:
                raise RuntimeError("InterfaceAnalyzerMover did not emit dG_separated")
            results.append({"pool_index": pair["pool_index"], "valid": True,
                            "score": metrics["dG_separated"], "metrics": metrics})
    out_path.write_text(json.dumps({"case": archive["case"], "archive_sha256": hashlib.sha256(
        archive_path.read_bytes()).hexdigest(), "scores": results}, indent=2) + "\n")
    return archive["case"], len(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    entries = {item["case"]: item for item in json.loads(args.index.read_text())["entries"]}
    tasks = [(str(path), entries[path.stem], str(args.out / f"{path.stem}.json"))
             for path in sorted(args.archives.glob("*.pt"))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_case, task) for task in tasks]
        for future in as_completed(futures): print("SCORED", *future.result(), flush=True)
    (args.out / "complete.json").write_text(json.dumps({"cases": len(tasks)}) + "\n")


if __name__ == "__main__":
    main()
