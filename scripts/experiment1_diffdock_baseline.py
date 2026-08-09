#!/usr/bin/env python3
"""
Experiment 1: Two-body DiffDock baseline on PROTAC benchmark.

Runs DiffDock inference on each PROTAC case split into two two-body pairs:
  - E3 + PROTAC  (protein1.pdb + ligand.sdf)
  - POI + PROTAC (protein2.pdb + ligand.sdf)

10 samples per pair. Metrics: DockQ, RMSD, clash rate.
Output: results/diffdock_baseline.json

ESM2-650M embeddings are generated on-the-fly via HF mirror to avoid
network restrictions on Facebook's dl.fbaipublicfiles.com.

Usage:
    python scripts/experiment1_diffdock_baseline.py --device cuda
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import torch
import yaml
from argparse import Namespace
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── Pure-PyTorch fallback for torch_cluster / torch_scatter ──────────────────
import torch_cluster
import torch_scatter


def _radius_pytorch(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32,
                    batch_size=None):
    """Replacement for torch_cluster.radius.

    torch_cluster.radius convention: "For each element in y, find all points
    in x within distance r."  Returns [y_indices, x_indices], i.e.:
      edge_index[0] = indices into y (the centers / query points)
      edge_index[1] = indices into x (the neighbors found)
    """
    if batch_x is None:
        batch_x = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    edge_index_list = []
    for bx in batch_x.unique():
        mask_x = (batch_x == bx)
        idx_x = mask_x.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask_x]
        for by_ in batch_y.unique():
            if bx != by_:
                continue
            mask_y = (batch_y == by_)
            idx_y = mask_y.nonzero(as_tuple=False).squeeze(-1)
            yj = y[mask_y]
            block_size = 1024
            for a in range(0, xi.shape[0], block_size):
                a_end = min(a + block_size, xi.shape[0])
                xi_block = xi[a:a_end]
                for b in range(0, yj.shape[0], block_size):
                    b_end = min(b + block_size, yj.shape[0])
                    yj_block = yj[b:b_end]
                    dist_block = torch.cdist(xi_block, yj_block)
                    rows, cols = (dist_block <= r).nonzero(as_tuple=True)
                    if rows.numel() > 0:
                        # torch_cluster convention: [y_idx, x_idx]
                        edge_index_list.append(
                            torch.stack([idx_y[b + cols], idx_x[a + rows]], dim=0))
    if not edge_index_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(edge_index_list, dim=1).int()


def _radius_graph_pytorch(x, r, batch=None, max_num_neighbors=32,
                          flow='source_to_target', loop=False, max_radius=None):
    """Replacement for torch_cluster.radius_graph.

    Follows torch_cluster convention: calls radius(x,x,r), then swaps
    indices based on flow direction.
    """
    assert flow in ['source_to_target', 'target_to_source']
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    # radius(x,x,r) returns [center_y_idx, neighbor_x_idx]
    ei = _radius_pytorch(x, x, r, batch, batch, max_num_neighbors)
    if flow == 'source_to_target':
        # Message flows from neighbor (source) to center (target)
        row, col = ei[1], ei[0]  # swap
    else:
        row, col = ei[0], ei[1]  # keep
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0).int()


def _knn_graph_pytorch(x, k, batch=None, loop=False, flow='source_to_target',
                       cosine=False, num_workers=1):
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    ei_list = []
    for b in batch.unique():
        mask = (batch == b)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        xi = x[mask]
        dist = torch.cdist(xi, xi)
        if not loop:
            dist.diagonal().fill_(float('inf'))
        kk = min(k, xi.shape[0])
        _, topk = dist.topk(kk, dim=1, largest=False)
        rows = torch.arange(xi.shape[0], device=x.device).unsqueeze(1).expand(-1, kk).reshape(-1)
        cols = topk.reshape(-1)
        valid = cols < xi.shape[0]
        rows, cols = rows[valid], cols[valid]
        ei_list.append(torch.stack([idx[rows], idx[cols]], dim=0))
    if not ei_list:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    return torch.cat(ei_list, dim=1).int()  # int32 for torch.bincount


def _scatter_pytorch(src, index, dim_size=None, dim=0, reduce='sum'):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    if index.numel() == 0:
        shape = list(src.shape)
        shape[dim] = dim_size
        return torch.zeros(shape, dtype=src.dtype, device=src.device)
    if reduce == 'sum':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        return out
    elif reduce == 'mean':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
        count.index_add_(0, index, torch.ones(src.shape[0], dtype=src.dtype, device=src.device))
        return out / count.clamp(min=1).unsqueeze(-1)
    elif reduce == 'max':
        out = torch.full((dim_size, *src.shape[1:]), float('-inf'),
                        dtype=src.dtype, device=src.device)
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax')
    elif reduce == 'min':
        out = torch.full((dim_size, *src.shape[1:]), float('inf'),
                        dtype=src.dtype, device=src.device)
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, reduce='amin')
    else:
        raise ValueError(f"Unknown reduce: {reduce}")


def _scatter_mean_pytorch(src, index, dim_size=None, dim=0):
    return _scatter_pytorch(src, index, dim_size, dim, reduce='mean')


torch_cluster.radius = _radius_pytorch
torch_cluster.radius_graph = _radius_graph_pytorch
torch_cluster.knn_graph = _knn_graph_pytorch
torch_scatter.scatter = _scatter_pytorch
torch_scatter.scatter_mean = _scatter_mean_pytorch
print("[MONKEY-PATCH] torch_cluster + torch_scatter → pure-PyTorch fallbacks")

# ── Path setup ────────────────────────────────────────────────────────────────
_COOPDIFF = str(Path(__file__).parent.parent)
if _COOPDIFF not in sys.path:
    sys.path.insert(0, _COOPDIFF)

_DIFFDOCK = str(Path(__file__).parent.parent.parent / "DiffDock")
if _DIFFDOCK not in sys.path:
    sys.path.insert(0, _DIFFDOCK)

_DEEPT = str(Path(__file__).parent.parent / "DeepTernary")
if _DEEPT not in sys.path:
    sys.path.insert(0, _DEEPT)

from utils.inference_utils import InferenceDataset, get_sequences
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from rdkit.Chem import RemoveAllHs
from datasets.process_mols import write_mol_with_coords


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 1: Two-body DiffDock baseline")
    p.add_argument("--protac-dir", default="DeepTernary/output/protac22")
    p.add_argument("--model-dir", default="checkpoints/score_model")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--out-dir", default="results/diffdock_baseline")
    p.add_argument("--output-json", default="results/diffdock_baseline.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--inference-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--cases", nargs="*", default=None,
                   help="Specific cases to run (default: all)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cases with existing output SDFs")
    return p


def generate_esm_embeddings(protein_paths, device):
    """Generate ESM2-650M embeddings for a list of protein PDB files.

    Uses DiffDock's get_sequences() for chain extraction (ProDy-based,
    matching moad_extract_receptor_structure order) and HF transformers
    via esm_embedder for actual embedding generation.

    Returns:
        dict: str (protein_path) -> list of torch.Tensor (one per chain),
              each shape (n_residues, 1280).
    """
    from tricoopdiff.esm_embedder import ESMEmbedder

    print(f"Loading ESM2-650M on {device} via HF mirror...", flush=True)
    embedder = ESMEmbedder(device=device)

    # Step 1: Extract sequences using DiffDock's ProDy-based function
    # Format: ['SEQ1:SEQ2'] per protein (chains joined by ':')
    print(f"Extracting sequences from {len(protein_paths)} proteins...",
          flush=True)
    seqs_list = get_sequences(protein_paths, [None] * len(protein_paths))

    # Step 2: Collect all unique chain sequences for batched embedding
    all_chain_seqs = []
    per_protein_info = []  # (path, [chain_seq, ...])

    for path, seq_str in zip(protein_paths, seqs_list):
        chain_seqs = seq_str.split(":")
        per_protein_info.append((path, chain_seqs))
        all_chain_seqs.extend(chain_seqs)

    # Deduplicate sequences for efficiency
    unique_seqs = list(dict.fromkeys(all_chain_seqs))
    print(f"  {len(all_chain_seqs)} chains total, "
          f"{len(unique_seqs)} unique sequences", flush=True)

    # Step 3: Generate embeddings for unique sequences
    print(f"Generating ESM2 embeddings...", flush=True)
    unique_embeddings = embedder.embed_from_fasta(
        unique_seqs, batch_size=4)

    seq_to_emb = dict(zip(unique_seqs, unique_embeddings))

    # Step 4: Build result map
    result = {}
    for path, chain_seqs in per_protein_info:
        result[path] = [seq_to_emb[s].cpu() for s in chain_seqs]

    total_res = sum(
        sum(t.shape[0] for t in chain_list)
        for chain_list in result.values()
    )
    print(f"Embeddings ready: {len(result)} proteins, "
          f"{total_res} total residues", flush=True)
    return result


def build_complex_pdb(protein_path, ligand_pos, ligand_mol, output_path):
    """Write a complex PDB with protein + docked ligand for DockQ.

    Uses BioPython to read the protein, adds the ligand as HETATM,
    and writes the combined PDB.
    """
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    import io

    # Read protein
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", protein_path)
    io_pdb = PDBIO()
    io_pdb.set_structure(structure)

    # Write protein part
    protein_str = io.StringIO()
    io_pdb.save(protein_str)
    protein_lines = protein_str.getvalue().splitlines()

    # Generate ligand HETATM lines
    ligand_lines = []
    if hasattr(ligand_mol, 'GetConformer'):
        conf = ligand_mol.GetConformer()
        atom_idx = 0
        serial = 10000
        for atom in ligand_mol.GetAtoms():
            pos_atom = conf.GetAtomPosition(atom_idx)
            x, y, z = pos_atom.x, pos_atom.y, pos_atom.z
            ligand_lines.append(
                f"HETATM{serial:5d}  {atom.GetSymbol():<3s} UNK X   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{atom.GetSymbol():>2s}  "
            )
            serial += 1
            atom_idx += 1

    # Write combined PDB (protein TER + ligand + END)
    with open(output_path, "w") as f:
        for line in protein_lines:
            if line.startswith("END"):
                continue
            if line.startswith("TER"):
                f.write(line + "\n")
                break
            f.write(line + "\n")
        for line in ligand_lines:
            f.write(line + "\n")
        f.write("TER\n")
        f.write("END\n")


def compute_dockq(model_pdb, native_pdb):
    """Compute DockQ between predicted complex and native complex."""
    from DockQ.dockq_util import cal_dockq
    try:
        fnat, irms, Lrms, dockq = cal_dockq(model_pdb, native_pdb)
        return {
            "dockq": float(dockq),
            "fnat": float(fnat),
            "irms": float(irms),
            "Lrms": float(Lrms),
        }
    except Exception as e:
        print(f"  [WARN] DockQ failed: {e}", flush=True)
        return {"dockq": 0.0, "fnat": 0.0, "irms": 0.0, "Lrms": 0.0}


def compute_rmsd(ligand_pos, native_ligand_pos):
    """Kabsch-aligned RMSD between two sets of ligand coordinates."""
    if len(ligand_pos) != len(native_ligand_pos):
        return float("inf")
    # Center both
    c1 = ligand_pos - ligand_pos.mean(axis=0)
    c2 = native_ligand_pos - native_ligand_pos.mean(axis=0)
    # Kabsch via SVD
    H = c1.T @ c2
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        # SVD can fail for degenerate point clouds (e.g., nearly collinear atoms)
        # Fall back to unaligned RMSD
        return float(np.sqrt(((c1 - c2) ** 2).sum(axis=1).mean()))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = c1 @ R
    return float(np.sqrt(((aligned - c2) ** 2).sum(axis=1).mean()))


def compute_clash_rate(protein_coords, ligand_coords, cutoff=3.0):
    """Fraction of ligand atoms that clash with protein atoms."""
    from scipy.spatial import KDTree
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        return 1.0
    tree = KDTree(protein_coords)
    dists, _ = tree.query(ligand_coords, distance_upper_bound=cutoff + 0.1)
    valid = dists < float("inf")
    if valid.sum() == 0:
        return 0.0
    return float((dists[valid] < cutoff).sum() / len(ligand_coords))


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Started: {datetime.now()}", flush=True)

    protac_dir = Path(args.protac_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover cases
    all_cases = sorted(
        d for d in os.listdir(protac_dir)
        if os.path.isdir(protac_dir / d)
    )
    if args.cases:
        cases = [c for c in args.cases if c in all_cases]
    else:
        cases = all_cases
    print(f"Cases to process: {len(cases)}", flush=True)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Load DiffDock model
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60, flush=True)
    print("Step 1: Loading DiffDock-L model", flush=True)
    print("=" * 60, flush=True)

    model_dir = args.model_dir
    with open(f"{model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))
    print(f"  ns={score_model_args.ns}, nv={score_model_args.nv}, "
          f"num_conv_layers={score_model_args.num_conv_layers}, "
          f"sh_lmax={score_model_args.sh_lmax}", flush=True)

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma,
                      no_parallel=True, old=False)
    ckpt_path = f"{model_dir}/{args.ckpt}"
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("  Model loaded.", flush=True)

    tr_schedule = get_t_schedule(
        inference_steps=args.inference_steps, sigma_schedule="expbeta")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: Generate ESM2 embeddings for all proteins
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60, flush=True)
    print("Step 2: Generating ESM2 embeddings", flush=True)
    print("=" * 60, flush=True)

    protein_paths = sorted(set(
        str(protac_dir / c / f"{p}.pdb")
        for c in cases
        for p in ["protein1", "protein2"]
    ))
    embeddings_map = generate_esm_embeddings(protein_paths, str(device))

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Run inference for each case × pair
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60, flush=True)
    print("Step 3: Running inference", flush=True)
    print("=" * 60, flush=True)

    per_case_results = []

    for case_name in tqdm(cases, desc="Cases"):
        for pair_name, protein_name in [
            ("e3l", "protein1"),
            ("poil", "protein2"),
        ]:
            pair_label = f"{case_name}/{pair_name}"
            protein_path = str(protac_dir / case_name / f"{protein_name}.pdb")
            ligand_path = str(protac_dir / case_name / "ligand.sdf")
            gt_complex_path = str(protac_dir / case_name / "complex.pdb")
            gt_ligand_path = str(protac_dir / case_name / "ligand.pdb")

            write_dir = out_dir / case_name / pair_name
            if args.skip_existing and (write_dir / "rank1.sdf").exists():
                print(f"  {pair_label}: skip (existing)", flush=True)
                continue
            write_dir.mkdir(parents=True, exist_ok=True)

            # Build precomputed embeddings in InferenceDataset format
            chain_embs = embeddings_map[protein_path]
            precomputed = [[t for t in chain_embs]]  # keep on CPU; moad_extract_receptor_structure uses np.concatenate

            try:
                # Create dataset and graph
                test_dataset = InferenceDataset(
                    out_dir=str(write_dir),
                    complex_names=[pair_label],
                    protein_files=[protein_path],
                    ligand_descriptions=[ligand_path],
                    protein_sequences=[None],
                    lm_embeddings=True,
                    precomputed_lm_embeddings=precomputed,
                    receptor_radius=score_model_args.receptor_radius,
                    remove_hs=score_model_args.remove_hs,
                    c_alpha_max_neighbors=(
                        score_model_args.c_alpha_max_neighbors),
                    all_atoms=score_model_args.all_atoms,
                    atom_radius=score_model_args.atom_radius,
                    atom_max_neighbors=score_model_args.atom_max_neighbors,
                    knn_only_graph=(
                        False if not hasattr(score_model_args, "not_knn_only_graph")
                        else not score_model_args.not_knn_only_graph
                    ),
                )
                test_loader = DataLoader(
                    dataset=test_dataset, batch_size=1, shuffle=False)
                orig_graph = next(iter(test_loader))
                # Keep on CPU — sampling() moves to device internally (line 94)

                if not orig_graph.success[0]:
                    print(f"  {pair_label}: SKIP (preprocessing failed)",
                          flush=True)
                    per_case_results.append({
                        "name": pair_label,
                        "error": "preprocessing_failed",
                    })
                    continue

                # Prepare N data copies and randomize
                N = args.samples
                data_list = [copy.deepcopy(orig_graph) for _ in range(N)]
                randomize_position(
                    data_list,
                    score_model_args.no_torsion,
                    False,
                    score_model_args.tr_sigma_max,
                    initial_noise_std_proportion=-1.0,
                    choose_residue=False,
                )

                # Run reverse diffusion
                data_list, _ = sampling(
                    data_list=data_list,
                    model=model,
                    inference_steps=args.inference_steps,
                    tr_schedule=tr_schedule,
                    rot_schedule=tr_schedule,
                    tor_schedule=tr_schedule,
                    device=device,
                    t_to_sigma=t_to_sigma,
                    model_args=score_model_args,
                    confidence_model=None,
                    confidence_data_list=None,
                    confidence_model_args=None,
                    batch_size=min(args.batch_size, N),
                    no_final_step_noise=True,
                )

                # Extract predicted ligand positions
                lig = orig_graph.mol[0]
                ligand_pos_list = [
                    g["ligand"].pos.cpu().numpy()
                    + orig_graph.original_center.cpu().numpy()
                    for g in data_list
                ]

                # Save SDF files
                for rank, pos in enumerate(ligand_pos_list):
                    mol_pred = copy.deepcopy(lig)
                    if score_model_args.remove_hs:
                        mol_pred = RemoveAllHs(mol_pred)
                    write_mol_with_coords(
                        mol_pred, pos,
                        os.path.join(str(write_dir), f"rank{rank+1}.sdf"))

                # Save PDB files (complex: protein + predicted ligand)
                for rank, pos in enumerate(ligand_pos_list):
                    mol_pred = copy.deepcopy(lig)
                    if score_model_args.remove_hs:
                        mol_pred = RemoveAllHs(mol_pred)
                    pdb_out = str(write_dir / f"rank{rank+1}.pdb")
                    build_complex_pdb(protein_path, pos, mol_pred, pdb_out)

                # ═══════════════════════════════════════════════════════════
                # Compute metrics for each sample
                # ═══════════════════════════════════════════════════════════
                # Load native ligand coords for RMSD
                native_lig_coords = None
                if os.path.exists(gt_ligand_path):
                    from Bio.PDB import PDBParser
                    parser = PDBParser(QUIET=True)
                    native_struct = parser.get_structure(
                        "native", gt_ligand_path)
                    native_atoms = list(native_struct.get_atoms())
                    native_lig_coords = np.array(
                        [a.get_coord() for a in native_atoms])

                # Load protein coords for clash
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                prot_struct = parser.get_structure("prot", protein_path)
                prot_coords = np.array(
                    [a.get_coord() for a in prot_struct.get_atoms()])

                sample_results = []
                for rank, pos in enumerate(ligand_pos_list):
                    # RMSD
                    rmsd = float("nan")
                    if native_lig_coords is not None and len(pos) == len(native_lig_coords):
                        rmsd = compute_rmsd(pos, native_lig_coords)

                    # Clash rate
                    clash = compute_clash_rate(prot_coords, pos)

                    # DockQ
                    pdb_out = str(write_dir / f"rank{rank+1}.pdb")
                    dockq_result = {"dockq": 0.0, "fnat": 0.0,
                                    "irms": 0.0, "Lrms": 0.0}
                    if os.path.exists(gt_complex_path):
                        dockq_result = compute_dockq(pdb_out, gt_complex_path)

                    sample_results.append({
                        "rank": rank + 1,
                        "dockq": dockq_result["dockq"],
                        "fnat": dockq_result["fnat"],
                        "irms": dockq_result["irms"],
                        "Lrms": dockq_result["Lrms"],
                        "rmsd": rmsd,
                        "clash_rate": clash,
                    })

                # Best DockQ sample
                best = max(sample_results, key=lambda r: r["dockq"])
                per_case_results.append({
                    "name": pair_label,
                    "case": case_name,
                    "pair": pair_name,
                    "dockq": best["dockq"],
                    "fnat": best["fnat"],
                    "irms": best["irms"],
                    "Lrms": best["Lrms"],
                    "rmsd": best["rmsd"],
                    "clash_rate": best["clash_rate"],
                    "samples": sample_results,
                })

                print(f"  {pair_label}: best DockQ={best['dockq']:.4f}, "
                      f"RMSD={best['rmsd']:.2f}, "
                      f"clash={best['clash_rate']:.3f}",
                      flush=True)

            except Exception as e:
                print(f"  {pair_label}: ERROR: {e}", flush=True)
                traceback.print_exc()
                per_case_results.append({
                    "name": pair_label,
                    "error": str(e),
                })

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Save results
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60, flush=True)
    print("Step 4: Saving results", flush=True)
    print("=" * 60, flush=True)

    # Aggregate stats
    successful = [r for r in per_case_results if "error" not in r]
    e3l_results = [r for r in successful if r["pair"] == "e3l"]
    poil_results = [r for r in successful if r["pair"] == "poil"]

    def agg_stats(results, key):
        vals = [r[key] for r in results if key in r and not (
            isinstance(r[key], float) and (
                np.isnan(r[key]) or np.isinf(r[key])))]
        if not vals:
            return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    results = {
        "model": "DiffDock-L v1.1",
        "date": datetime.now().isoformat(),
        "ns": score_model_args.ns,
        "nv": score_model_args.nv,
        "num_conv_layers": score_model_args.num_conv_layers,
        "sh_lmax": score_model_args.sh_lmax,
        "n_cases": len(cases),
        "n_samples_per_pair": args.samples,
        "e3l": {
            "n": len(e3l_results),
            "dockq": agg_stats(e3l_results, "dockq"),
            "rmsd": agg_stats(e3l_results, "rmsd"),
            "clash_rate": agg_stats(e3l_results, "clash_rate"),
            "fnat": agg_stats(e3l_results, "fnat"),
        },
        "poil": {
            "n": len(poil_results),
            "dockq": agg_stats(poil_results, "dockq"),
            "rmsd": agg_stats(poil_results, "rmsd"),
            "clash_rate": agg_stats(poil_results, "clash_rate"),
            "fnat": agg_stats(poil_results, "fnat"),
        },
        "overall": {
            "n": len(successful),
            "dockq": agg_stats(successful, "dockq"),
            "rmsd": agg_stats(successful, "rmsd"),
            "clash_rate": agg_stats(successful, "clash_rate"),
            "fnat": agg_stats(successful, "fnat"),
        },
        "per_case": per_case_results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {args.output_json}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for label, r in [("E3-L  ", e3l_results), ("POI-L ", poil_results),
                     ("Overall", successful)]:
        if not r:
            continue
        d = agg_stats(r, "dockq")
        rm = agg_stats(r, "rmsd")
        c = agg_stats(r, "clash_rate")
        print(f"  {label}: n={len(r)}, "
              f"DockQ={d['mean']:.4f}±{d['std']:.4f}, "
              f"RMSD={rm['mean']:.2f}±{rm['std']:.2f}, "
              f"Clash={c['mean']:.3f}±{c['std']:.3f}",
              flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
