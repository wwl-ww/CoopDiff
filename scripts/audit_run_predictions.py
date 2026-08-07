#!/usr/bin/env python
"""
DeepTernary Audit — Prediction Runner
=====================================
Runs 40-seed predictions for all PROTAC test cases and saves all output PDBs
permanently. Adapted from DeepTernary/predict_cpu.py.

Key differences from predict_cpu.py:
- Saves ALL predicted complex PDBs permanently (not in a temp dir)
- Outputs per-case summary CSV with all per-seed metrics
- Supports single-threaded and multi-process modes
- Graceful error handling: failed seeds are skipped, not fatal

Output structure (for each case):
    audit_results/{case_name}/
        complex_pred_{case_name}_{seed}.pdb   # 40 predicted ternary complexes
        summary_{case_name}.csv               # per-seed metrics table

Usage:
    python scripts/audit_run_predictions.py                        # all cases, single process
    python scripts/audit_run_predictions.py --case 5T35_D_A_759   # single case
    python scripts/audit_run_predictions.py --workers 4            # 4 processes
    python scripts/audit_run_predictions.py --cpu-only             # force CPU
    python scripts/audit_run_predictions.py --seeds 10             # fewer seeds for testing

Requirements:
    Must be run from CoopDiff/ root directory.
    DeepTernary must be installed as a package (pip install -e DeepTernary/ or equivalent).
"""

import argparse
import os
import sys
import json
import csv
import time
import traceback
from copy import deepcopy
from datetime import datetime
from functools import partial

import numpy as np
import scipy.spatial as spa
import torch
from biopandas.pdb import PandasPdb
from rdkit import Chem
from scipy.special import softmax
from tqdm import tqdm

# Allow running from CoopDiff/ root — add DeepTernary to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEEPTERNARY_ROOT = os.path.join(_PROJECT_ROOT, 'DeepTernary')
sys.path.insert(0, _DEEPTERNARY_ROOT)

import dgl
from mmengine.config import Config

# DeepTernary imports
from deepternary.models.correct import correct_ligand
from deepternary.models.geometry_utils import random_rotation_translation
from deepternary.models.logger import log as dt_log
from deepternary.models.path import IDEAL_PATH, PDB_PATH
from deepternary.models.pdb_utils import (
    merge_pdbs,
    set_new_chain,
    set_new_coords,
    write_pdb,
)
from deepternary.models.process_mols import (
    distance_featurizer,
    get_geometry_graph_ring,
    get_lig_graph_revised,
    get_rdkit_coords_v2,
    get_rec_graph,
    get_receptor_inference,
    lig_atom_featurizer,
    read_molecule,
    rigid_transform_Kabsch_3D,
)
from deepternary.models.rotate_utils import kabsch, rotate_and_translate
from deepternary.models.ternary_pdb import get_pocket_and_mask

# ── Global settings ──────────────────────────────────────────────────────────

FIX_TWO_ENDS = True   # Use fixed-two-ends RDKit conformer generation
CORRECT_LIGAND = True  # Run EquiBind ligand correction
USE_RDKIT_COORDS = True
UNBOUND = True

# Fallback set for molecules where RDKit embedding fails
_embedding_failed_names = set()


# ── Helper functions (adapted from predict_cpu.py) ────────────────────────────

def auto_download_ideal_sdf(lig_name):
    """Download ideal SDF from RCSB if not cached locally."""
    import requests
    ideal_sdf_path = os.path.join(IDEAL_PATH, f'{lig_name}_ideal.sdf')
    if not os.path.exists(ideal_sdf_path):
        os.makedirs(IDEAL_PATH, exist_ok=True)
        url = f'https://files.rcsb.org/ligands/download/{lig_name}_ideal.sdf'
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(ideal_sdf_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"  [WARN] Failed to download ideal SDF for {lig_name}: {e}")
            return None
    return ideal_sdf_path


def adjust_bounds(bounds_mat, atom_map, mol):
    """Adjust RDKit bounds matrix based on known atom positions."""
    from rdkit.Chem import rdDistGeom
    from rdkit.DistanceGeometry.DistGeom import DoTriangleSmoothing

    for i in range(len(atom_map)):
        atom_i = atom_map[i]
        for j in range(i + 1, len(atom_map)):
            atom_j = atom_map[j]
            a = min(atom_i, atom_j)
            b = max(atom_i, atom_j)
            pos1 = mol.GetConformer().GetAtomPosition(a)
            pos2 = mol.GetConformer().GetAtomPosition(b)
            distance = pos1.Distance(pos2)
            bounds_mat[a, b] = distance + 0.1
            bounds_mat[b, a] = distance - 0.1
    check = DoTriangleSmoothing(bounds_mat)
    return bounds_mat, check


def get_rdkit_coords_with_fixed_ends(ligand, lig1, lig2, seed=None):
    """Generate RDKit conformer with fixed warhead positions."""
    from rdkit.Chem import rdDistGeom

    params = rdDistGeom.ETKDGv3()
    ch1_map = ligand.GetSubstructMatch(lig1)
    ch2_map = ligand.GetSubstructMatch(lig2)
    if not (len(ch1_map) and len(ch2_map)):
        conf_id = rdDistGeom.EmbedMolecule(ligand, params)
    else:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(ligand)
        bounds, check1 = adjust_bounds(bounds, ch1_map, ligand)
        bounds, check2 = adjust_bounds(bounds, ch2_map, ligand)
        params.SetBoundsMat(bounds)
        if seed is not None:
            params.randomSeed = seed
        conf_ids = rdDistGeom.EmbedMultipleConfs(ligand, 1, params)
    ligand = Chem.RemoveHs(ligand)
    conf = ligand.GetConformer()
    lig_coords = conf.GetPositions()
    return ligand, lig_coords


def get_lig_graph_protac(mol, name, lig1=None, lig2=None, radius=20,
                         max_neighbors=None, use_rdkit_coords=False,
                         use_random_coords=True, seed=None):
    """Build PROTAC ligand graph with RDKit-generated coordinates."""
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()

    if use_rdkit_coords:
        if name in _embedding_failed_names:
            rdkit_coords = None
        else:
            try:
                mol, rdkit_coords = get_rdkit_coords_with_fixed_ends(
                    mol, lig1, lig2, seed=seed)
            except Exception as e:
                print(f"  [WARN] RDKit coords failed for {name}: {e}")
                rdkit_coords = None

        if rdkit_coords is None:
            print(f"  [INFO] {name}: falling back to ideal.sdf")
            _embedding_failed_names.add(name)
            ideal_path = auto_download_ideal_sdf(name.split('_')[-1])
            if ideal_path is None:
                raise RuntimeError(f"Cannot get ideal SDF for {name}")
            ideal_mol = read_molecule(ideal_path, sanitize=True, remove_hs=True)
            if ideal_mol is None:
                raise ValueError(f"ideal_mol is None for {ideal_path}")
            try:
                mol, rdkit_coords = get_rdkit_coords_with_fixed_ends(
                    deepcopy(ideal_mol), lig1, lig2, seed=seed)
            except Exception:
                mol, rdkit_coords = get_rdkit_coords_v2(
                    deepcopy(ideal_mol), seed=seed, use_random_coords=use_random_coords)
            if mol is None:
                mol = ideal_mol
                try:
                    rdkit_coords = ideal_mol.GetConformer().GetPositions()
                except Exception as e:
                    raise RuntimeError(f"ideal_mol {ideal_path} coords failed: {e}")
            if rdkit_coords.shape != true_lig_coords.shape:
                raise RuntimeError(
                    f"{name}: shape mismatch — rdkit {rdkit_coords.shape} vs true {true_lig_coords.shape}")

        R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
        lig_coords = (R @ (rdkit_coords).T).T + t.squeeze()
    else:
        lig_coords = true_lig_coords

    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list, dst_list, dist_list, mean_norm_list = [], [], [], []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors is not None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1:max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1.0, 2.0, 5.0, 10.0, 30.0]).reshape((-1, 1))
        weights = softmax(-valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]
        mean_vec = weights.dot(diff_vecs)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator
        mean_norm_list.append(mean_vec_ratio_norm)

    graph = dgl.graph(
        (torch.tensor(src_list), torch.tensor(dst_list)),
        num_nodes=num_nodes, idtype=torch.int32)
    graph.ndata["feat"] = lig_atom_featurizer(mol)
    graph.ndata["x"] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata["mu_r_norm"] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata["new_x"] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return mol, graph


def RMSD(pred, gt):
    """Compute root-mean-square deviation."""
    diff = pred - gt
    if isinstance(pred, torch.Tensor):
        return torch.sqrt(torch.sum(diff ** 2, dim=1).mean()).item()
    elif isinstance(pred, np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=1).mean())
    else:
        raise NotImplementedError


def get_lig_coords(lig_path):
    """Get ligand atom coordinates from PDB file."""
    lig = read_molecule(lig_path, sanitize=True, remove_hs=True)
    conf = lig.GetConformer()
    true_lig_coords = conf.GetPositions()
    return torch.from_numpy(true_lig_coords).float()


def replace_to_unbound_coords(whole_coords, part_coords):
    """Replace whole PROTAC atom coordinates with unbound fragment coords."""
    cdist = torch.cdist(whole_coords, part_coords)
    update_mask = cdist.min(dim=1)[0] < 1
    whole_coords[update_mask] = part_coords[cdist.argmin(dim=1)[update_mask]]
    return whole_coords, update_mask


def get_model(cfg, checkpoint_path, device='cuda'):
    """Load the trained DeepTernary model from checkpoint."""
    from mmpretrain.apis.model import get_model as _get_model
    model = _get_model(cfg, pretrained=checkpoint_path, device=device)
    model.to(device)
    model.device = device
    model.eval()
    return model


# ── Core prediction function ─────────────────────────────────────────────────

@torch.no_grad()
def predict_one_case(name, model, cfg, out_dir, seed_num=40, device='cuda'):
    """
    Run multi-seed prediction for one PROTAC case.
    Saves all predicted complex PDBs and per-seed metrics to out_dir.

    Returns:
        dict: {name, results: [{seed, fnat, irms, Lrms, dockq, lig_rmsd, ...}], success: bool}
    """
    ds_cfg = cfg.test_dataloader.dataset
    unboud_folder = 'DeepTernary/output/protac22'

    lig_path = os.path.join(unboud_folder, name, 'ligand.pdb')
    p1_path = os.path.join(unboud_folder, name, 'unbound_protein1.pdb')
    p2_path = os.path.join(unboud_folder, name, 'unbound_protein2.pdb')

    # Build receptor graphs (once — shared across seeds)
    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p1_path)
    p1_graph_origin = get_rec_graph(
        recs, recs_coords, c_alpha_coords, n_coords, c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p2_path)
    p2_graph_origin = get_rec_graph(
        recs, recs_coords, c_alpha_coords, n_coords, c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)
    protein2_coords_gt = deepcopy(p2_graph_origin.ndata['x'])

    if FIX_TWO_ENDS:
        lig1 = read_molecule(
            os.path.join(unboud_folder, name, 'unbound_lig1.pdb'),
            sanitize=True, remove_hs=True)
        lig2 = read_molecule(
            os.path.join(unboud_folder, name, 'unbound_lig2.pdb'),
            sanitize=True, remove_hs=True)

    results = []
    out_dir_case = os.path.join(out_dir, name)
    os.makedirs(out_dir_case, exist_ok=True)

    for seed in range(seed_num):
        try:
            lig = deepcopy(lig_origin)
            p1_graph = deepcopy(p1_graph_origin)
            p2_graph = deepcopy(p2_graph_origin)

            if FIX_TWO_ENDS:
                graph_func = partial(get_lig_graph_protac, lig1=lig1, lig2=lig2)
            else:
                graph_func = get_lig_graph_revised

            lig, lig_graph = graph_func(
                lig, name=name, max_neighbors=ds_cfg.lig_max_neighbors,
                use_random_coords=False, seed=seed,
                use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)
            geometry_graph = get_geometry_graph_ring(lig)

            lig_coords_gt = deepcopy(lig_graph.ndata['x'])

            # Unbound anchor/warhead replacement
            lig1_coords = get_lig_coords(os.path.join(unboud_folder, name, 'unbound_lig1.pdb'))
            lig2_coords = get_lig_coords(os.path.join(unboud_folder, name, 'unbound_lig2.pdb'))
            lig_coords_gt, update_mask1 = replace_to_unbound_coords(lig_coords_gt, lig1_coords)
            lig_coords_gt, update_mask2 = replace_to_unbound_coords(lig_coords_gt, lig2_coords)
            pocket_mask = update_mask1 | update_mask2

            # Pocket extraction
            p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = \
                get_pocket_and_mask(lig_coords_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
            p2lig_lig_pocket_coords_origin, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = \
                get_pocket_and_mask(lig_coords_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)

            p1lig_lig_pocket_mask = p1lig_lig_pocket_mask & pocket_mask
            p1lig_lig_pocket_coords = lig_coords_gt[p1lig_lig_pocket_mask]
            p1lig_p1_pocket_coords = deepcopy(p1lig_lig_pocket_coords)

            p2lig_lig_pocket_mask = p2lig_lig_pocket_mask & pocket_mask
            p2lig_lig_pocket_coords_origin = lig_coords_gt[p2lig_lig_pocket_mask]

            # Build data dict
            data = dict(
                lig_graph=lig_graph,
                rec_graph=p1_graph,
                rec2_graph=p2_graph,
                geometry_graph=geometry_graph,
                complex_name=[name],
                rec2_coords=[protein2_coords_gt],
                p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
                p1lig_p1_pocket_coords=[p1lig_p1_pocket_coords],
                p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
                p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
                p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
                p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords_origin],
                p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
                p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords_origin],
            )

            # Randomly perturb ligand position
            rot_T, rot_b = random_rotation_translation(translation_distance=5)
            if USE_RDKIT_COORDS:
                lig_coords_to_move = data['lig_graph'].ndata['new_x']
            else:
                lig_coords_to_move = data['lig_graph'].ndata['x']
            mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
            data['lig_graph'].ndata['new_x'] = (
                rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
            data['p1lig_lig_pocket_coords'][0] = (
                rot_T @ (data['p1lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b
            data['p2lig_lig_pocket_coords'][0] = (
                rot_T @ (data['p2lig_lig_pocket_coords'][0] - mean_to_remove).T).T + rot_b

            # Randomly perturb protein2 position
            protein2_rot_T, protein2_rot_b = random_rotation_translation(translation_distance=5)
            protein2_coord_to_move = data['rec2_graph'].ndata['x']
            protein2_mean_to_remove = protein2_coord_to_move.mean(dim=0, keepdims=True)
            data['rec2_graph'].ndata['x'] = (
                protein2_rot_T @ (protein2_coord_to_move - protein2_mean_to_remove).T).T + protein2_rot_b
            data['p2lig_p2_pocket_coords'][0] = (
                protein2_rot_T @ (data['p2lig_p2_pocket_coords'][0] - protein2_mean_to_remove).T).T + protein2_rot_b
            data['rec2_coords_input'] = [data['rec2_graph'].ndata['x']]

            # Forward pass
            model_outputs = model(**data, mode='predict')[0]

            prediction = model_outputs['ligs_coords_pred']
            p2_to_p1_rotation = model_outputs['rotation_2']
            p2_to_p1_translation = model_outputs['translation_2']
            p2_rmsd_pred = model_outputs['p2_rmsd_pred'].item()
            lig_coords_gt_dev = data['lig_graph'].ndata['x'].to(prediction.device)

            # Ligand correction (EquiBind)
            if CORRECT_LIGAND:
                prediction = correct_ligand(prediction, lig_graph.ndata['new_x'], lig)
                prediction = torch.from_numpy(prediction).float()

            # Save predicted ligand PDB
            lig_pred_path = os.path.join(out_dir_case, f'lig_pred_{name}_{seed}.pdb')
            pred_lig_pdb = set_new_coords(lig_path, prediction.cpu().numpy(), is_ligand=True)
            pred_lig_pdb = set_new_chain(pred_lig_pdb, 'A', is_ligand=True)
            write_pdb(pred_lig_pdb, lig_pred_path)

            # Compute protein2 transformation
            p2_gt = data['rec2_coords'][0].to(prediction.device)
            p2_pred = rotate_and_translate(
                data['rec2_coords_input'][0].to(prediction.device),
                p2_to_p1_rotation, p2_to_p1_translation)
            p2_rmsd_gt = RMSD(p2_gt, p2_pred)
            p1_gt = p1_graph.ndata['x']
            overall_rmsd = RMSD(
                torch.cat((p1_gt, p2_gt), dim=0),
                torch.cat((p1_gt, p2_pred), dim=0))
            p2_rot, p2_trans, _ = kabsch(p2_gt, p2_pred)

            # Save predicted protein2 PDB
            p2_pred_file = os.path.join(out_dir_case, f'p2_pred_{name}_{seed}.pdb')
            p2_pdb = PandasPdb().read_pdb(p2_path)
            p2_gt_coords = p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].to_numpy()
            p2_pred_coords = rotate_and_translate(
                p2_gt_coords, p2_rot.cpu().numpy(), p2_trans.cpu().numpy())
            p2_pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = p2_pred_coords
            p2_pdb.to_pdb(p2_pred_file, records=['ATOM'], gz=False)

            # Save full complex PDB
            complex_save_path = os.path.join(out_dir_case, f'complex_pred_{name}_{seed}.pdb')
            merge_pdbs(
                pdb_files=[p1_path, p2_pred_file, lig_pred_path],
                path=complex_save_path)

            # Compute per-seed metrics
            lig_rmsd = RMSD(prediction, lig_coords_gt_dev)
            pocket_mask_combined = p1lig_lig_pocket_mask | p2lig_lig_pocket_mask
            sm_rmsd = RMSD(prediction[pocket_mask_combined], lig_coords_gt_dev[pocket_mask_combined])

            # DockQ
            from DockQ.dockq_util import cal_dockq
            gt_complex_path = os.path.join(unboud_folder, name, 'gt_complex.pdb')
            try:
                fnat, irms, Lrms, DockQ = cal_dockq(complex_save_path, gt_complex_path)
            except Exception as e:
                print(f"  [WARN] DockQ failed for {name} seed {seed}: {e}")
                fnat, irms, Lrms, DockQ = 0.0, 0.0, 0.0, 0.0

            results.append(dict(
                seed=seed,
                fnat=fnat,
                irms=irms,
                Lrms=Lrms,
                dockq=DockQ,
                pred_p2_rmsd=p2_rmsd_pred,
                gt_p2_rmsd=p2_rmsd_gt,
                lig_rmsd=lig_rmsd,
                sm_rmsd=sm_rmsd,
                overall_rmsd=overall_rmsd,
            ))
            print(f"  [{name}] seed {seed:02d}/{seed_num}: DockQ={DockQ:.4f}, ligRMSD={lig_rmsd:.2f}")

        except Exception as e:
            print(f"  [ERROR] {name} seed {seed}: {e}")
            traceback.print_exc()
            results.append(dict(
                seed=seed,
                fnat=0.0, irms=0.0, Lrms=0.0, dockq=0.0,
                pred_p2_rmsd=0.0, gt_p2_rmsd=0.0,
                lig_rmsd=0.0, sm_rmsd=0.0, overall_rmsd=0.0,
                error=str(e),
            ))

    # Sort by pred_p2_rmsd (best first) and save summary CSV
    results_sorted = sorted(results, key=lambda x: x['pred_p2_rmsd'])
    csv_path = os.path.join(out_dir_case, f'summary_{name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'seed', 'fnat', 'irms', 'Lrms', 'dockq',
            'pred_p2_rmsd', 'gt_p2_rmsd', 'lig_rmsd', 'sm_rmsd', 'overall_rmsd',
            'error'])
        writer.writeheader()
        writer.writerows(results_sorted)

    return {'name': name, 'results': results_sorted, 'success': True}


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='DeepTernary Audit — Run 40-seed predictions for PROTAC benchmark')
    p.add_argument('--work-dir', default='DeepTernary/output/checkpoints/PROTAC',
                   help='Model work directory containing config and checkpoints')
    p.add_argument('--config', default=None,
                   help='Test config file path (auto-detected from work-dir if not set)')
    p.add_argument('--checkpoint', default=None,
                   help='Checkpoint file path (auto-detected from work-dir if not set)')
    p.add_argument('--out-dir', default='audit_results',
                   help='Output directory for prediction results')
    p.add_argument('--case', default=None,
                   help='Run a single case (default: all cases from protac22.txt)')
    p.add_argument('--cases-file', default=None,
                   help='Path to file listing case names (default: DeepTernary/data/PROTAC/protac22.txt)')
    p.add_argument('--seeds', type=int, default=40,
                   help='Number of seeds per case')
    p.add_argument('--device', default='cuda',
                   help='Device to run on (cuda/cpu)')
    p.add_argument('--save-every', type=int, default=1,
                   help='Save progress JSON every N cases')
    p.add_argument('--resume-from', default=None,
                   help='Path to progress JSON to resume from')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve work dir paths
    work_dir = args.work_dir
    if args.config is None:
        files = [f for f in os.listdir(work_dir) if f.endswith('.py')]
        if len(files) != 1:
            sys.exit(f"Expected 1 config file in {work_dir}, found {len(files)}: {files}")
        args.config = os.path.join(work_dir, files[0])
    if args.checkpoint is None:
        last_ckpt_path = os.path.join(work_dir, 'last_checkpoint')
        if not os.path.exists(last_ckpt_path):
            sys.exit(f"last_checkpoint not found in {work_dir}")
        with open(last_ckpt_path, 'r') as f:
            args.checkpoint = f.read().strip()
        if not os.path.exists(args.checkpoint):
            # Try relative to work_dir
            alt = os.path.join(work_dir, os.path.basename(args.checkpoint))
            if os.path.exists(alt):
                args.checkpoint = alt
            else:
                sys.exit(f"Checkpoint not found: {args.checkpoint}")

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output:     {args.out_dir}")
    print(f"Device:     {args.device}")
    print(f"Seeds:      {args.seeds}")

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.model.noise_initial = 0  # disable noise for inference
    cfg.device = args.device
    cfg.model.device = args.device

    # Load model
    print("\nLoading model...")
    from mmpretrain.apis.model import get_model as _get_model
    model = _get_model(cfg, pretrained=args.checkpoint, device=args.device)
    model.to(args.device)
    model.device = args.device
    model.eval()
    cfg.nn_model = model
    print("Model loaded.\n")

    # Get case list
    if args.case:
        complex_names = [args.case]
    else:
        cases_file = args.cases_file or 'DeepTernary/data/PROTAC/protac22.txt'
        with open(cases_file, 'r') as f:
            complex_names = f.read().splitlines()
        complex_names = [n.strip() for n in complex_names if n.strip()]
    print(f"Cases to run: {len(complex_names)}")
    for n in complex_names:
        print(f"  - {n}")

    # Handle resume
    completed = set()
    all_results = {}
    if args.resume_from and os.path.exists(args.resume_from):
        with open(args.resume_from, 'r') as f:
            prev = json.load(f)
        completed = set(prev.get('completed', []))
        all_results = prev.get('results', {})
        print(f"Resuming: {len(completed)} already completed")

    # Run predictions
    start_time = time.time()
    for i, name in enumerate(complex_names):
        if name in completed:
            print(f"[{i+1}/{len(complex_names)}] SKIP {name} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(complex_names)}] {name}")
        print(f"{'='*60}")

        case_start = time.time()
        result = predict_one_case(
            name, model, cfg, args.out_dir,
            seed_num=args.seeds, device=args.device)
        elapsed = time.time() - case_start

        all_results[name] = result['results']
        completed.add(name)
        print(f"  Done in {elapsed:.1f}s ({elapsed/args.seeds:.1f}s/seed)")

        # Periodic save
        if (i + 1) % args.save_every == 0 or (i + 1) == len(complex_names):
            progress_path = os.path.join(args.out_dir, 'prediction_progress.json')
            with open(progress_path, 'w') as f:
                json.dump({
                    'completed': list(completed),
                    'results': all_results,
                    'last_updated': datetime.now().isoformat(),
                }, f, indent=2)
            print(f"  [Progress saved to {progress_path}]")

    total_time = time.time() - start_time
    n_ran = len(completed)
    print(f"\n{'='*60}")
    print(f"ALL DONE. {n_ran} cases in {total_time:.0f}s ({total_time/n_ran:.0f}s/case)")
    print(f"Results saved to {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
