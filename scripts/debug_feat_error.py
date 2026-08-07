#!/usr/bin/env python
"""Debug script to trace the KeyError: 'feat' issue for 6W7O/6W8I cases."""

import os, sys, traceback
from copy import deepcopy
from functools import partial

import numpy as np
import scipy.spatial as spa
import torch
import dgl
from biopandas.pdb import PandasPdb
from rdkit import Chem
from scipy.special import softmax
from mmengine.config import Config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEEPTERNARY_ROOT = os.path.join(_PROJECT_ROOT, 'DeepTernary')
sys.path.insert(0, _DEEPTERNARY_ROOT)

from deepternary.models.process_mols import (
    read_molecule, get_receptor_inference, get_rec_graph,
    lig_atom_featurizer, distance_featurizer,
    get_geometry_graph_ring, rigid_transform_Kabsch_3D,
    get_rdkit_coords_v2,
)
from deepternary.models.triplet_dock import _get_model
from deepternary.models.pdb_utils import get_lig_coords

# Replicate the audit script's get_lig_graph_protac
_embedding_failed_names = set()

def auto_download_ideal_sdf(lig_name):
    import requests
    IDEAL_PATH = 'DeepTernary/output/ideal/'
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

    print(f"  [DEBUG] lig_graph: nodes={num_nodes}, edges={len(src_list)}")
    print(f"  [DEBUG] src_list sample: {src_list[:5]}, dst_list sample: {dst_list[:5]}")

    graph.ndata["feat"] = lig_atom_featurizer(mol)
    graph.edata["feat"] = distance_featurizer(
        dist_list, 0.75)
    graph.ndata["x"] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata["mu_r_norm"] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata["new_x"] = torch.from_numpy(np.array(lig_coords).astype(np.float32))

    # Verify edata is set
    print(f"  [DEBUG] After building: ndata keys={list(graph.ndata.keys())}")
    print(f"  [DEBUG] After building: edata keys={list(graph.edata.keys())}")
    print(f"  [DEBUG] edata['feat'] shape: {graph.edata['feat'].shape}")

    return mol, graph


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "6W7O_C_A_TL7"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing case: {name}, device: {device}")

    # Load model
    work_dir = 'DeepTernary/output/checkpoints/PROTAC/20240301_081911'
    config_path = os.path.join(work_dir, '20240301_081911.py')
    last_ckpt_path = os.path.join(work_dir, 'last_checkpoint')

    with open(last_ckpt_path, 'r') as f:
        checkpoint = f.read().strip()
    if not os.path.exists(checkpoint):
        checkpoint = os.path.join(work_dir, os.path.basename(checkpoint))

    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint}")

    cfg = Config.fromfile(config_path)
    cfg.model.noise_initial = 0

    model = _get_model(cfg)
    state_dict = torch.load(checkpoint, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Build graphs
    unboud_folder = 'DeepTernary/output/protac22'
    lig_path = os.path.join(unboud_folder, name, 'ligand.pdb')
    p1_path = os.path.join(unboud_folder, name, 'unbound_protein1.pdb')
    p2_path = os.path.join(unboud_folder, name, 'unbound_protein2.pdb')
    lig1_path = os.path.join(unboud_folder, name, 'unbound_lig1.pdb')
    lig2_path = os.path.join(unboud_folder, name, 'unbound_lig2.pdb')

    ds_cfg = cfg.test_dataloader.dataset

    print("\n--- Building graphs ---")
    lig_origin = read_molecule(lig_path, sanitize=True, remove_hs=ds_cfg.remove_h)
    print(f"Ligand loaded: {lig_origin.GetNumAtoms()} atoms")

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p1_path)
    p1_graph = get_rec_graph(
        recs, recs_coords, c_alpha_coords, n_coords, c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)
    print(f"p1_graph: nodes={p1_graph.num_nodes()}, edges={p1_graph.num_edges()}")
    print(f"p1 ndata keys: {list(p1_graph.ndata.keys())}, edata keys: {list(p1_graph.edata.keys())}")

    recs, recs_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(p2_path)
    p2_graph = get_rec_graph(
        recs, recs_coords, c_alpha_coords, n_coords, c_coords,
        use_rec_atoms=ds_cfg.use_rec_atoms, rec_radius=ds_cfg.rec_graph_radius,
        surface_max_neighbors=ds_cfg.surface_max_neighbors,
        surface_graph_cutoff=ds_cfg.surface_graph_cutoff,
        surface_mesh_cutoff=ds_cfg.surface_mesh_cutoff,
        c_alpha_max_neighbors=ds_cfg.c_alpha_max_neighbors)
    print(f"p2_graph: nodes={p2_graph.num_nodes()}, edges={p2_graph.num_edges()}")
    print(f"p2 ndata keys: {list(p2_graph.ndata.keys())}, edata keys: {list(p2_graph.edata.keys())}")

    lig1 = read_molecule(lig1_path, sanitize=True, remove_hs=True)
    lig2 = read_molecule(lig2_path, sanitize=True, remove_hs=True)
    print(f"lig1: {lig1.GetNumAtoms()} atoms, lig2: {lig2.GetNumAtoms()} atoms")

    lig = deepcopy(lig_origin)
    lig, lig_graph = get_lig_graph_protac(
        lig, name=name, lig1=lig1, lig2=lig2,
        max_neighbors=ds_cfg.lig_max_neighbors,
        use_random_coords=False, seed=0,
        use_rdkit_coords=True, radius=ds_cfg.lig_graph_radius)

    geometry_graph = get_geometry_graph_ring(lig)
    print(f"geometry_graph: nodes={geometry_graph.num_nodes()}, edges={geometry_graph.num_edges()}")
    print(f"geometry ndata keys: {list(geometry_graph.ndata.keys())}, edata keys: {list(geometry_graph.edata.keys())}")

    # Check lig_graph edata one more time
    print(f"\nlig_graph BEFORE model call: edata keys={list(lig_graph.edata.keys())}")

    # Try moving to device
    lig_graph = lig_graph.to(device)
    p1_graph = p1_graph.to(device)
    p2_graph = p2_graph.to(device)
    geometry_graph = geometry_graph.to(device)

    print(f"lig_graph AFTER .to(device): edata keys={list(lig_graph.edata.keys())}")

    # Build data dict
    from deepternary.models.ternary_pdb import get_pocket_and_mask
    from deepternary.models.rotate_utils import random_rotation_translation

    lig_coords_gt = deepcopy(lig_graph.ndata['x'])
    lig1_coords = get_lig_coords(lig1_path)
    lig2_coords = get_lig_coords(lig2_path)

    # Simplified: no replacement
    pocket_mask = torch.zeros(lig_coords_gt.shape[0], dtype=torch.bool)

    p1lig_lig_pocket_coords, p1lig_lig_pocket_mask, p1lig_p1_pocket_mask = \
        get_pocket_and_mask(lig_coords_gt, p1_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)
    p2lig_lig_pocket_coords, p2lig_lig_pocket_mask, p2lig_p2_pocket_mask = \
        get_pocket_and_mask(lig_coords_gt, p2_graph.ndata['x'], cutoff=ds_cfg.pocket_cutoff)

    data = dict(
        lig_graph=lig_graph,
        rec_graph=p1_graph,
        rec2_graph=p2_graph,
        geometry_graph=geometry_graph,
        complex_name=[name],
        rec2_coords=[p2_graph.ndata['x']],
        p1lig_p1_pocket_mask=[p1lig_p1_pocket_mask],
        p1lig_p1_pocket_coords=[p1lig_lig_pocket_coords],
        p1lig_lig_pocket_mask=[p1lig_lig_pocket_mask],
        p1lig_lig_pocket_coords=[p1lig_lig_pocket_coords],
        p2lig_p2_pocket_mask=[p2lig_p2_pocket_mask],
        p2lig_p2_pocket_coords=[p2lig_lig_pocket_coords],
        p2lig_lig_pocket_mask=[p2lig_lig_pocket_mask],
        p2lig_lig_pocket_coords=[p2lig_lig_pocket_coords],
    )

    # Perturb ligand
    rot_T, rot_b = random_rotation_translation(translation_distance=5)
    lig_coords_to_move = data['lig_graph'].ndata['new_x']
    mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
    data['lig_graph'].ndata['new_x'] = (
        rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b

    print(f"\n--- Running model forward ---")
    print(f"lig_graph before model: ndata keys={list(data['lig_graph'].ndata.keys())}, edata keys={list(data['lig_graph'].edata.keys())}")

    try:
        with torch.no_grad():
            model_outputs = model(**data, mode='predict')[0]
        print(f"SUCCESS: ligs_coords_pred shape={model_outputs['ligs_coords_pred'].shape}")
    except KeyError as e:
        print(f"KeyError: {e}")
        tb = traceback.format_exc()
        print(tb[:2000])
    except Exception as e:
        print(f"Other error: {e}")
        tb = traceback.format_exc()
        print(tb[:2000])

if __name__ == '__main__':
    main()
