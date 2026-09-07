"""Measure inference-ETKDG versus training-native ligand endpoint mismatch."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_entry import TernaryDataLoader, _load_score_model_args, seed_training
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.dual_consensus import DualConsensusFlow as SDPIFlow


def aligned_rmsd(moving, target):
    """Proper Kabsch RMSD for equal, canonically indexed atom arrays."""
    moving = np.asarray(moving, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cm, ct = moving.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((moving - cm).T @ (target - ct))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    fitted = (moving - cm) @ (u @ correction @ vt) + ct
    return float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=-1))))


def distance_rmse(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_dist = np.linalg.norm(left[:, None] - left[None], axis=-1)
    right_dist = np.linalg.norm(right[:, None] - right[None], axis=-1)
    upper = np.triu_indices(len(left), k=1)
    return float(np.sqrt(np.mean((left_dist[upper] - right_dist[upper]) ** 2)))


def _wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def _dihedral(coords, indices):
    a, b, c, d = (np.asarray(coords[index], dtype=np.float64) for index in indices)
    axis = c - b
    axis /= np.linalg.norm(axis)
    first = a - b
    last = d - c
    first -= np.dot(first, axis) * axis
    last -= np.dot(last, axis) * axis
    first /= np.linalg.norm(first)
    last /= np.linalg.norm(last)
    return float(np.arctan2(np.dot(np.cross(first, last), axis),
                            np.dot(first, last)))


def torsion_quartets(mol, bonds):
    """Pick deterministic neighboring atoms defining every rotatable dihedral."""
    quartets = []
    for u, v in np.asarray(bonds):
        left = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(int(u)).GetNeighbors()
                      if atom.GetIdx() != v)
        right = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(int(v)).GetNeighbors()
                       if atom.GetIdx() != u)
        if not left or not right:
            raise ValueError(f'rotatable bond {(int(u), int(v))} has no dihedral neighbors')
        quartets.append((left[0], int(u), int(v), right[0]))
    return quartets


def align_rotatable_dihedrals(moving, target, bonds, masks, mol, passes=2):
    """Match all accessible absolute torsions before measuring scaffold residual."""
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    moving = np.asarray(moving, dtype=np.float64).copy()
    bonds_tensor = torch.as_tensor(bonds, dtype=torch.long)
    quartets = torsion_quartets(mol, bonds)
    epsilon = 1e-4
    for _ in range(passes):
        for index, quartet in enumerate(quartets):
            current = _dihedral(moving, quartet)
            desired = _dihedral(target, quartet)
            one_mask = np.asarray(masks[index:index + 1])
            one_bond = bonds_tensor[index:index + 1]
            probe = modify_conformer_torsion_angles(
                moving, one_bond, one_mask, np.asarray([epsilon]), as_numpy=True)
            response = _wrap_angle(_dihedral(probe, quartet) - current) / epsilon
            if not np.isfinite(response) or abs(response) < .5:
                raise FloatingPointError('could not determine torsion update convention')
            update = _wrap_angle(desired - current) / response
            moving = modify_conformer_torsion_angles(
                moving, one_bond, one_mask, np.asarray([update]), as_numpy=True)
    residuals = [_wrap_angle(_dihedral(moving, quartet) - _dihedral(target, quartet))
                 for quartet in quartets]
    return moving, float(max((abs(value) for value in residuals), default=0.0))


def local_geometry_errors(left, right, mol):
    """Bond lengths and angles are invariant under all supported torsion updates."""
    length_errors = []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length_errors.append(np.linalg.norm(left[u] - left[v]) -
                             np.linalg.norm(right[u] - right[v]))
    angle_errors = []
    for center in range(mol.GetNumAtoms()):
        neighbors = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors())
        for i, u in enumerate(neighbors):
            for v in neighbors[i + 1:]:
                def angle(coords):
                    a, b = coords[u] - coords[center], coords[v] - coords[center]
                    return np.arccos(np.clip(np.dot(a, b) /
                                     (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
                angle_errors.append(_wrap_angle(angle(left) - angle(right)))
    return {
        'bond_length_rmse_A': float(np.sqrt(np.mean(np.square(length_errors)))),
        'bond_angle_rmse_deg': float(np.degrees(np.sqrt(np.mean(np.square(angle_errors))))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=43000)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    args = parser.parse_args()
    if args.limit < 1 or args.seeds < 1:
        parser.error('limit and seeds must be positive')
    payload = json.loads(Path(args.index).read_text())
    if payload.get('metadata', {}).get('split') != 'val':
        raise ValueError('endpoint audit requires an explicitly labeled validation index')
    score_args = _load_score_model_args(args.expert_model_dir)
    loader = TernaryDataLoader(
        args.index, score_args, shuffle=False, device='cuda',
        cache_dir=args.cache_dir)
    batches = [loader._load_one(entry) for entry in loader.entries[:args.limit]]
    flow = SDPIFlow()
    initializer = TernaryInitializer(no_torsion=bool(score_args.no_torsion))
    records = []
    for case_index, batch in enumerate(batches):
        for seed_offset in range(args.seeds):
            seed = args.seed + case_index * 1000 + seed_offset
            seed_training(seed, torch.device('cuda'))
            inference_state, metadata = flow.initialize(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'], seed=seed, no_torsion=initializer.no_torsion)
            coupling = flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'], no_torsion=initializer.no_torsion)
            # Both samples use the same ETKDG manifold. Remove their independent
            # torsion/pose differences below before measuring internal support.
            training_state, _ = flow.path(
                batch['ligand_data'], batch['poi_data']['coords'], coupling,
                torch.tensor([1.], device='cuda'))
            inference = inference_state['a'][0].detach().cpu().numpy()
            training = training_state['a'][0].detach().cpu().numpy()
            from rdkit import Chem
            mol = batch['ligand_data']['mol']
            if isinstance(mol, bytes):
                mol = Chem.Mol(mol)
            aligned_torsions, torsion_residual = align_rotatable_dihedrals(
                inference, training, coupling['bonds'], coupling['masks'],
                mol)
            records.append({
                'case': batch['case'], 'seed': seed,
                'n_atoms': len(inference),
                'n_torsions': len(metadata['torsion_updates']),
                'raw_aligned_rmsd_A': aligned_rmsd(inference, training),
                'raw_pair_distance_rmse_A': distance_rmse(inference, training),
                'torsion_aligned_rmsd_A': aligned_rmsd(aligned_torsions, training),
                'torsion_aligned_pair_distance_rmse_A': distance_rmse(
                    aligned_torsions, training),
                'max_torsion_residual_deg': float(np.degrees(torsion_residual)),
                **local_geometry_errors(inference, training, mol),
            })
    summary = {}
    for key in ('raw_aligned_rmsd_A', 'raw_pair_distance_rmse_A',
                'torsion_aligned_rmsd_A', 'torsion_aligned_pair_distance_rmse_A',
                'max_torsion_residual_deg', 'bond_length_rmse_A',
                'bond_angle_rmse_deg'):
        values = np.asarray([row[key] for row in records])
        summary[key] = {'mean': float(values.mean()),
                        'median': float(np.median(values)),
                        'max': float(values.max())}
    output = {'scope': 'validation endpoint support; same torsions and atom order',
              'args': vars(args), 'cases': [b['case'] for b in batches],
              'summary': summary, 'records': records}
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
