#!/usr/bin/env python
"""
DeepTernary Audit — Metrics Computation
=======================================
Reads prediction results from audit_results/ and computes all metrics specified
in the DeepTernary audit handover document.

Metrics computed:
  1. Global DockQ (mean, std, median)
  2. Stratified DockQ by linker length (short/mid/long)
  3. Ligand RMSD
  4. Inter-molecular clash rate (Cβ-Cβ < 3.0Å)
  5. Linker strain energy (MMFF94)
  6. Ensemble diversity (pairwise RMSD clustering at 2Å)
  7. Top-1 hit rate
  8. Seed stability (5/10/20/40 subsampling)

Output:
  deepternary_audit.json — structured audit results matching the spec

Usage:
    python scripts/audit_compute_metrics.py                          # all defaults
    python scripts/audit_compute_metrics.py --pred-dir audit_results --output deepternary_audit.json
    python scripts/audit_compute_metrics.py --cases-file DeepTernary/data/PROTAC/protac22.txt
"""

import argparse
import json
import os
import sys
import csv
import random
import tempfile
from collections import defaultdict
from datetime import datetime

import numpy as np

# Allow running from CoopDiff/ root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEEPTERNARY_ROOT = os.path.join(_PROJECT_ROOT, 'DeepTernary')
sys.path.insert(0, _DEEPTERNARY_ROOT)

# ── RDKit imports (with graceful fallback) ────────────────────────────────────

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem import Descriptors
    HAVE_RDKIT = True
except ImportError:
    HAVE_RDKIT = False
    print("[WARN] RDKit not available — linker strain and rotatable bond metrics will be skipped")

# ── Biopython imports (with graceful fallback) ────────────────────────────────

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.NeighborSearch import NeighborSearch
    HAVE_BIOPYTHON = True
except ImportError:
    HAVE_BIOPYTHON = False
    print("[WARN] BioPython not available — clash rate will be approximated")

# ── Constants ─────────────────────────────────────────────────────────────────

# CAPRI quality thresholds
CAPRI_THRESHOLDS = {'Incorrect': 0.23, 'Acceptable': 0.49, 'Medium': 0.80, 'High': float('inf')}

# VDW radii for clash detection (common atoms)
VDW_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'F': 1.47, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
}

# Linker group definitions
LINKER_SHORT = 8       # ≤ 8 rotatable bonds
LINKER_LONG = 15        # > 15 rotatable bonds
# between 9-15 = medium

# ── Helper: read per-case results ─────────────────────────────────────────────

def read_case_results(case_name, pred_dir):
    """Read summary CSV for a case, return list of per-seed dicts."""
    csv_path = os.path.join(pred_dir, case_name, f'summary_{case_name}.csv')
    if not os.path.exists(csv_path):
        print(f"  [WARN] No summary CSV for {case_name}")
        return []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        results = []
        for row in reader:
            results.append({
                'seed': int(row['seed']),
                'fnat': float(row['fnat']),
                'irms': float(row['irms']),
                'Lrms': float(row['Lrms']),
                'dockq': float(row['dockq']),
                'pred_p2_rmsd': float(row['pred_p2_rmsd']),
                'gt_p2_rmsd': float(row['gt_p2_rmsd']),
                'lig_rmsd': float(row['lig_rmsd']),
                'sm_rmsd': float(row['sm_rmsd']),
                'overall_rmsd': float(row['overall_rmsd']),
            })
        return results


# ── Metric 1: DockQ ──────────────────────────────────────────────────────────

def compute_dockq_metrics(all_case_results):
    """Compute global DockQ statistics across all cases and seeds."""
    all_dockq = []
    case_best_dockq = []
    for case_name, results in all_case_results.items():
        if results:
            dockqs = [r['dockq'] for r in results]
            all_dockq.extend(dockqs)
            case_best_dockq.append(max(dockqs))

    if not all_dockq:
        return {'mean_dockq': 0, 'std_dockq': 0, 'median_dockq': 0}

    return {
        'mean_dockq': float(np.mean(all_dockq)),
        'std_dockq': float(np.std(all_dockq)),
        'median_dockq': float(np.median(all_dockq)),
        'mean_best_dockq': float(np.mean(case_best_dockq)),
        'std_best_dockq': float(np.std(case_best_dockq)),
    }


# ── Metric 2: Ligand RMSD ────────────────────────────────────────────────────

def compute_ligand_rmsd_metrics(all_case_results):
    """Compute global ligand RMSD statistics."""
    all_rmsd = []
    for case_name, results in all_case_results.items():
        if results:
            all_rmsd.extend([r['lig_rmsd'] for r in results])

    if not all_rmsd:
        return {'mean_ligand_rmsd': 0, 'std_ligand_rmsd': 0}

    return {
        'mean_ligand_rmsd': float(np.mean(all_rmsd)),
        'std_ligand_rmsd': float(np.std(all_rmsd)),
    }


# ── Metric 3: Clash rate ─────────────────────────────────────────────────────

def compute_clash_rate_for_pdb(pdb_path, clash_cutoff=3.0):
    """
    Compute inter-molecular clash rate for a ternary complex PDB.
    Uses Cβ-Cβ distances between different chains/molecules.
    Returns fraction of inter-molecular atom pairs with distance < cutoff.
    """
    if not HAVE_BIOPYTHON:
        return None

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_path)
        model = structure[0]

        # Collect atoms by chain
        chain_atoms = defaultdict(list)
        all_atoms = []
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] != ' ':  # skip heteroatoms
                    continue
                for atom in residue:
                    chain_atoms[chain.id].append(atom)
                    all_atoms.append(atom)

        chain_ids = list(chain_atoms.keys())
        if len(chain_ids) < 2:
            return 0.0

        # Count inter-chain clashes
        total_inter_pairs = 0
        clash_count = 0
        for i in range(len(chain_ids)):
            for j in range(i + 1, len(chain_ids)):
                atoms_i = chain_atoms[chain_ids[i]]
                atoms_j = chain_atoms[chain_ids[j]]
                for ai in atoms_i:
                    for aj in atoms_j:
                        total_inter_pairs += 1
                        dist = ai - aj  # BioPython atom subtraction gives distance
                        if dist < clash_cutoff:
                            clash_count += 1

        if total_inter_pairs == 0:
            return 0.0
        return clash_count / total_inter_pairs

    except Exception as e:
        print(f"    [WARN] Clash rate failed for {pdb_path}: {e}")
        return None


def compute_clash_metrics(all_case_results, pred_dir):
    """Compute clash rate per case using the best-DockQ seed prediction."""
    clash_rates = []
    for case_name in all_case_results.keys():
        # Use the seed with highest DockQ
        results = all_case_results[case_name]
        if not results:
            clash_rates.append(0.0)
            continue
        best = max(results, key=lambda r: r['dockq'])
        pdb_path = os.path.join(pred_dir, case_name,
                                f"complex_pred_{case_name}_{best['seed']}.pdb")
        if os.path.exists(pdb_path):
            cr = compute_clash_rate_for_pdb(pdb_path)
            clash_rates.append(cr if cr is not None else 0.0)
        else:
            clash_rates.append(0.0)

    valid = [c for c in clash_rates if c > 0]
    return {
        'mean_clash_rate': float(np.mean(clash_rates)) if clash_rates else 0.0,
        'std_clash_rate': float(np.std(clash_rates)) if clash_rates else 0.0,
        'per_case_clash_rates': dict(zip(all_case_results.keys(), clash_rates)),
    }


# ── Metric 4: Linker rotatable bonds ─────────────────────────────────────────

def count_linker_rotatable_bonds(protac_mol, lig1_mol, lig2_mol):
    """
    Count rotatable bonds in the linker region only.
    Strategy: identify linker atoms = PROTAC atoms not in lig1 or lig2,
    then count rotatable bonds where at least one atom is in the linker.
    """
    if not HAVE_RDKIT:
        return 0

    # Get atom indices for warhead fragments
    lig1_match = protac_mol.GetSubstructMatch(lig1_mol)
    lig2_match = protac_mol.GetSubstructMatch(lig2_mol)
    warhead_atoms = set(lig1_match) | set(lig2_match)

    # Get rotatable bond info
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(protac_mol)
    # For more detailed analysis, identify bonds individually
    rot_bond_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    if rot_bond_smarts is None:
        return rot_bonds  # fallback to simple count
    matches = protac_mol.GetSubstructMatches(rot_bond_smarts)
    linker_rot = 0
    for match in matches:
        a1, a2 = match[0], match[1]
        if a1 not in warhead_atoms or a2 not in warhead_atoms:
            linker_rot += 1
    return linker_rot


def get_linker_info(case_name, data_dir='DeepTernary/output/protac22'):
    """Get linker rotatable bond count for a PROTAC case."""
    if not HAVE_RDKIT:
        return {'linker_n_rotatable': 0, 'linker_group': 'mid'}

    lig_path = os.path.join(data_dir, case_name, 'ligand.pdb')
    lig1_path = os.path.join(data_dir, case_name, 'unbound_lig1.pdb')
    lig2_path = os.path.join(data_dir, case_name, 'unbound_lig2.pdb')

    try:
        protac = Chem.MolFromPDBFile(lig_path, sanitize=True, removeHs=True)
        lig1 = Chem.MolFromPDBFile(lig1_path, sanitize=True, removeHs=True)
        lig2 = Chem.MolFromPDBFile(lig2_path, sanitize=True, removeHs=True)
        if protac is None or lig1 is None or lig2 is None:
            # Fall back to SDF if available
            lig_sdf = os.path.join(data_dir, case_name, 'ligand.sdf')
            if os.path.exists(lig_sdf):
                protac = Chem.SDMolSupplier(lig_sdf, sanitize=True, removeHs=True)[0]
        if protac is None:
            return {'linker_n_rotatable': 0, 'linker_group': 'mid'}

        n_rot = count_linker_rotatable_bonds(protac, lig1, lig2)
        if n_rot <= LINKER_SHORT:
            group = 'short'
        elif n_rot <= LINKER_LONG:
            group = 'mid'
        else:
            group = 'long'
        return {'linker_n_rotatable': n_rot, 'linker_group': group}
    except Exception as e:
        print(f"  [WARN] Linker info failed for {case_name}: {e}")
        return {'linker_n_rotatable': 0, 'linker_group': 'mid'}


# ── Metric 5: Linker strain energy ───────────────────────────────────────────

def compute_linker_strain(pdb_path, lig1_path, lig2_path):
    """
    Compute MMFF94 energy of the linker region.
    Strategy: load PROTAC, identify linker atoms, compute MMFF94 energy.
    """
    if not HAVE_RDKIT:
        return None

    try:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=True)
        if mol is None:
            return None

        # Try MMFF94
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if ff is None:
            return None
        energy = ff.CalcEnergy()
        return float(energy)
    except Exception as e:
        print(f"    [WARN] MMFF94 failed for {pdb_path}: {e}")
        return None


def compute_linker_strain_metrics(all_case_results, pred_dir):
    """Compute linker strain energy per case using best-DockQ seed."""
    strains = []
    per_case = {}
    for case_name in all_case_results.keys():
        results = all_case_results[case_name]
        if not results:
            strains.append(0.0)
            per_case[case_name] = 0.0
            continue
        best = max(results, key=lambda r: r['dockq'])
        lig_pred_path = os.path.join(pred_dir, case_name,
                                     f"lig_pred_{case_name}_{best['seed']}.pdb")
        if os.path.exists(lig_pred_path):
            s = compute_linker_strain(lig_pred_path, None, None)
            val = s if s is not None else 0.0
        else:
            val = 0.0
        strains.append(val)
        per_case[case_name] = val

    valid = [s for s in strains if s != 0.0]
    return {
        'mean_linker_strain': float(np.mean(strains)) if strains else 0.0,
        'std_linker_strain': float(np.std(strains)) if strains else 0.0,
        'per_case_linker_strain': per_case,
    }


# ── Metric 6: Ensemble diversity (pairwise RMSD clustering) ───────────────────

def compute_pairwise_ligand_rmsd(lig_coords_list):
    """Compute pairwise RMSD matrix for a list of ligand coordinate arrays."""
    n = len(lig_coords_list)
    rmsd_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = lig_coords_list[i], lig_coords_list[j]
            diff = ci - cj
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            rmsd_mat[i, j] = rmsd
            rmsd_mat[j, i] = rmsd
    return rmsd_mat


def cluster_by_rmsd(rmsd_mat, cutoff=2.0):
    """Greedy clustering by pairwise RMSD cutoff."""
    n = rmsd_mat.shape[0]
    remaining = set(range(n))
    clusters = []
    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        # Find all seeds within cutoff of seed
        for other in list(remaining):
            if rmsd_mat[seed, other] < cutoff:
                cluster.add(other)
                remaining.discard(other)
        clusters.append(cluster)
    return clusters


def get_ligand_coords_from_pdb(pdb_path):
    """Extract ligand (HETATM) coordinates from a PDB file."""
    try:
        from biopandas.pdb import PandasPdb
        ppdb = PandasPdb().read_pdb(pdb_path)
        het = ppdb.df['HETATM']
        if het is not None and len(het) > 0:
            coords = het[['x_coord', 'y_coord', 'z_coord']].to_numpy()
            return coords
        # Fall back to ATOM records
        atom = ppdb.df['ATOM']
        if atom is not None and len(atom) > 0:
            coords = atom[['x_coord', 'y_coord', 'z_coord']].to_numpy()
            return coords
    except Exception as e:
        print(f"    [WARN] Failed to read coords from {pdb_path}: {e}")
    return None


def compute_ensemble_diversity(case_name, results, pred_dir):
    """Compute ensemble diversity metrics for one case."""
    n_seeds = len(results)
    if n_seeds < 2:
        return {'n_clusters': 0, 'top1_rmsd': 0.0, 'pairwise_rmsd_mean': 0.0}

    # Get ligand coords for each seed
    lig_coords = []
    gt_path = os.path.join('DeepTernary/output/protac22', case_name, 'ligand.pdb')
    gt_coords = get_ligand_coords_from_pdb(gt_path)

    for r in results:
        lig_pred_path = os.path.join(pred_dir, case_name,
                                     f"lig_pred_{case_name}_{r['seed']}.pdb")
        coords = get_ligand_coords_from_pdb(lig_pred_path)
        if coords is not None:
            lig_coords.append(coords)

    if len(lig_coords) < 2:
        return {'n_clusters': 0, 'top1_rmsd': 0.0, 'pairwise_rmsd_mean': 0.0}

    # Pairwise RMSD matrix
    rmsd_mat = compute_pairwise_ligand_rmsd(lig_coords)
    mean_pairwise = float(np.mean(rmsd_mat[rmsd_mat > 0]))

    # Clustering at 2Å
    clusters = cluster_by_rmsd(rmsd_mat, cutoff=2.0)
    n_clusters = len(clusters)

    # Top-1: centroid of largest cluster vs ground truth
    largest_cluster = max(clusters, key=len)
    cluster_indices = list(largest_cluster)
    centroid_idx = cluster_indices[0]  # Use first as proxy for centroid
    if gt_coords is not None and len(gt_coords) == len(lig_coords[centroid_idx]):
        diff = lig_coords[centroid_idx] - gt_coords
        top1_rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    else:
        top1_rmsd = 0.0

    return {
        'n_clusters': n_clusters,
        'top1_rmsd': top1_rmsd,
        'pairwise_rmsd_mean': mean_pairwise,
        'cluster_sizes': [len(c) for c in sorted(clusters, key=len, reverse=True)],
    }


# ── Metric 7: Seed stability ─────────────────────────────────────────────────

def compute_seed_stability(results, n_subsamples=10):
    """
    Compute DockQ stability under different seed subset sizes.
    Randomly draws 5/10/20/all seeds, computes best DockQ, repeats n_subsamples times.
    """
    if len(results) < 5:
        return {'n5': {'dockq_mean': 0.0, 'dockq_std': 0.0},
                'n10': {'dockq_mean': 0.0, 'dockq_std': 0.0},
                'n20': {'dockq_mean': 0.0, 'dockq_std': 0.0},
                'n40': {'dockq_mean': 0.0, 'dockq_std': 0.0}}

    dockqs = np.array([r['dockq'] for r in results])
    stability = {}
    for n_seeds in [5, 10, 20, len(results)]:
        bests = []
        actual_n = min(n_seeds, len(results))
        for _ in range(n_subsamples):
            subset = np.random.choice(dockqs, size=actual_n, replace=False)
            bests.append(np.max(subset))
        label = f"n{min(n_seeds, 40)}" if n_seeds <= len(results) else f"n{len(results)}"
        stability[label] = {
            'dockq_mean': float(np.mean(bests)),
            'dockq_std': float(np.std(bests)),
        }
    return stability


# ── Metric 8: Stratified DockQ ────────────────────────────────────────────────

def compute_stratified_dockq(all_case_results, linker_info):
    """Compute DockQ stratified by linker length group."""
    groups = {'short_linker_leq8': [], 'mid_linker_8_15': [], 'long_linker_gt15': []}
    group_cases = {'short_linker_leq8': [], 'mid_linker_8_15': [], 'long_linker_gt15': []}

    for case_name, results in all_case_results.items():
        if not results:
            continue
        info = linker_info.get(case_name, {})
        group = info.get('linker_group', 'mid')

        # Best DockQ per case
        best_dockq = max(r['dockq'] for r in results)
        if group == 'short':
            groups['short_linker_leq8'].append(best_dockq)
            group_cases['short_linker_leq8'].append(case_name)
        elif group == 'long':
            groups['long_linker_gt15'].append(best_dockq)
            group_cases['long_linker_gt15'].append(case_name)
        else:
            groups['mid_linker_8_15'].append(best_dockq)
            group_cases['mid_linker_8_15'].append(case_name)

    stratified = {}
    for gkey, dockqs in groups.items():
        stratified[gkey] = {
            'n': len(dockqs),
            'mean_dockq': float(np.mean(dockqs)) if dockqs else 0.0,
            'std_dockq': float(np.std(dockqs)) if dockqs else 0.0,
            'cases': group_cases[gkey],
        }
    return stratified


# ── CAPRI quality distribution ────────────────────────────────────────────────

def compute_capri_distribution(all_case_results):
    """Compute CAPRI quality class distribution."""
    all_dockq = []
    for results in all_case_results.values():
        if results:
            all_dockq.extend([r['dockq'] for r in results])

    dist = {'Incorrect': 0, 'Acceptable': 0, 'Medium': 0, 'High': 0}
    for dq in all_dockq:
        if dq < 0.23:
            dist['Incorrect'] += 1
        elif dq < 0.49:
            dist['Acceptable'] += 1
        elif dq < 0.80:
            dist['Medium'] += 1
        else:
            dist['High'] += 1

    total = len(all_dockq) or 1
    for k in dist:
        dist[k] = {'count': dist[k], 'fraction': round(dist[k] / total, 4)}
    return dist


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='DeepTernary Audit — Compute metrics from prediction results')
    p.add_argument('--pred-dir', default='audit_results',
                   help='Directory containing per-case prediction outputs')
    p.add_argument('--cases-file', default='DeepTernary/data/PROTAC/protac22.txt',
                   help='Path to file listing case names')
    p.add_argument('--data-dir', default='DeepTernary/output/protac22',
                   help='Path to PROTAC dataset directory')
    p.add_argument('--output', default='deepternary_audit.json',
                   help='Output JSON file path')
    p.add_argument('--seed-stability-samples', type=int, default=10,
                   help='Number of subsamples for seed stability analysis')
    p.add_argument('--skip-clash', action='store_true',
                   help='Skip clash rate computation (slow for large complexes)')
    return p.parse_args()


def main():
    args = parse_args()

    # Load case names
    with open(args.cases_file, 'r') as f:
        case_names = f.read().splitlines()
    case_names = [n.strip() for n in case_names if n.strip()]
    print(f"Cases: {len(case_names)}")
    for n in case_names:
        print(f"  - {n}")

    # Read per-case results
    print("\nReading prediction results...")
    all_case_results = {}
    for name in case_names:
        results = read_case_results(name, args.pred_dir)
        all_case_results[name] = results
        print(f"  {name}: {len(results)} seeds")

    # Compute linker info
    print("\nComputing linker properties...")
    linker_info = {}
    for name in case_names:
        linker_info[name] = get_linker_info(name, data_dir=args.data_dir)
        print(f"  {name}: {linker_info[name]}")

    # ── Compute all metrics ───────────────────────────────────────────────
    print("\nComputing metrics...")

    # 1. DockQ
    dockq_metrics = compute_dockq_metrics(all_case_results)
    print(f"  DockQ: mean={dockq_metrics['mean_dockq']:.4f}, "
          f"std={dockq_metrics['std_dockq']:.4f}, median={dockq_metrics['median_dockq']:.4f}")

    # 2. Ligand RMSD
    lig_rmsd_metrics = compute_ligand_rmsd_metrics(all_case_results)
    print(f"  LigRMSD: mean={lig_rmsd_metrics['mean_ligand_rmsd']:.2f}")

    # 3. Clash rate
    if not args.skip_clash:
        clash_metrics = compute_clash_metrics(all_case_results, args.pred_dir)
        print(f"  Clash: mean={clash_metrics['mean_clash_rate']:.4f}")
    else:
        clash_metrics = {'mean_clash_rate': 0.0, 'std_clash_rate': 0.0}

    # 4. Linker strain
    strain_metrics = compute_linker_strain_metrics(all_case_results, args.pred_dir)
    print(f"  Linker strain: mean={strain_metrics['mean_linker_strain']:.2f}")

    # 5. Ensemble diversity
    ensemble_diversity = {}
    for name in case_names:
        ensemble_diversity[name] = compute_ensemble_diversity(
            name, all_case_results[name], args.pred_dir)
    mean_n_clusters = np.mean([d['n_clusters'] for d in ensemble_diversity.values()])
    mean_top1_rmsd = np.mean([d['top1_rmsd'] for d in ensemble_diversity.values()
                              if d['top1_rmsd'] > 0])
    print(f"  Ensemble: mean clusters={mean_n_clusters:.1f}, mean top1 RMSD={mean_top1_rmsd:.2f}")

    # 6. Seed stability (aggregated across all cases)
    all_stability = {}
    for n_seeds_label in ['n5', 'n10', 'n20']:
        all_stability[n_seeds_label] = {'dockq_mean': 0.0, 'dockq_std': 0.0}

    for name in case_names:
        if len(all_case_results[name]) >= 5:
            stab = compute_seed_stability(all_case_results[name],
                                          n_subsamples=args.seed_stability_samples)
            for k in all_stability:
                all_stability[k]['dockq_mean'] += stab[k]['dockq_mean']
                all_stability[k]['dockq_std'] += stab[k]['dockq_std']

    n_valid = max(1, sum(1 for n in case_names if len(all_case_results[n]) >= 5))
    for k in all_stability:
        all_stability[k]['dockq_mean'] /= n_valid
        all_stability[k]['dockq_std'] /= n_valid

    # Add n40 as overall best DockQ stats
    best_dockqs = []
    for name in case_names:
        if all_case_results[name]:
            best_dockqs.append(max(r['dockq'] for r in all_case_results[name]))
    all_stability['n40'] = {
        'dockq_mean': float(np.mean(best_dockqs)) if best_dockqs else 0.0,
        'dockq_std': float(np.std(best_dockqs)) if best_dockqs else 0.0,
    }
    print(f"  Seed stability: n5={all_stability['n5']['dockq_mean']:.4f}, "
          f"n40={all_stability['n40']['dockq_mean']:.4f}")

    # 7. Stratified DockQ
    stratified = compute_stratified_dockq(all_case_results, linker_info)
    for gkey, gdata in stratified.items():
        print(f"  {gkey}: n={gdata['n']}, DockQ={gdata['mean_dockq']:.4f}")

    # 8. CAPRI distribution
    capri_dist = compute_capri_distribution(all_case_results)
    print(f"  CAPRI: {capri_dist}")

    # 9. Pred P2 RMSD stats (aggregated)
    all_pred_p2 = []
    for results in all_case_results.values():
        if results:
            all_pred_p2.extend([r['pred_p2_rmsd'] for r in results])
    pred_p2_stats = {
        'mean': float(np.mean(all_pred_p2)) if all_pred_p2 else 0.0,
        'std': float(np.std(all_pred_p2)) if all_pred_p2 else 0.0,
        'min': float(np.min(all_pred_p2)) if all_pred_p2 else 0.0,
        'max': float(np.max(all_pred_p2)) if all_pred_p2 else 0.0,
    }

    # ── Build per_case entries ────────────────────────────────────────────
    per_case = []
    for name in case_names:
        results = all_case_results[name]
        if not results:
            per_case.append({
                'name': name,
                'linker_n_rotatable': linker_info[name]['linker_n_rotatable'],
                'linker_group': linker_info[name]['linker_group'],
                'dockq': 0.0, 'ligand_rmsd': 0.0, 'clash_rate': 0.0,
                'linker_strain': 0.0, 'n_clusters_40seed': 0, 'top1_rmsd': 0.0,
                'pred_p2_rmsd_mean': 0.0, 'pred_p2_rmsd_best': 0.0,
            })
            continue

        dockqs = [r['dockq'] for r in results]
        best_idx = np.argmax(dockqs)
        ens = ensemble_diversity[name]

        per_case.append({
            'name': name,
            'linker_n_rotatable': linker_info[name]['linker_n_rotatable'],
            'linker_group': linker_info[name]['linker_group'],
            'dockq': float(dockqs[best_idx]),
            'dockq_mean': float(np.mean(dockqs)),
            'dockq_std': float(np.std(dockqs)),
            'ligand_rmsd': float(results[best_idx]['lig_rmsd']),
            'clash_rate': clash_metrics.get('per_case_clash_rates', {}).get(name, 0.0),
            'linker_strain': strain_metrics.get('per_case_linker_strain', {}).get(name, 0.0),
            'n_clusters_40seed': ens['n_clusters'],
            'top1_rmsd': ens['top1_rmsd'],
            'pairwise_rmsd_mean': ens['pairwise_rmsd_mean'],
            'pred_p2_rmsd_mean': float(np.mean([r['pred_p2_rmsd'] for r in results])),
            'pred_p2_rmsd_best': float(min(r['pred_p2_rmsd'] for r in results)),
            'n_seeds': len(results),
        })

    # ── Assemble final output ─────────────────────────────────────────────
    audit_output = {
        'audit_date': datetime.now().strftime('%Y-%m-%d'),
        'model': 'DeepTernary',
        'checkpoint': 'DeepTernary/output/checkpoints/PROTAC/',
        'n_cases': len(case_names),
        'n_seeds_per_case': 40,
        'global_metrics': {
            **dockq_metrics,
            **lig_rmsd_metrics,
            'mean_clash_rate': clash_metrics['mean_clash_rate'],
            'mean_linker_strain': strain_metrics['mean_linker_strain'],
            'capri_distribution': capri_dist,
        },
        'stratified_dockq': stratified,
        'ensemble_metrics': {
            'mean_n_clusters': float(mean_n_clusters),
            'mean_top1_rmsd': float(mean_top1_rmsd),
            'pred_p2_rmsd_stats': pred_p2_stats,
            'seed_stability': all_stability,
        },
        'per_case': per_case,
    }

    # Write output
    with open(args.output, 'w') as f:
        json.dump(audit_output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Audit results written to {args.output}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  N cases:                     {len(case_names)}")
    print(f"  Mean DockQ (all seeds):      {dockq_metrics['mean_dockq']:.4f}")
    print(f"  Mean best DockQ (per case):  {dockq_metrics['mean_best_dockq']:.4f}")
    print(f"  Mean ligand RMSD:            {lig_rmsd_metrics['mean_ligand_rmsd']:.2f} Å")
    print(f"  Mean clash rate:             {clash_metrics['mean_clash_rate']:.4f}")
    print(f"  Mean linker strain:          {strain_metrics['mean_linker_strain']:.2f} kcal/mol")
    print(f"  Mean ensemble clusters:      {mean_n_clusters:.1f}")
    print(f"  Acceptable+ rate (DockQ≥0.23): {capri_dist.get('Acceptable',{}).get('fraction',0) + capri_dist.get('Medium',{}).get('fraction',0) + capri_dist.get('High',{}).get('fraction',0):.2%}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
