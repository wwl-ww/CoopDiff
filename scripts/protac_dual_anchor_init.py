"""Generate a tiny native-free PROTAC pose pool from two binary anchors.

The full PROTAC contributes its chemical graph only. Each binary ligand PDB
contributes the observed local warhead geometry and its matching protein frame.
The linker conformer determines the relative placement of protein 2. Native
complex coordinates are read only after candidate selection, for DockQ scoring.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdMolTransforms, rdmolops
from rdkit.DistanceGeometry.DistGeom import DoTriangleSmoothing
from scipy.spatial import cKDTree

from scripts.audit_protac_anchor_chemistry import ccd_fragment
from tricoopdiff.initialization.protac import match_external_binary_anchors
from tricoopdiff.metrics.dockq import compute_fullatom_pose_dockq


def _protein_records(path):
    lines, coords, residue_keys, radii = [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith('ATOM') or line[16] not in (' ', 'A'):
            continue
        if line[76:78].strip().upper() in ('H', 'D'):
            continue
        lines.append(line)
        coords.append([float(line[start:start + 8]) for start in (30, 38, 46)])
        residue_keys.append((line[21], line[22:27], line[17:20]))
        element = line[76:78].strip().capitalize() or line[12:16].strip()[0]
        radii.append(Chem.GetPeriodicTable().GetRvdw(
            Chem.GetPeriodicTable().GetAtomicNumber(element)))
    if not coords:
        raise ValueError(f'no protein heavy atoms in {path}')
    return lines, np.asarray(coords, dtype=float), residue_keys, np.asarray(radii)


def _pairs(left, right, cutoff):
    return cKDTree(left).sparse_distance_matrix(
        cKDTree(right), cutoff, output_type='coo_matrix')


def _vdw_overlaps(pairs, left_radii, right_radii):
    """Count inter-component pairs closer than 85% of summed vdW radii."""
    return int((pairs.data < .85 * (left_radii[pairs.row]
                                     + right_radii[pairs.col])).sum())


def _fit_rigid(source, target):
    """Fit row-vector coordinates with a proper rotation."""
    source_center, target_center = source.mean(0), target.mean(0)
    left, _, right = np.linalg.svd(
        (source-source_center).T @ (target-target_center))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    shift = target_center-source_center @ rotation
    rmsd = np.sqrt(np.mean(np.sum((source @ rotation+shift-target)**2, axis=1)))
    return rotation, shift, float(rmsd)


def _assemble_mapping(ligand, protein2, hypothesis):
    first, second = hypothesis['anchor1_atoms'], hypothesis['anchor2_atoms']
    rotation1, shift1, error1 = _fit_rigid(
        ligand[list(first)], np.asarray(hypothesis['anchor1_xyz']))
    placed_ligand = ligand @ rotation1+shift1
    rotation2, shift2, error2 = _fit_rigid(
        np.asarray(hypothesis['anchor2_xyz']), placed_ligand[list(second)])
    return placed_ligand, protein2 @ rotation2+shift2, error1, error2


def _evaluate_mappings(ligand, protein1, protein2, protein1_residues,
                       protein2_residues, protein1_radii, protein2_radii,
                       ligand_radii, mapping, energy, indices=None,
                       strict_vdw=False, check_all_ligand=False):
    """Choose a graph-symmetric anchor assignment using only input geometry."""
    hypotheses = mapping['mapping_hypotheses']
    best = None
    for index in range(len(hypotheses)) if indices is None else indices:
        placed_ligand, placed2, error1, error2 = _assemble_mapping(
            ligand, protein2, hypotheses[index])
        if max(error1, error2) > 1.:
            continue
        pp = _pairs(protein1, placed2, 8.)
        # Do not relabel observed binary-pocket contacts as clashes introduced
        # by ternary assembly; only the newly placed ligand atoms are tested.
        new_ligand1 = (list(range(len(ligand))) if check_all_ligand else
                       sorted(set(range(len(ligand)))
                              - set(hypotheses[index]['anchor1_atoms'])))
        new_ligand2 = (list(range(len(ligand))) if check_all_ligand else
                       sorted(set(range(len(ligand)))
                              - set(hypotheses[index]['anchor2_atoms'])))
        new_pl1 = _pairs(protein1, placed_ligand[new_ligand1], 4.)
        new_pl2 = _pairs(placed2, placed_ligand[new_ligand2], 4.)
        severe_pp = int((pp.data < 2.).sum())
        severe_pl = int((new_pl1.data < 2.).sum()
                        + (new_pl2.data < 2.).sum())
        vdw_overlaps = (_vdw_overlaps(pp, protein1_radii, protein2_radii)
                        + _vdw_overlaps(new_pl1, protein1_radii,
                                        ligand_radii[new_ligand1])
                        + _vdw_overlaps(new_pl2, protein2_radii,
                                        ligand_radii[new_ligand2]))
        close = int((pp.data < 2.5).sum()
                    + (new_pl1.data < 2.5).sum()
                    + (new_pl2.data < 2.5).sum())
        residue_pairs = {(protein1_residues[i], protein2_residues[j])
                         for i, j in zip(pp.row, pp.col)}
        score = (energy + 250.*(severe_pp+severe_pl)
                 + (200.*vdw_overlaps if strict_vdw else 0.) + 20.*close
                 - len(residue_pairs) + 100.*(error1+error2))
        invalid = (vdw_overlaps > 0 if strict_vdw
                   else severe_pp+severe_pl > 0)
        current = ((invalid, score), index,
                   placed_ligand, placed2, error1, error2,
                   severe_pp, severe_pl, vdw_overlaps, len(residue_pairs), score)
        if best is None or current[0] < best[0]:
            best = current
    return best


def _linker_dihedrals(molecule, attachment_atoms):
    """Rotatable bonds on the chemical path joining the two binary warheads."""
    path = rdmolops.GetShortestPath(molecule, *attachment_atoms)
    dihedrals = []
    for index in range(len(path)-1):
        left, right = path[index:index+2]
        bond = molecule.GetBondBetweenAtoms(left, right)
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        before = (path[index-1] if index else next(
            (atom.GetIdx() for atom in molecule.GetAtomWithIdx(left).GetNeighbors()
             if atom.GetIdx() != right and atom.GetAtomicNum() > 1), None))
        after = (path[index+2] if index+2 < len(path) else next(
            (atom.GetIdx() for atom in molecule.GetAtomWithIdx(right).GetNeighbors()
             if atom.GetIdx() != left and atom.GetAtomicNum() > 1), None))
        if before is not None and after is not None:
            dihedrals.append((before, left, right, after))
    return dihedrals


def _refine_joint_pose(molecule, conformer_id, properties, hypothesis,
                       attachment_atoms, protein1, protein2, *, steps, seed):
    """Move linker torsions while scoring the resulting ternary pose."""
    work = Chem.Mol(molecule)
    conformer = work.GetConformer(conformer_id)
    dihedrals = _linker_dihedrals(work, attachment_atoms)
    if not dihedrals or steps < 1:
        return conformer.GetPositions().copy(), 0
    heavy = Chem.RemoveHs(work).GetNumAtoms()
    noncore1 = sorted(set(range(heavy))-set(hypothesis['anchor1_atoms']))
    noncore2 = sorted(set(range(heavy))-set(hypothesis['anchor2_atoms']))
    random = np.random.default_rng(seed)
    initial_field = AllChem.MMFFGetMoleculeForceField(
        work, properties, confId=conformer_id)
    initial_energy = float(initial_field.CalcEnergy())

    def score():
        coordinates = conformer.GetPositions()[:heavy]
        ligand, placed2, error1, error2 = _assemble_mapping(
            coordinates, protein2, hypothesis)
        if max(error1, error2) > 1.:
            return 1e6
        pp = _pairs(protein1, placed2, 8.).data
        pl1 = _pairs(protein1, ligand[noncore1], 4.).data
        pl2 = _pairs(placed2, ligand[noncore2], 4.).data
        overlap = sum(np.square(np.clip(3.-distances, 0., None)).sum()
                      for distances in (pp, pl1, pl2))
        contacts = np.exp(-np.square((pp-4.8)/1.2)).sum()
        field = AllChem.MMFFGetMoleculeForceField(
            work, properties, confId=conformer_id)
        if field is None:
            return 1e6
        energy = float(field.CalcEnergy())
        if energy > initial_energy + 75.:
            return 1e6 + energy-initial_energy
        return float(energy + 30.*overlap - .06*contacts)

    current = best = score()
    best_coordinates = conformer.GetPositions().copy()
    for iteration in range(steps):
        indices = dihedrals[int(random.integers(len(dihedrals)))]
        previous = rdMolTransforms.GetDihedralDeg(conformer, *indices)
        delta = float(random.normal(0., 30. if random.random() < .8 else 90.))
        rdMolTransforms.SetDihedralDeg(conformer, *indices, previous+delta)
        proposed = score()
        temperature = 12.*(1.-iteration/steps)+.5
        if proposed < current or random.random() < np.exp(
                min(0., (current-proposed)/temperature)):
            current = proposed
            if proposed < best:
                best = proposed
                best_coordinates = conformer.GetPositions().copy()
        else:
            rdMolTransforms.SetDihedralDeg(conformer, *indices, previous)
    return best_coordinates, steps+1


def _binary_ligand(path, ccd_dir):
    lines = [line for line in Path(path).read_text().splitlines()
             if line.startswith('HETATM') and line[76:78].strip().upper() not in ('H', 'D')]
    if not lines:
        raise ValueError(f'no binary ligand heavy atoms in {path}')
    component = lines[0][17:20].strip()
    molecule, _ = ccd_fragment(path, Path(ccd_dir) / f'{component}.cif')
    return molecule


def _embed_dual_anchor_conformers(full, mapping, count, seed):
    hypothesis = mapping['mapping_hypotheses'][0]
    anchors = ((hypothesis['anchor1_atoms'], hypothesis['anchor1_xyz']),
               (hypothesis['anchor2_atoms'], hypothesis['anchor2_xyz']))
    molecule = Chem.AddHs(Chem.Mol(full))
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    for atom_indices, anchor_xyz in anchors:
        atom_indices = np.asarray(atom_indices, dtype=int)
        anchor_xyz = np.asarray(anchor_xyz, dtype=float)
        for left in range(len(atom_indices)):
            for right in range(left + 1, len(atom_indices)):
                i, j = sorted((int(atom_indices[left]), int(atom_indices[right])))
                distance = float(np.linalg.norm(anchor_xyz[left] - anchor_xyz[right]))
                bounds[i, j], bounds[j, i] = distance + .10, max(.10, distance - .10)
    if not DoTriangleSmoothing(bounds):
        raise ValueError('the two observed binary anchor geometries are inconsistent')
    parameters = AllChem.ETKDGv3()
    parameters.SetBoundsMat(bounds)
    parameters.numThreads = 1
    parameters.randomSeed = int(seed) % (2**31 - 1) or 1
    parameters.pruneRmsThresh = -1
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=count,
                                                    params=parameters))
    if len(conformer_ids) != count:
        # Long, highly flexible PROTACs can fail distance-geometry placement
        # from RDKit's standard coordinates even when the same constrained
        # bounds are feasible. Random-coordinate placement retains the bounds.
        molecule.RemoveAllConformers()
        parameters.useRandomCoords = True
        parameters.maxIterations = 1000
        conformer_ids = list(AllChem.EmbedMultipleConfs(
            molecule, numConfs=count, params=parameters))
    if len(conformer_ids) != count:
        raise ValueError(f'ETKDG generated {len(conformer_ids)}/{count} requested conformers')
    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    if properties is None:
        raise ValueError('MMFF parameters are unavailable for this PROTAC graph')
    return molecule, conformer_ids, properties, anchors


def generate_case(entry, ccd_dir, outdir, count=5, seed=73, refine_steps=0,
                  search_conformers=None):
    case = entry['case']
    directory = Path(entry['protein1']).parent
    full = Chem.MolFromMolFile(str(entry['ligand']), removeHs=True, sanitize=True)
    if full is None:
        raise ValueError(f'{case}: PROTAC graph cannot be read')
    full.RemoveAllConformers()  # Discard any deposited full-ligand pose.
    binary = [_binary_ligand(directory / f'unbound_lig{side}.pdb', ccd_dir)
              for side in (1, 2)]
    mapping = match_external_binary_anchors(full, *binary, min_coverage=.75)
    if mapping is None:
        raise ValueError(f'{case}: no disjoint exact-bond binary anchor mapping')
    search_count = search_conformers or count
    molecule, conformer_ids, properties, _ = _embed_dual_anchor_conformers(
        full, mapping, search_count, seed)
    hypothesis = mapping['mapping_hypotheses'][0]
    anchor1_atoms = np.asarray(hypothesis['anchor1_atoms'], dtype=int)
    anchor2_atoms = np.asarray(hypothesis['anchor2_atoms'], dtype=int)
    p1_path, p2_path = directory / 'unbound_protein1.pdb', directory / 'unbound_protein2.pdb'
    p1_lines, p1_xyz, p1_residues, p1_radii = _protein_records(p1_path)
    p2_lines, p2_xyz, p2_residues, p2_radii = _protein_records(p2_path)
    ligand_radii = np.asarray([Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())
                               for atom in full.GetAtoms()])
    # DeepTernary's released gt_complex normalizes protein roles to A/B;
    # case-key chain letters name the source PDB, not the released metric file.
    p1_chain, p2_chain = 'A', 'B'
    case_dir = Path(outdir) / case
    case_dir.mkdir(parents=True, exist_ok=True)
    searched = []
    for rank, conformer_id in enumerate(conformer_ids, 1):
        conformer = molecule.GetConformer(conformer_id)
        ligand_xyz = np.asarray(conformer.GetPositions(), dtype=float)[:full.GetNumAtoms()]
        forcefield = AllChem.MMFFGetMoleculeForceField(
            molecule, properties, confId=conformer_id)
        energy = float(forcefield.CalcEnergy())
        best_mapping = _evaluate_mappings(
            ligand_xyz, p1_xyz, p2_xyz, p1_residues, p2_residues,
            p1_radii, p2_radii, ligand_radii, mapping, energy,
            strict_vdw=not refine_steps)
        if best_mapping is None:
            continue
        search_evaluations = 0
        if refine_steps:
            refined_xyz, search_evaluations = _refine_joint_pose(
                molecule, conformer_id, properties,
                mapping['mapping_hypotheses'][best_mapping[1]],
                mapping['attachment_atoms'], p1_xyz, p2_xyz,
                steps=refine_steps, seed=seed+rank*9973)
            ligand_xyz = refined_xyz[:full.GetNumAtoms()]
            for atom_index, point in enumerate(refined_xyz):
                conformer.SetAtomPosition(atom_index, point)
            field = AllChem.MMFFGetMoleculeForceField(
                molecule, properties, confId=conformer_id)
            energy = float(field.CalcEnergy())
            best_mapping = _evaluate_mappings(
                ligand_xyz, p1_xyz, p2_xyz, p1_residues, p2_residues,
                p1_radii, p2_radii, ligand_radii, mapping, energy,
                indices=(best_mapping[1],), strict_vdw=True)
        (_, mapping_index, placed_ligand, placed_p2, error1, error2,
         severe_pp, severe_pl, vdw_overlaps, residue_pair_count, score) = best_mapping
        searched.append(({
            'search_rank': rank, 'conformer_id': int(conformer_id),
            'mmff_energy': energy, 'joint_search_evaluations': search_evaluations,
            'mapping_index': mapping_index,
            'anchor1_fit_rmsd': error1, 'anchor2_fit_rmsd': error2,
            'protein_protein_pairs_lt2A': severe_pp,
            'protein_ligand_pairs_lt2A': severe_pl,
            'cross_component_vdw_overlaps': vdw_overlaps,
            'protein_interface_residue_pairs_lt8A': residue_pair_count,
            'selection_score': score,
        }, placed_ligand, placed_p2))
    if not searched:
        raise ValueError('no generated conformer fits both binary warheads')
    searched.sort(key=lambda item: (
        item[0]['cross_component_vdw_overlaps'] > 0,
        item[0]['selection_score']))
    candidates = []
    for rank, (row, placed_ligand, placed_p2) in enumerate(searched[:count], 1):
        row['rank'] = rank
        ligand_mol = Chem.Mol(full)
        ligand_conf = Chem.Conformer(ligand_mol.GetNumAtoms())
        for atom_index, point in enumerate(placed_ligand):
            ligand_conf.SetAtomPosition(atom_index, point.tolist())
        ligand_mol.AddConformer(ligand_conf)
        sdf_path = case_dir / f'candidate_{rank}.sdf'
        with Chem.SDWriter(str(sdf_path)) as writer:
            writer.write(ligand_mol)
        protein_path = case_dir / f'candidate_{rank}_proteins.pdb'
        with protein_path.open('w') as handle:
            for chain, lines, points in ((p1_chain, p1_lines, None),
                                         (p2_chain, p2_lines, placed_p2)):
                for index, line in enumerate(lines):
                    if points is None:
                        handle.write(line[:21] + chain + line[22:] + '\n')
                    else:
                        point = points[index]
                        handle.write(line[:21] + chain + line[22:30]
                                     + ''.join(f'{v:8.3f}' for v in point)
                                     + line[54:] + '\n')
            handle.write('END\n')
        row.update(sdf=str(sdf_path), protein_pdb=str(protein_path))
        candidates.append(row)
    physical_candidates = [row for row in candidates
                           if row['cross_component_vdw_overlaps'] == 0]
    selected = (min(physical_candidates, key=lambda row: row['selection_score'])
                if physical_candidates else None)
    # DockQ is evaluated only after the native-blind winner is fixed.
    native = Path(entry['complex'])
    if native.is_file():
        for row in candidates:
            metric = compute_fullatom_pose_dockq(
                row['protein_pdb'], native, Path(__file__).resolve().parents[1] / 'DeepTernary')
            row.update(fnat=metric['fnat'], irmsd=metric['irms'],
                       lrmsd=metric['lrms'], dockq=metric['dockq'])
    return {
        'protocol': 'native_free_dual_binary_core_distances_linker_kinematics_v2',
        'case': case, 'candidate_count': len(candidates),
        'search_conformer_count': search_count,
        'search_anchor_fit_count': len(searched),
        'search_physical_valid_count': sum(
            row['cross_component_vdw_overlaps'] == 0
            for row, _, _ in searched),
        'joint_refine_steps_per_candidate': refine_steps,
        'candidate_mapping_hypotheses': mapping['symmetry_mapping_count'],
        'mapping_policy': 'enumerate every graph-symmetric mapping; native-blind physical score',
        'binary_core_coverage': list(mapping['binary_core_coverage']),
        'anchor_core_sizes': [len(anchor1_atoms), len(anchor2_atoms)],
        'linker_atom_count': len(mapping['bridge_atoms']),
        'selection_rule': 'require zero cross-component vdW overlaps, then minimize MMFF+clash+interface score; never DockQ',
        'physical_validity_threshold': 'zero new cross-component heavy-atom pairs <0.85 * summed element vdW radii; observed binary-core pocket contacts excluded',
        'physical_valid_candidates': sum(
            row['cross_component_vdw_overlaps'] == 0 for row in candidates),
        'selected_rank': selected['rank'] if selected is not None else None,
        'selected_dockq': selected.get('dockq') if selected is not None else None,
        'selection_status': ('selected_physical_candidate'
                             if selected is not None else 'no_physical_candidate'),
        'mean_candidate_dockq': (float(np.mean([row['dockq'] for row in candidates]))
                                 if all('dockq' in row for row in candidates) else None),
        'candidates': candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--ccd-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--search-conformers', type=int)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--refine-steps', type=int, default=0)
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    if (args.candidates < 1 or args.refine_steps < 0
            or args.search_conformers is not None
            and args.search_conformers < args.candidates):
        parser.error('need 1 <= candidates <= search-conformers and nonnegative refine-steps')
    entries = json.loads(args.index.read_text())['entries']
    if args.case:
        selected = set(args.case)
        entries = [entry for entry in entries if entry['case'] in selected]
        if len(entries) != len(selected):
            raise ValueError('one or more requested cases are absent from the index')
    rows = []
    for entry in entries:
        try:
            row = generate_case(entry, args.ccd_dir, args.out,
                                count=args.candidates, seed=args.seed,
                                refine_steps=args.refine_steps,
                                search_conformers=args.search_conformers)
        except Exception as error:
            row = {'case': entry['case'], 'status': 'failed',
                   'error': f'{type(error).__name__}: {error}'}
        rows.append(row)
        print(f"{entry['case']} {row.get('status', 'complete')} "
              f"selected_DockQ={row.get('selected_dockq')} "
              f"error={row.get('error')}", flush=True)
        case_dir = args.out / entry['case']
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'summary.json').write_text(
            json.dumps(row, indent=2, allow_nan=False) + '\n')
    complete = [row for row in rows if row.get('status', 'complete') == 'complete']
    selected_dockq = [row['selected_dockq'] for row in complete
                      if row.get('selected_dockq') is not None]
    aggregate = {
        'protocol': 'native_free_dual_binary_core_distances_linker_kinematics_v2',
        'requested_cases': len(entries), 'completed_cases': len(complete),
        'failed_cases': [row['case'] for row in rows if row.get('status') == 'failed'],
        'candidate_count_per_case': args.candidates,
        'search_conformers_per_case': args.search_conformers or args.candidates,
        'selected_scored_cases': len(selected_dockq),
        'selected_mean_dockq_posthoc': (float(np.mean(selected_dockq))
                                        if selected_dockq else None),
        'selected_mean_dockq_failures_as_zero_posthoc': (
            float(sum(selected_dockq) / len(entries)) if entries else None),
        'selected_dockq_at_least_0_4_cases': sum(
            score >= .4 for score in selected_dockq),
        'search_anchor_fit_count': sum(
            row['search_anchor_fit_count'] for row in complete),
        'search_physical_valid_count': sum(
            row['search_physical_valid_count'] for row in complete),
        'selected_physical_valid_rate': (float(np.mean([
            row.get('selection_status') == 'selected_physical_candidate'
            for row in rows])) if rows else None),
        'cases': rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(aggregate, indent=2) + '\n')


if __name__ == '__main__':
    main()
