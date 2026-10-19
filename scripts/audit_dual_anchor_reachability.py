"""Audit whether released binary warheads can constrain a PROTAC search.

The input audit uses only the full PROTAC graph and observed binary ligands.
The optional posthoc section may read the deposited full-ligand coordinates;
none of its measurements may be used for candidate generation or ranking.
"""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdmolops
from rdkit.DistanceGeometry.DistGeom import DoTriangleSmoothing

from scripts.audit_protac_anchor_chemistry import ccd_fragment, match_graph
from tricoopdiff.initialization.protac import match_external_binary_anchors
from tricoopdiff.metrics.dockq import compute_fullatom_pose_dockq


def _fragment(entry, ccd_dir, side):
    path = Path(entry['protein1']).parent / f'unbound_lig{side}.pdb'
    component = next(line[17:20].strip() for line in path.read_text().splitlines()
                     if line.startswith('HETATM'))
    return ccd_fragment(path, ccd_dir / f'{component}.cif')


def _shape_rmsd(reference, observed):
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    x, y = reference - reference.mean(0), observed - observed.mean(0)
    left, _, right = np.linalg.svd(x.T @ y)
    diagonal = np.diag([1., 1., np.linalg.det(left @ right)])
    return float(np.sqrt(np.mean(np.sum((x @ left @ diagonal @ right-y)**2, axis=1))))


def _protein_ca(path):
    """CA coordinates keyed by residue number and identity, ignoring chain name."""
    result = {}
    for line in path.read_text().splitlines():
        if (line.startswith('ATOM') and line[12:16].strip() == 'CA'
                and line[16] in (' ', 'A')):
            result[(line[22:27], line[17:20])] = np.asarray(
                [float(line[start:start+8]) for start in (30, 38, 46)])
    return result


def _protein_to_native_transform(unbound, bound):
    from_xyz, to_xyz = _protein_ca(unbound), _protein_ca(bound)
    common = sorted(from_xyz.keys() & to_xyz.keys())
    if len(common) < 20:
        raise ValueError(f'only {len(common)} common numbered CA residues')
    source = np.asarray([from_xyz[key] for key in common])
    target = np.asarray([to_xyz[key] for key in common])
    source_center, target_center = source.mean(0), target.mean(0)
    left, _, right = np.linalg.svd((source-source_center).T @
                                    (target-target_center))
    rotation = left @ np.diag([1., 1., np.linalg.det(left @ right)]) @ right
    shift = target_center-source_center @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum(
        (source @ rotation+shift-target)**2, axis=1))))
    return rotation, shift, len(common), rmsd


def _raw_binary_frame_dockq(entry):
    """Posthoc-only check for target-relative pose hidden in released PDB frames."""
    directory = Path(entry['protein1']).parent
    lines = []
    for side, chain in ((1, 'A'), (2, 'B')):
        for line in (directory / f'unbound_protein{side}.pdb').read_text().splitlines():
            if line.startswith('ATOM'):
                lines.append(line[:21] + chain + line[22:])
    with tempfile.TemporaryDirectory(prefix='raw_binary_frame_') as temporary:
        path = Path(temporary) / 'proteins.pdb'
        path.write_text('\n'.join(lines) + '\nEND\n')
        metric = compute_fullatom_pose_dockq(
            path, entry['complex'], Path(__file__).resolve().parents[1] / 'DeepTernary')
    return {key: float(metric[key]) for key in ('dockq', 'fnat', 'irms', 'lrms')}


def _sample_linker_span(full, endpoints, count, seed):
    """Sample graph-only conformers; this is not proof of geometric reachability."""
    molecule = Chem.AddHs(Chem.Mol(full))
    parameters = AllChem.ETKDGv3()
    parameters.numThreads = 1
    parameters.randomSeed = seed
    parameters.pruneRmsThresh = -1
    ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=count,
                                         params=parameters))
    distances = [float(np.linalg.norm(
        np.asarray(molecule.GetConformer(i).GetAtomPosition(endpoints[0]))-
        np.asarray(molecule.GetConformer(i).GetAtomPosition(endpoints[1]))))
        for i in ids]
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    a, b = sorted(endpoints)
    return {'sampled_conformer_count': len(ids),
            'sampled_attachment_span_min_A': min(distances) if distances else None,
            'sampled_attachment_span_max_A': max(distances) if distances else None,
            'graph_distance_lower_bound_A': float(bounds[b, a]),
            'graph_distance_upper_bound_A': float(bounds[a, b]),
            'linker_graph_bonds': len(rdmolops.GetShortestPath(full, *endpoints))-1}


def _oracle_closure(full, mapping, protein_fits, native_coords, count, seed):
    """Retrospective linker closure at native PPI geometry; never an input."""
    def anchor_data(hypothesis):
        indices, coordinates = [], []
        for side in (1, 2):
            rotation, shift, _, _ = protein_fits[side-1]
            indices.extend(hypothesis[f'anchor{side}_atoms'])
            coordinates.extend(np.asarray(hypothesis[f'anchor{side}_xyz']) @
                               rotation+shift)
        return np.asarray(indices, dtype=int), np.asarray(coordinates)

    # Native-selected graph symmetry is an oracle diagnostic, not a mapping
    # allowed in deployment.  It avoids confusing symmetry with closure failure.
    choice = min(mapping['mapping_hypotheses'], key=lambda hypothesis: sum(
        np.sum((np.asarray(hypothesis[f'anchor{side}_xyz']) @
                protein_fits[side-1][0]+protein_fits[side-1][1]-
                native_coords[list(hypothesis[f'anchor{side}_atoms'])])**2)
        for side in (1, 2)))
    indices, target = anchor_data(choice)
    molecule = Chem.AddHs(Chem.Mol(full))
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(molecule)
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            a, b = sorted((int(indices[i]), int(indices[j])))
            distance = float(np.linalg.norm(target[i]-target[j]))
            bounds[a, b] = distance+.75
            bounds[b, a] = max(.10, distance-.75)
    if not DoTriangleSmoothing(bounds):
        return {'status': 'inconsistent_pair_distance_bounds',
                'attempted_conformers': count, 'embedded_conformers': 0}
    parameters = AllChem.ETKDGv3()
    parameters.SetBoundsMat(bounds)
    parameters.numThreads = 1
    parameters.randomSeed = seed
    parameters.pruneRmsThresh = -1
    conformers = list(AllChem.EmbedMultipleConfs(
        molecule, numConfs=count, params=parameters))
    if not conformers:
        return {'status': 'embedding_failed',
                'attempted_conformers': count, 'embedded_conformers': 0}
    properties = AllChem.MMFFGetMoleculeProperties(molecule)
    errors, energies = [], []
    for conformer_id in conformers:
        points = molecule.GetConformer(conformer_id).GetPositions()[indices]
        errors.append(_shape_rmsd(points, target))
        if properties is not None:
            forcefield = AllChem.MMFFGetMoleculeForceField(
                molecule, properties, confId=conformer_id)
            energies.append(float(forcefield.CalcEnergy()))
    return {'status': 'embedded_connected_linker',
            'attempted_conformers': count,
            'embedded_conformers': len(conformers),
            'best_joint_anchor_fit_rmsd_A': min(errors),
            'median_joint_anchor_fit_rmsd_A': float(np.median(errors)),
            'min_unrelaxed_mmff_energy': min(energies) if energies else None,
            'interpretation': 'DG closure only; does not establish protein-clash-free or low-strain pose'}


def audit_case(entry, ccd_dir, count, seed, posthoc_native,
               oracle_closure=False, mcs_timeout=15):
    full = Chem.MolFromMolFile(str(entry['ligand']), removeHs=True)
    if full is None:
        raise ValueError('full ligand graph cannot be read')
    native_coords = full.GetConformer().GetPositions().copy() if posthoc_native else None
    full.RemoveAllConformers()
    fragments, sides = [], []
    for side in (1, 2):
        fragment, chemistry = _fragment(entry, ccd_dir, side)
        _, graph = match_graph(full, fragment, timeout=mcs_timeout)
        fragments.append(fragment)
        sides.append({'side': side, 'component': chemistry['component'],
                      'core_coverage': graph['core_coverage'],
                      'common_core_atoms': graph['common_core_atoms'],
                      'binary_heavy_atoms': fragment.GetNumAtoms(),
                      'missing_ccd_atoms': len(chemistry['missing_ccd_atoms']),
                      'mcs_timed_out': graph['search_timed_out']})
    row = {'case': entry['case'], 'sides': sides,
           'native_free_input': True, 'posthoc_native_enabled': posthoc_native}
    if posthoc_native:
        row['posthoc_raw_binary_frame'] = _raw_binary_frame_dockq(entry)
        row['posthoc_raw_binary_frame']['interpretation'] = (
            'diagnostic leakage check only; never use this relative pose as a prediction')
    if any(side['mcs_timed_out'] for side in sides):
        row['input_status'] = 'unresolved_chemical_match_timeout'
    elif any(side['core_coverage'] < .75 for side in sides):
        row['input_status'] = 'binary_reference_chemical_mismatch'
    else:
        mapping = match_external_binary_anchors(full, *fragments,
                                                min_coverage=.75,
                                                timeout=mcs_timeout)
        if mapping is None:
            row['input_status'] = 'no_unique_disjoint_attachment_mapping'
        else:
            row['input_status'] = 'chemically_mappable'
            row['mapping_hypotheses'] = mapping['symmetry_mapping_count']
            row['attachment_atoms'] = list(mapping['attachment_atoms'])
            row['bridge_atoms'] = len(mapping['bridge_atoms'])
            row['linker_span'] = _sample_linker_span(
                full, mapping['attachment_atoms'], count, seed)
            # Unknown without a candidate protein pose.  The linker graph
            # alone cannot decide whether a *particular* PPI is reachable.
            row['input_pose_reachability'] = 'undetermined_without_protein_relative_pose'
            if posthoc_native:
                a, b = mapping['attachment_atoms']
                directory = Path(entry['protein1']).parent
                protein_fits = [
                    _protein_to_native_transform(
                        directory / f'unbound_protein{side}.pdb',
                        directory / f'protein{side}.pdb')
                    for side in (1, 2)]
                # This minimum is explicitly an oracle *audit* over chemically
                # symmetric atom mappings. It must not select an input pose.
                anchor_rmsds = []
                for side in (1, 2):
                    rotation, shift, _, _ = protein_fits[side-1]
                    anchor_rmsds.append(min(
                        float(np.sqrt(np.mean(np.sum((
                            np.asarray(hypothesis[f'anchor{side}_xyz']) @ rotation
                            + shift - native_coords[list(
                                hypothesis[f'anchor{side}_atoms'])])**2, axis=1))))
                        for hypothesis in mapping['mapping_hypotheses']))
                row['posthoc_native'] = {
                    'native_attachment_span_A': float(np.linalg.norm(
                        native_coords[a]-native_coords[b])),
                    'protein_ca_common_count': [fit[2] for fit in protein_fits],
                    'unbound_to_bound_protein_ca_rmsd_A': [fit[3] for fit in protein_fits],
                    'binary_anchor_to_native_after_protein_alignment_rmsd_A': anchor_rmsds,
                    'binary_to_native_core_shape_rmsd_A': [
                        min(_shape_rmsd(native_coords[list(hypothesis[f'anchor{side}_atoms'])],
                                        hypothesis[f'anchor{side}_xyz'])
                            for hypothesis in mapping['mapping_hypotheses'])
                        for side in (1, 2)],
                    'meaning': 'retrospective native conformer audit only; not an input'}
                if oracle_closure:
                    row['posthoc_native']['oracle_linker_closure'] = _oracle_closure(
                        full, mapping, protein_fits, native_coords, count, seed)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--ccd-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--conformers', type=int, default=24)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--mcs-timeout', type=int, default=15)
    parser.add_argument('--posthoc-native', action='store_true')
    parser.add_argument('--oracle-closure', action='store_true')
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    if args.conformers < 1 or args.mcs_timeout < 1:
        parser.error('--conformers and --mcs-timeout must be positive')
    if args.oracle_closure and not args.posthoc_native:
        parser.error('--oracle-closure requires --posthoc-native')
    rows = []
    entries = json.loads(args.index.read_text())['entries']
    if args.case:
        selected = set(args.case)
        entries = [entry for entry in entries if entry['case'] in selected]
        if len(entries) != len(selected):
            parser.error('one or more --case values are absent from --index')
    for entry in entries:
        try:
            row = audit_case(entry, args.ccd_dir, args.conformers,
                             args.seed, args.posthoc_native,
                             args.oracle_closure, args.mcs_timeout)
        except Exception as error:
            row = {'case': entry['case'], 'input_status': 'audit_error',
                   'error': f'{type(error).__name__}: {error}'}
        rows.append(row)
        print(entry['case'], row['input_status'], flush=True)
    report = {'protocol': 'dual_anchor_input_reachability_audit_v1',
              'posthoc_native_enabled': args.posthoc_native,
              'case_count': len(rows), 'cases': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
