"""Score saved independent ligand poses with official DockQ 2.1.3.

Native bound receptors are scoring references only, never inference inputs.
Retain each prediction and failure; do not select/re-dock using native poses.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.metrics.dockq import _featurized_ca_coords, _fit_rigid, _records


def write_complex(path, protein_path, rotation, translation, mol, ligand_coords):
    """Use explicit A=receptor, B=ligand chains and identical ligand atom IDs."""
    lines, indices, coordinates = _records(protein_path, 'ATOM')
    atoms = coordinates @ rotation + translation
    records = []
    for index, xyz in zip(indices, atoms):
        line = lines[index]
        records.append(f'{line[:21]}A{line[22:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}')
    records.append('TER')
    if mol.GetNumAtoms() != len(ligand_coords):
        raise ValueError('ligand atom count mismatch')
    for i, (atom, xyz) in enumerate(zip(mol.GetAtoms(), ligand_coords), start=1):
        if atom.GetAtomicNum() == 1:
            raise ValueError('expected cached heavy-atom molecule')
        name = f'L{i}'
        records.append(f'HETATM{i:5d} {name:>4s} LIG B   1    '
                       f'{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}'
                       f'  1.00  0.00          {atom.GetSymbol():>2s}')
    path.write_text('\n'.join(records) + '\nEND\n')


def score(model, native):
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    predicted = load_PDB(str(model), small_molecule=True)
    reference = load_PDB(str(native), small_molecule=True)
    if len(list(predicted['B'].get_atoms())) != len(list(reference['B'].get_atoms())):
        raise ValueError('DockQ parser lost ligand atoms')
    scores, total = run_on_all_native_interfaces(
        predicted, reference, chain_map={'A': 'A', 'B': 'B'})
    if 'AB' not in scores:
        return {'valid': False, 'reason': 'no_native_receptor_ligand_interface'}
    result = scores['AB']
    finite = math.isfinite(result['LRMSD'])
    diagnostics = {}
    if not finite:
        from DockQ.DockQ import create_graph
        predicted_atoms = list(predicted['B'].get_atoms())
        native_atoms = list(reference['B'].get_atoms())
        graphs = [create_graph(np.array([a.coord for a in atoms]), [a.element for a in atoms])
                  for atoms in (predicted_atoms, native_atoms)]
        edges = [{tuple(sorted(edge)) for edge in graph.edges if edge[0] != edge[1]}
                 for graph in graphs]
        diagnostics = {'inferred_model_edges': len(edges[0]), 'inferred_native_edges': len(edges[1]),
                       'extra_edges_by_atom_index': sorted(edges[0] - edges[1]),
                       'missing_edges_by_atom_index': sorted(edges[1] - edges[0])}
    return {'valid': finite,
            'LRMSD': float(result['LRMSD']) if finite else None,
            'mapping': result.get('mapping'),
            # DockQ uses this internal surrogate for mapping optimization.
            # It is NOT the CAPRI three-term protein-protein DockQ metric.
            'internal_small_molecule_DockQ': float(result['DockQ']),
            'reason': None if finite else 'no_coordinate_inferred_graph_isomorphism',
            'graph_diagnostics': diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--index', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    version = importlib.metadata.version('DockQ')
    if version != '2.1.3':
        raise ValueError(f'expected pinned DockQ 2.1.3, got {version}')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.results)
    rows = json.loads(source.read_text())
    entries = {}
    for index in args.index:
        entries.update({e['case']: e for e in json.loads(Path(index).read_text())['entries']})
    inputs, controls, scored = {}, {}, []
    for row in rows:
        key = (row['case'], row['branch'])
        label = '_'.join(key)
        if key not in inputs:
            payload = torch.load(row['cache'], map_location='cpu', weights_only=False)
            if payload['source_signature'] != row['source_signature']:
                raise ValueError('feature cache changed since prediction')
            entry = entries[row['case']]
            component = 'protein1' if row['branch'] == 'e3_data' else 'protein2'
            protein = Path(entry[component])
            bound = protein.with_name(component + '.pdb') if protein.name.startswith('unbound_') else protein
            if not bound.is_file():
                raise FileNotFoundError(bound)
            rec = payload[row['branch']]['coords'].reshape(-1, 3).numpy()
            ca = _featurized_ca_coords(protein)
            rotation, translation = _fit_rigid(ca, rec)
            frame_error = float(np.max(np.abs(ca @ rotation + translation - rec)))
            if frame_error > 0.002:
                raise ValueError(f'cache/PDB receptor frame mismatch: {frame_error}')
            mol = Chem.Mol(payload['ligand_data']['mol'])
            native_coords = payload['ligand_data']['coords'].reshape(-1, 3).numpy()
            native = output / f'{label}_native.pdb'
            write_complex(native, bound, rotation, translation, mol, native_coords)
            control = score(native, native)
            if not control['valid'] or control['LRMSD'] > 0.001:
                raise ValueError(f'native identity control failed: {control}')
            controls[label] = {'native_identity': control, 'model_receptor': str(protein),
                               'native_receptor': str(bound), 'frame_fit_max_error': frame_error,
                               'ligand_heavy_atoms': mol.GetNumAtoms()}
            inputs[key] = (protein, rotation, translation, mol, native)
        result = {k: row[k] for k in ('case', 'branch', 'seed', 'sample')}
        if not row['valid']:
            result.update(valid=False, LRMSD=None, reason='nonfinite_prediction')
        else:
            coordinate_path = source.parent / f'{label}_{row["sample"]}.pt'
            pos = torch.load(coordinate_path, map_location='cpu', weights_only=False)['final']
            protein, rotation, translation, mol, native = inputs[key]
            model = output / f'{label}_{row["sample"]}_model.pdb'
            write_complex(model, protein, rotation, translation, mol, pos.reshape(-1, 3).numpy())
            result.update(score(model, native))
            result['model_pdb'] = str(model)
        scored.append(result)
        print(json.dumps(result), flush=True)
        (output / 'results.json').write_text(json.dumps(scored, indent=2, allow_nan=False))
    summary = {}
    for key in inputs:
        group = [r for r in scored if (r['case'], r['branch']) == key]
        values = [r['LRMSD'] for r in group if r['valid']]
        summary['_'.join(key)] = {'n_total': len(group), 'n_scored': len(values),
                                  'best_LRMSD': min(values) if values else None,
                                  'mean_LRMSD': float(np.mean(values)) if values else None,
                                  'n_below_2A': sum(v < 2 for v in values)}
    import DockQ.DockQ as implementation
    metadata = {'DockQ_version': version, 'metric': 'pocket-aligned symmetry-corrected ligand RMSD',
                'ranking': 'oracle best-of-4; not confidence-ranked top-1',
                'source_results': str(source),
                'source_results_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'DockQ_source_sha256': hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
                'DockQ_backend': implementation.residue_distances.__module__,
                'command': sys.argv,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'controls': controls, 'summary': summary}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
