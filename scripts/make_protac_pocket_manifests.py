"""Create native-free, pocket-only condition manifests from binary templates.

The input index supplies each unbound protein.  Its sibling
``unbound_lig{1,2}.pdb`` is used only in that protein's local coordinate frame
to mark C-alpha nodes within the requested distance.  No ternary native file,
warhead atom mapping, or cross-protein geometry is read.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem
import prody as pr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from evaluation_common import sha256_file
from tricoopdiff.condition_manifest import (MANIFEST_VERSION, canonical_isomeric_smiles,
                                            load_condition_manifest)
from tricoopdiff.diffdock.datasets.parse_chi import get_coords


def pdb_coordinates(path, *, alpha_carbons=False):
    """Read ordered coordinates without relying on ligand bond perception."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        altloc = line[16:17]
        if altloc not in (' ', 'A'):
            continue
        atom_name = line[12:16].strip()
        if alpha_carbons:
            if not line.startswith('ATOM  ') or atom_name != 'CA':
                continue
        else:
            element = line[76:78].strip().upper()
            if element in ('H', 'D') or (not element and atom_name.upper().startswith(('H', 'D'))):
                continue
        try:
            rows.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError as error:
            raise ValueError(f'{path}: invalid PDB coordinate') from error
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[1:] != (3,) or not len(result) or not np.isfinite(result).all():
        raise ValueError(f'{path}: no finite requested PDB coordinates')
    return result


def receptor_node_coordinates(path):
    """Return C-alpha coordinates in DiffDock receptor-node order.

    Raw ``ATOM CA`` rows are not sufficient: ProDy's protein C-alpha selection
    excludes some modified residues (for example CAS), and DiffDock constructs
    its receptor graph from that selection.  Reuse the same parser and ordering
    here so pocket arrays are exactly node aligned at inference time.
    """
    protein = pr.parsePDB(str(path))
    if protein is None or protein.ca is None:
        raise ValueError(f'{path}: no protein C-alpha nodes')
    result = get_coords(protein)[:, 1, :].astype(np.float32, copy=False)
    if result.ndim != 2 or result.shape[1:] != (3,) or not len(result) or not np.isfinite(result).all():
        raise ValueError(f'{path}: no finite DiffDock receptor-node coordinates')
    return result


def pocket_probabilities(protein_path, ligand_path, cutoff=6.):
    protein = receptor_node_coordinates(protein_path)
    ligand = pdb_coordinates(ligand_path, alpha_carbons=False)
    distances = np.linalg.norm(protein[:, None, :] - ligand[None, :, :], axis=-1)
    return (distances.min(1) <= cutoff).astype(np.float32), protein, ligand


def read_ligand(path):
    suffix = Path(path).suffix.lower()
    if suffix == '.sdf':
        molecule = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)[0]
    else:
        molecule = Chem.MolFromMolFile(str(path), removeHs=True, sanitize=True)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError(f'{path}: full ligand chemistry is unavailable')
    return molecule


def build_case(entry, root, cutoff):
    case = entry['case']
    if case != Path(case).name or case in ('.', '..'):
        raise ValueError('unsafe case identifier')
    ligand_path = (root / entry['ligand']).resolve()
    molecule = read_ligand(ligand_path)
    sides, sizes, evidence = [], [], []
    for side in (1, 2):
        protein = (root / entry[f'protein{side}']).resolve()
        fragment = protein.parent / f'unbound_lig{side}.pdb'
        if not protein.is_file() or not fragment.is_file():
            raise FileNotFoundError(f'{case}: binary template side {side} is incomplete')
        pocket, protein_coords, ligand_coords = pocket_probabilities(protein, fragment, cutoff)
        sizes.append(len(protein_coords))
        evidence_id = (
            f'deepternary-template:{protein.name}:{sha256_file(protein)}:'
            f'{fragment.name}:{sha256_file(fragment)}')
        evidence.append({
            'side': side, 'protein': str(protein), 'binary_ligand': str(fragment),
            'protein_sha256': sha256_file(protein), 'binary_ligand_sha256': sha256_file(fragment),
            'protein_ca_nodes': len(protein_coords), 'binary_ligand_heavy_atoms': len(ligand_coords),
            'pocket_ca_nodes': int(pocket.sum()),
        })
        sides.append({
            'provenance': 'template', 'evidence_id': evidence_id,
            'confidence': 1., 'pocket_probabilities': pocket.tolist(),
        })
    common = {
        'schema': MANIFEST_VERSION, 'case': case,
        'ligand_atom_count': molecule.GetNumAtoms(),
        'ligand_canonical_isomeric_smiles': canonical_isomeric_smiles(molecule),
        'linker_indices': [], 'protac': True,
    }
    pocket_payload = {**common, 'sides': sides}
    absent_payload = {**common, 'sides': [None, None]}
    # Exercise the production loader now, including exact node-array lengths.
    features, audit = load_condition_manifest_payload(
        pocket_payload, case=case, molecule=molecule, protein_sizes=sizes)
    if any(features[f'protein{i}'].shape[0] != sizes[i - 1] for i in (1, 2)):
        raise AssertionError('generated pocket manifest is not node aligned')
    return pocket_payload, absent_payload, {
        'case': case, 'ligand_sha256': sha256_file(ligand_path),
        'sides': evidence,
        'loader_audit': {key: value for key, value in audit.items() if key != 'path'},
    }


def load_condition_manifest_payload(payload, *, case, molecule, protein_sizes):
    """Validate an in-memory payload through the exact file-based loader."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='protac-pocket-manifest-') as directory:
        path = Path(directory) / 'condition.json'
        path.write_text(json.dumps(payload))
        return load_condition_manifest(
            path, case=case, molecule=molecule, protein_sizes=protein_sizes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cutoff', type=float, default=6.)
    args = parser.parse_args(argv)
    if args.out.exists() or not np.isfinite(args.cutoff) or not 0 < args.cutoff < 20:
        parser.error('--out must be new and --cutoff must be finite in (0,20)')
    index, root = args.index.resolve(), args.root.resolve()
    source = json.loads(index.read_text())
    entries = source['entries'] if isinstance(source, dict) else source
    if not entries:
        raise ValueError('empty PROTAC index')
    pocket_dir, absent_dir = args.out / 'pocket', args.out / 'absent'
    pocket_dir.mkdir(parents=True)
    absent_dir.mkdir()
    records = []
    for entry in entries:
        pocket, absent, record = build_case(entry, root, args.cutoff)
        case = entry['case']
        (pocket_dir / f'{case}.json').write_text(json.dumps(pocket, indent=2) + '\n')
        (absent_dir / f'{case}.json').write_text(json.dumps(absent, indent=2) + '\n')
        records.append(record)
    report = {
        'protocol': 'protac_binary_template_pocket_manifests_v1',
        'scope': ('pocket-only deploy conditions; local binary templates are treated as provided '
                  'templates, not verified experimental structures'),
        'index': str(index), 'index_sha256': sha256_file(index),
        'cutoff_angstrom': args.cutoff, 'cases': len(records),
        'all_cases_complete': len(records) == len(entries),
        'native_ternary_geometry_read': False,
        'warhead_atom_mapping_used': False, 'cross_protein_geometry_used': False,
        'records': records,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in (
        'protocol', 'cases', 'all_cases_complete', 'native_ternary_geometry_read')}, indent=2))


if __name__ == '__main__':
    main()
