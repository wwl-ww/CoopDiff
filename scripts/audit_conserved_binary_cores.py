"""Audit graph-only analogue cores without changing deployed conditions."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from rdkit import Chem

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from tricoopdiff.conserved_binary_cores import conserved_binary_cores
from tricoopdiff.conserved_core_geometry import core_geometry_evidence
from tricoopdiff.tautomer_binary_cores import tautomer_conserved_binary_cores
from make_protac_pocket_manifests import receptor_node_coordinates


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def molecule(path):
    result = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    if result is None:
        raise ValueError(f'cannot read molecule: {path}')
    result.RemoveAllConformers()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chemistry-report', type=Path, required=True)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tautomer-normalized', action='store_true',
                        help='Separate audit of complete canonical-tautomer correspondence')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = json.loads(args.chemistry_report.read_text())
    rows = []
    for entry in report['cases']:
        full_path = args.case_root / entry['case'] / 'ligand.sdf'
        if digest(full_path) != entry['full_ligand_sha256']:
            raise ValueError(f'full ligand hash mismatch: {full_path}')
        full = molecule(full_path)
        for side, record in entry['sides'].items():
            source_path = Path(record['restored_sdf'])
            if digest(source_path) != record['restored_sha256']:
                raise ValueError(f'binary fragment hash mismatch: {source_path}')
            binary = Chem.SDMolSupplier(str(source_path), removeHs=True)[0]
            if binary is None:
                raise ValueError(f'cannot read binary molecule: {source_path}')
            discover = tautomer_conserved_binary_cores if args.tautomer_normalized else conserved_binary_cores
            cores, audit = discover(full, binary)
            protein_path = args.case_root / entry['case'] / f'unbound_protein{side}.pdb'
            protein = receptor_node_coordinates(protein_path)
            serializable = []
            for core in cores:
                core = dict(core)
                core['binary_local_evidence'] = core_geometry_evidence(core, binary, protein)
                # Atom maps correspond to this explicit source-index order,
                # not the atom ordering of a reparsed canonical SMILES.
                fragment = core.pop('fragment')
                core['fragment_molblock_generated_2d'] = Chem.MolToMolBlock(fragment)
                serializable.append(core)
            rows.append({'case': entry['case'], 'side': int(side),
                         'full_ligand_sha256': digest(full_path),
                         'binary_fragment_sha256': digest(source_path),
                         'binary_protein_sha256': digest(protein_path),
                         'audit': audit, 'cores': serializable})
            print(json.dumps({'case': entry['case'], 'side': side, **audit}), flush=True)
    output = {'audit_only': True, 'deployment_eligible': False,
              'tautomer_normalized': args.tautomer_normalized,
              'chemistry_report_sha256': digest(args.chemistry_report),
              'native_coordinates_used': False, 'sides': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(output, handle, indent=2)


if __name__ == '__main__':
    main()
