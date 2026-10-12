#!/usr/bin/env python
"""Build a separate CCD-corrected index offline; never modify source inputs.

By default a failed component aborts publication of the index. Explicit
quarantine mode publishes a separately labelled complete-component cohort. No atom is dropped and no
coordinate or chemical definition is inferred from native partner geometry.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.inputs.ligand_chemistry import CHEMISTRY_CONTRACT, restore_ligand


def prepare_index(index_path, ccd_dirs, out_dir, quarantine_incomplete=False):
    index_path, out_dir = Path(index_path).resolve(), Path(out_dir).resolve()
    payload = json.loads(index_path.read_text())
    if quarantine_incomplete and not isinstance(payload, dict):
        payload = {'entries': payload}
    entries = payload['entries'] if isinstance(payload, dict) else payload
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'index.json').exists():
        raise FileExistsError('use a new output directory to preserve published chemistry provenance')
    rows, corrected, failures = [], [], []
    for entry in entries:
        row = dict(entry)
        source = Path(row['ligand'])
        if source.suffix.lower() != '.pdb':
            # Bond-order-bearing files still get new legal torsion masks at
            # featurization. Their chemistry has not been validated against CCD.
            corrected.append(row)
            rows.append({'case': row['case'], 'source': str(source),
                         'chemistry_source': 'existing_bond_order_file'})
            continue
        try:
            from rdkit import Chem
            deposited = Chem.MolFromPDBFile(str(source), sanitize=False,
                                           removeHs=False, proximityBonding=False)
            components = {a.GetPDBResidueInfo().GetResidueName().strip()
                          for a in deposited.GetAtoms() if a.GetAtomicNum() > 1}
            if len(components) != 1:
                raise ValueError('expected exactly one CCD component')
            component = next(iter(components))
            ccd = next((Path(d) / (component + '.cif') for d in ccd_dirs
                        if (Path(d) / (component + '.cif')).is_file()), None)
            if ccd is None:
                raise FileNotFoundError(f'offline CCD definition missing: {component}.cif')
            case = row['case']
            if Path(case).name != case or case in ('.', '..'):
                raise ValueError('case must be a simple filename')
            destination = out_dir / 'ligands' / case / 'ligand.sdf'
            report = restore_ligand(source, ccd, destination)
            report.update(case=case, chemistry_contract=CHEMISTRY_CONTRACT,
                          sdf_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
            row['ligand'] = str(destination)
            row['chemistry_contract'] = CHEMISTRY_CONTRACT
            corrected.append(row)
            rows.append(report)
        except Exception as exc:
            failures.append({'case': row['case'], 'error': str(exc)})
    quarantined = ([r for r in failures if r['error'].startswith('incomplete or modified component:')]
                   if quarantine_incomplete else [])
    blocking = [r for r in failures if r not in quarantined]
    manifest = {'source_index': str(index_path),
                'source_index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
                'chemistry_contract': CHEMISTRY_CONTRACT, 'rows': rows,
                'failures': failures, 'quarantined': quarantined,
                'incomplete_policy': 'quarantine' if quarantine_incomplete else 'reject',
                'published': not blocking and bool(corrected)}
    (out_dir / 'chemistry_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if blocking or not corrected:
        raise ValueError(f'{len(failures)} chemistry failures; index not published; '
                         f'see {out_dir / "chemistry_manifest.json"}')
    if isinstance(payload, dict):
        payload = dict(payload, entries=corrected, chemistry_contract=CHEMISTRY_CONTRACT)
        if quarantined:
            allowed = {r['case'] for r in corrected}
            original_clusters = payload.get('clusters', [])
            clusters = []
            for cluster in original_clusters:
                items = [c for c in cluster['items'] if c in allowed]
                if items:
                    representatives = [c for c in cluster.get('representative', []) if c in allowed]
                    clusters.append(dict(cluster, items=items, representative=representatives))
            if original_clusters:
                payload['clusters'] = clusters
            payload['rejected'] = list(payload.get('rejected', [])) + [
                dict(case=r['case'], code='incomplete_ccd_component', reason=r['error'])
                for r in quarantined]
            payload['source_metadata'] = payload.get('metadata', {})
            payload['metadata'] = dict(payload.get('metadata', {}),
                chemistry_protocol='complete_ccd_cohort_v1',
                original_entries=len(entries), included_entries=len(corrected),
                quarantined_entries=len(quarantined), original_clusters=len(original_clusters),
                included_clusters=len(clusters), split_membership_preserved=True)
    else:
        payload = corrected
    destination = out_dir / 'index.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(destination)
    return destination



def fetch_missing_ccd(index_paths, ccd_dir):
    """Download official component definitions once, with bounded concurrency."""
    ccd_dir = Path(ccd_dir).resolve()
    ccd_dir.mkdir(parents=True, exist_ok=True)
    components = set()
    for index_path in index_paths:
        payload = json.loads(Path(index_path).read_text())
        entries = payload['entries'] if isinstance(payload, dict) else payload
        for entry in entries:
            path = Path(entry['ligand'])
            if path.suffix.lower() == '.pdb':
                components.update(line[17:20].strip() for line in path.read_text().splitlines()
                                  if line.startswith(('ATOM  ', 'HETATM')) and
                                  line[76:78].strip().upper() not in ('H', 'D', 'T'))
    def fetch(component):
        if not re.fullmatch(r'[A-Za-z0-9]{1,8}', component):
            raise ValueError(f'invalid CCD component identifier: {component!r}')
        destination = ccd_dir / (component + '.cif')
        if destination.exists():
            return
        url = f'https://files.rcsb.org/ligands/download/{component}.cif'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        import io
        definition = MMCIF2Dict(io.StringIO(content.decode()))
        if definition.get('_chem_comp.id') != [component]:
            raise ValueError(f'CCD response identity mismatch: {component}')
        temporary = destination.with_suffix('.cif.tmp')
        temporary.write_bytes(content)
        temporary.replace(destination)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, sorted(components)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--ccd-dir', action='append', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--quarantine-incomplete', action='store_true',
                        help='publish a separately labelled complete-CCD cohort with explicit exclusions')
    parser.add_argument('--download-missing', action='store_true',
                        help='fetch missing official CCD files into the first --ccd-dir')
    args = parser.parse_args()
    if args.download_missing:
        fetch_missing_ccd([args.index], args.ccd_dir[0])
    print(prepare_index(args.index, args.ccd_dir, args.out_dir,
                        quarantine_incomplete=args.quarantine_incomplete))


if __name__ == '__main__':
    main()
