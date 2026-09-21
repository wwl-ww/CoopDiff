"""Restore selected training chemistry and rebuild isolated ligand caches."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from featurize_common import _FEATURIZE_CACHE_VERSION, _featurize_signature
from restore_ligand_chemistry import restore_ligand
from train_entry import _load_index, _load_score_model_args
from tricoopdiff.ligand_chemistry import rebuild_ligand_graph


def fetch_component(component, directory):
    path = directory / f'{component}.cif'
    if path.exists():
        return component, None
    try:
        url = f'https://www.ebi.ac.uk/pdbe/static/files/pdbechem_v2/{component}.cif'
        with urlopen(url, timeout=20) as response:
            content = response.read()
        if not content.lstrip().startswith(b'data_'):
            raise ValueError('CCD endpoint returned non-CIF content')
        with path.open('xb') as handle:
            handle.write(content)
        return component, None
    except Exception as error:
        return component, f'{type(error).__name__}: {error}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'cache-dir', 'expert-model-dir', 'ccd-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fetch-missing-ccd', action='store_true')
    parser.add_argument('--all-entries', action='store_true',
                        help='Restore all indexed cases with existing caches, not only cluster representatives')
    parser.add_argument('--disjoint-from', type=Path,
                        help='Required for validation: original full training index, before chemistry filtering')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('chemistry migration must use a new output directory')
    torch.set_num_threads(1)
    entries, clusters, by_case = _load_index(args.index)
    source_index = json.loads(args.index.read_text())
    split = source_index.get('metadata', {}).get('split')
    if split not in ('train', 'val'):
        raise ValueError('explicit train/val split metadata is required')
    disjoint_metadata = {}
    if split == 'val':
        if args.disjoint_from is None:
            raise ValueError('validation restoration requires the original full training index')
        training = json.loads(args.disjoint_from.read_text())
        if training.get('metadata', {}).get('split') != 'train':
            raise ValueError('disjointness reference must be a training index')
        training_ids = sorted({row['case'].split('_')[0].upper() for row in training['entries']})
        if set(training_ids) & {row['case'].split('_')[0].upper() for row in entries}:
            raise ValueError('validation overlaps the original training population')
        disjoint_metadata = {
            'disjoint_from_index_sha256': hashlib.sha256(args.disjoint_from.read_bytes()).hexdigest(),
            'excluded_training_pdb_ids': training_ids}
    excluded = source_index.get('metadata', {}).get('excluded_benchmark_pdb_ids')
    if not excluded or any(row['case'].split('_')[0].upper() in {x.upper() for x in excluded}
                           for row in entries):
        raise ValueError('training index must declare and obey benchmark exclusions')
    if clusters and not args.all_entries:
        entries = [by_case[next(case for case in (cluster.get('representative') or cluster['items'])
                                if case in by_case)] for cluster in clusters]
    if any(row['case'] != Path(row['case']).name or row['case'] in ('.', '..') for row in entries):
        raise ValueError('unsafe case identifier')
    args.out.mkdir(parents=True, exist_ok=False)
    cache_out = args.out / 'cache'
    cache_out.mkdir()
    components, failed = {}, {}
    for row in entries:
        names = {line[17:20].strip() for line in Path(row['ligand']).read_text().splitlines()
                 if line.startswith(('ATOM  ', 'HETATM'))}
        if len(names) != 1 or not next(iter(names)).isalnum():
            failed[row['case']] = 'not a single named CCD component'
        else:
            components[row['case']] = next(iter(names))
    downloads = {}
    if args.fetch_missing_ccd:
        args.ccd_dir.mkdir(parents=True, exist_ok=True)
        unique = sorted(set(components.values()))
        with ThreadPoolExecutor(max_workers=4) as executor:
            for index, (component, error) in enumerate(executor.map(
                    lambda value: fetch_component(value, args.ccd_dir), unique), start=1):
                downloads[component] = error
                if index % 25 == 0:
                    print(json.dumps({'ccd_processed': index, 'total': len(unique)}), flush=True)
    score_args = _load_score_model_args(str(args.expert_model_dir))
    reports, restored_entries = [], []
    for index, row in enumerate(entries):
        case = row['case']
        report = {'case': case}
        try:
            if case in failed:
                raise ValueError(failed[case])
            ccd = args.ccd_dir / f'{components[case]}.cif'
            if not ccd.is_file():
                raise FileNotFoundError(f'CCD missing: {ccd}; download={downloads.get(components[case])}')
            cache = torch.load(args.cache_dir / f'{case}.pt', map_location='cpu', weights_only=False)
            signature = _featurize_signature(row['protein1'], row['protein2'], row['ligand'], score_args)
            if cache.get('cache_version') != _FEATURIZE_CACHE_VERSION or cache.get('source_signature') != signature:
                raise ValueError('original feature cache is stale')
            destination = args.out / case / 'ligand.sdf'
            chemistry = restore_ligand(row['ligand'], ccd, destination)
            molecule = Chem.MolFromMolFile(str(destination), removeHs=True)
            ligand, graph_audit = rebuild_ligand_graph(molecule, cache['ligand_data'])
            updated = {**row, 'ligand': str(destination.resolve())}
            cache = {**cache, 'ligand_data': ligand,
                     'source_signature': _featurize_signature(
                         row['protein1'], row['protein2'], updated['ligand'], score_args),
                     'chemistry_migration': {'source_signature': signature,
                                              'ccd_sha256': chemistry['ccd_sha256'], **graph_audit}}
            torch.save(cache, cache_out / f'{case}.pt')
            restored_entries.append(updated)
            report.update(status='restored', chemistry=chemistry, graph=graph_audit)
        except (ValueError, RuntimeError, OSError) as error:
            report.update(status='rejected', error=f'{type(error).__name__}: {error}')
        reports.append(report)
        if (index + 1) % 25 == 0:
            print(json.dumps({'cases_processed': index + 1,
                              'counts': dict(Counter(r['status'] for r in reports))}), flush=True)
    restored_index = {'entries': restored_entries,
                      'metadata': {**source_index['metadata'], **disjoint_metadata,
                                   'scope': (f'restored {split} all indexed entries with valid existing caches'
                                             if args.all_entries else f'restored {split} cluster representatives only'),
                                   'source_index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest()}}
    if args.all_entries and clusters:
        accepted = {row['case'] for row in restored_entries}
        restored_clusters = []
        for cluster in clusters:
            items = [case for case in cluster['items'] if case in accepted]
            if items:
                representatives = [case for case in cluster.get('representative', []) if case in accepted]
                restored_clusters.append({**cluster, 'items': items, 'representative': representatives})
        restored_index['clusters'] = restored_clusters
    with (args.out / 'index.json').open('x') as handle:
        json.dump(restored_index, handle, indent=2)
    report = {'counts': dict(Counter(row['status'] for row in reports)), 'cases': reports,
              'scope': 'chemical definitions only; deposited coordinates retained; benchmark excluded',
              'ccd_download_errors': {key: value for key, value in downloads.items() if value}}
    with (args.out / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
