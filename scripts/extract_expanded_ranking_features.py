"""Extract native-free heavy-atom features from sealed ranking archives."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import prody as pr
import torch

from evaluation_common import evaluation_input_signature, sha256_file
from tricoopdiff.diffdock.datasets.parse_chi import get_coords
from tricoopdiff.ranking.ranking_archive import (EXPANDED_GEOMETRY_NAMES, apply_fitted_rigid,
                                           expanded_geometry_features)
from tricoopdiff.ranking.ranking_archive import GEOMETRY_NAMES
from tricoopdiff.ranking.ranking_inference import archive_rows
from tricoopdiff.runtime import atomic_json


PROTOCOL = 'native_free_heavy_atom_ranking_features_v1'




def protein_coordinates(path):
    structure = pr.parsePDB(str(path))
    selection = structure.select('protein and not hydrogen')
    if selection is None:
        raise ValueError(f'{path}: no protein heavy atoms')
    return np.asarray(get_coords(structure)[:, 1, :]), np.asarray(selection.getCoords())


def extract_case(archive, entry):
    raw_e3_ca, raw_e3_heavy = protein_coordinates(entry['protein1'])
    raw_poi_ca, raw_poi_heavy = protein_coordinates(entry['protein2'])
    fixed_e3_ca = archive['input_graphs']['e3_data']['coords'][0].numpy()
    e3_heavy, e3_fit = apply_fitted_rigid(raw_e3_ca, fixed_e3_ca, raw_e3_heavy)
    rows = []
    source_rows = {(row['pool_index'], row['stage']): row for row in archive_rows(archive)}
    for pair in archive['pairs']:
        for stage in ('initial', 'final'):
            if stage == 'final' and not pair['valid']:
                continue
            state = (archive['pool'][pair['pool_index']]['state'] if stage == 'initial'
                     else pair['final'])
            poi_heavy, poi_fit = apply_fitted_rigid(
                raw_poi_ca, torch.as_tensor(state['poi']).numpy(), raw_poi_heavy)
            heavy = expanded_geometry_features(
                e3_heavy, poi_heavy, torch.as_tensor(state['ligand']).numpy())
            archived = source_rows[pair['pool_index'], stage]['features']['geometry']
            vector = np.concatenate((np.asarray(archived, dtype=np.float32), heavy))
            rows.append({
                'pool_index': pair['pool_index'], 'stage': stage, 'source': pair['source'],
                'features': vector, 'e3_ca_fit_rmsd': e3_fit,
                'poi_ca_fit_rmsd': poi_fit,
            })
    if len(rows) != len(source_rows):
        raise ValueError('expanded feature rows do not cover archived states')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error('invalid shard index')
    archive_manifest_path = args.archives / 'manifest.json'
    archive_manifest = json.loads(archive_manifest_path.read_text())
    index = json.loads(args.index.read_text())
    if set(index) != {'entries'}:
        raise ValueError('feature extraction index must not contain labels')
    entries = {entry['case']: entry for entry in index['entries']}
    if (len(entries) != len(index['entries'])
            or not set(archive_manifest['cases']).issubset(entries)):
        raise ValueError('feature index must uniquely cover every archive case')
    manifest = {
        'protocol': PROTOCOL,
        'archive_manifest_sha256': sha256_file(archive_manifest_path),
        'inference_index_sha256': sha256_file(args.index),
        'feature_names': list(GEOMETRY_NAMES) + list(EXPANDED_GEOMETRY_NAMES),
        'native_or_label_inputs': False,
        'cases': archive_manifest['cases'],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('expanded feature manifest mismatch')
    atomic_json(manifest, manifest_path)
    completed, failures = [], []
    progress_path = args.out / f'worker_{args.shard_index}.json'
    started = time.monotonic()
    for case in archive_manifest['cases'][args.shard_index::args.num_shards]:
        try:
            entry = entries[case]
            paths = {key: Path(entry[key]).resolve() for key in ('protein1', 'protein2', 'ligand')}
            if evaluation_input_signature(paths) != archive_manifest['input_signatures'][case]:
                raise ValueError('input structure signature differs from sealed archive')
            archive_path = args.archives / f'{case}.pt'
            archive = torch.load(archive_path, map_location='cpu', weights_only=False)
            rows = extract_case(archive, entry)
            payload = {'protocol': PROTOCOL, 'case': case,
                       'archive_sha256': sha256_file(archive_path), 'rows': rows}
            destination = args.out / f'{case}.pt'
            temporary = destination.with_suffix(f'.pt.tmp.{os.getpid()}')
            torch.save(payload, temporary); temporary.replace(destination)
            completed.append(case)
        except Exception as error:
            failures.append({'case': case, 'error': f'{type(error).__name__}: {error}'})
        atomic_json({'state': 'running', 'completed': completed, 'failures': failures,
                     'elapsed_seconds': time.monotonic() - started}, progress_path)
    atomic_json({'state': 'failed' if failures else 'complete', 'completed': completed,
                 'failures': failures, 'elapsed_seconds': time.monotonic() - started}, progress_path)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
