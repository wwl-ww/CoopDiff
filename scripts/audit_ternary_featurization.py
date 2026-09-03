#!/usr/bin/env python
"""Audit a ternary JSON index through the production featurization path."""

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml

from featurize_common import featurize_complex


def _resolve(path):
    value = Path(path)
    return value if value.is_absolute() else _PROJECT_ROOT / value


def _tensor_finite(value):
    return not torch.is_tensor(value) or bool(torch.isfinite(value).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--model-dir', default='checkpoints/score_model')
    parser.add_argument('--cache-dir', default='results/featurize_cache_ternary')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(_resolve(args.index).read_text())
    entries = payload.get('entries', payload)
    if args.limit is not None:
        entries = entries[:args.limit]
    with (_resolve(args.model_dir) / 'model_parameters.yml').open() as handle:
        model_args = Namespace(**yaml.full_load(handle))

    records = []
    for index, entry in enumerate(entries, start=1):
        started = time.time()
        record = {'case': entry['case'], 'ok': False}
        try:
            e3, poi, ligand, feature_dim = featurize_complex(
                entry['case'], str(_resolve(entry['protein1'])),
                str(_resolve(entry['protein2'])), str(_resolve(entry['ligand'])),
                model_args, args.device, cache_dir=str(_resolve(args.cache_dir)),
            )
            tensors = {
                'e3_coords': e3['coords'], 'e3_features': e3['features'],
                'poi_coords': poi['coords'], 'poi_features': poi['features'],
                'ligand_coords': ligand['coords'],
                'ligand_features': ligand['features'],
            }
            nonfinite = [name for name, value in tensors.items() if not _tensor_finite(value)]
            if nonfinite:
                raise FloatingPointError(f'non-finite tensors: {nonfinite}')
            record.update({
                'ok': True,
                'feature_dim': int(feature_dim),
                'e3_nodes': int(e3['coords'].shape[1]),
                'poi_nodes': int(poi['coords'].shape[1]),
                'ligand_nodes': int(ligand['coords'].shape[1]),
                'rotatable_bonds': int(ligand['edge_mask'].sum())
                if ligand.get('edge_mask') is not None else 0,
            })
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        record['seconds'] = round(time.time() - started, 3)
        records.append(record)
        print(f"[{index}/{len(entries)}] {record}", flush=True)

    report = {
        'index': args.index,
        'requested': len(entries),
        'passed': sum(record['ok'] for record in records),
        'failed': sum(not record['ok'] for record in records),
        'records': records,
    }
    destination = _resolve(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('requested', 'passed', 'failed')}))
    raise SystemExit(0 if report['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
