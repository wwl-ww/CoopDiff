"""Build graph-only conformer libraries; no training/benchmark scoring."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tricoopdiff.chemical_torsion_prior import CHEMICAL_PRIOR_POLICY, independent_conformers, ordered_graph_sha256
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-manifest', type=Path,
                        help='Reuse hash-verified graph-identical libraries with the same prior contract')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('conformer libraries require a new output directory')
    if args.seed < 0 or args.count < 1 or args.workers < 1:
        parser.error('nonnegative seed and positive conformer count required')
    payload = json.loads(args.index.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    graph_cache, graph_lock = {}, threading.Lock()
    if args.reuse_manifest:
        reuse = json.loads(args.reuse_manifest.read_text())
        if reuse['root_seed'] != args.seed or reuse['count'] != args.count:
            raise ValueError('reused conformers must match root seed and requested pool size')
        for row in reuse['cases']:
            if row['status'] != 'prepared':
                continue
            path = Path(row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('reused chemical prior artifact changed')
            saved = torch.load(path, map_location='cpu', weights_only=False)
            audit, identity = saved['audit'], saved['ordered_graph_sha256']
            expected_seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{identity}'.encode()).digest()[:4], 'big') % (2**31)
            if (audit['policy'] != CHEMICAL_PRIOR_POLICY or audit['requested'] != args.count
                    or audit['seed'] != expected_seed or audit['input_coordinates_used'] is not False
                    or audit.get('mmff_max_iterations') != 500
                    or audit['rdkit_version'] != Chem.rdBase.rdkitVersion):
                raise ValueError('reused chemical prior implementation/provenance mismatch')
            future = Future()
            future.set_result((saved['coordinates'], audit))
            graph_cache[identity] = future

    def prepare(entry):
        case = entry['case']
        cached = torch.load(args.cache_dir / f'{case}.pt', map_location='cpu', weights_only=False)
        mol = cached['ligand_data']['mol']
        graph_sha = ordered_graph_sha256(mol)
        seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{graph_sha}'.encode()).digest()[:4], 'big') % (2**31)
        try:
            with graph_lock:
                owner = graph_sha not in graph_cache
                if owner:
                    graph_cache[graph_sha] = Future()
                future = graph_cache[graph_sha]
            if owner:
                try:
                    future.set_result(independent_conformers(mol, seed=seed, count=args.count))
                except Exception as error:
                    future.set_exception(error)
                    raise
            coordinates, audit = future.result()
            if (coordinates.shape != (audit['accepted'], Chem.Mol(mol).GetNumAtoms(), 3)
                    or not torch.isfinite(coordinates).all()):
                raise RuntimeError('reused prior graph shape or finite coordinate contract failed')
        except ValueError as error:
            return {'case': case, 'status': 'rejected', 'reason': str(error)}
        path = args.out / f'{case}.pt'
        with path.open('xb') as handle:
            torch.save({'coordinates': coordinates, 'audit': audit,
                        'ordered_graph_sha256': graph_sha}, handle)
        return {'case': case, 'status': 'prepared', 'audit': audit,
                'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.out / 'progress.jsonl').open('x', buffering=1) as log:
        for row in pool.map(prepare, payload['entries']):
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps({'completed': len(rows), 'total': len(payload['entries']), **row}), flush=True)
    with (args.out / 'manifest.json').open('x') as handle:
        json.dump({'root_seed': args.seed, 'count': args.count,
                   'reuse_manifest_sha256': (hashlib.sha256(args.reuse_manifest.read_bytes()).hexdigest()
                                            if args.reuse_manifest else None),
                   'index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
                   'prepared': sum(row['status'] == 'prepared' for row in rows), 'cases': rows}, handle, indent=2)


if __name__ == '__main__':
    main()
