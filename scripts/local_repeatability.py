#!/usr/bin/env python
"""Replay disk-saved SDPI states through the fixed experimental EMA baseline."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# Set before importing numerical libraries; expose only the assigned GPU.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'
os.environ['HF_HUB_CACHE'] = '/home/wl/workspace/CoopDiff/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRICOOPDIFF_CACHE_DIR'] = '/home/wl/workspace/CoopDiff/results/runtime_cache/diffdock'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/wl/workspace/CoopDiff')  # missing extension stubs only

import numpy as np
import torch
import yaml
from rdkit import Chem

import tricoopdiff.diffdock_compat  # before vendored model imports
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import ExperimentalConsensus
from tricoopdiff.evaluate import compute_deepternary_dockq, compute_rmsd
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.sampler import sample_single
from tricoopdiff.sdpi_flow import SDPIFlow


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_hash(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    return hashlib.sha256(str((array.shape, array.dtype)).encode() + array.tobytes()).hexdigest()


def relocate(path, artifacts):
    old = Path(path)
    if old.is_relative_to('/tmp/coopdiff'):
        return artifacts / old.relative_to('/tmp/coopdiff')
    return old


def load_case(entry, artifacts, score_args):
    """Read cache without modifying it; verify original path-dependent signature."""
    cache_path = artifacts / 'results/overnight_root_cause_v3_plan/cache_ccd' / (entry['case'] + '.pt')
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    assert cached['cache_version'] == 7, 'requires corrected v7 receptor graphs'
    digest = hashlib.sha256()
    sources = {}
    for key in ('protein1', 'protein2', 'ligand'):
        old = entry[key]
        path = relocate(old, artifacts)
        digest.update(old.encode())
        digest.update(path.read_bytes())
        sources[key] = {'path': str(path), 'sha256': sha(path)}
    digest.update(json.dumps(score_args, sort_keys=True, default=str).encode())
    assert digest.hexdigest() == cached['source_signature'], 'cache source signature mismatch'
    batch = {}
    for name in ('e3_data', 'poi_data', 'ligand_data'):
        batch[name] = {}
        for key, value in cached[name].items():
            if torch.is_tensor(value):
                value = value.clone().to('cuda')
            else:
                value = copy.deepcopy(value)
            if key in ('coords', 'features'):
                value = value.unsqueeze(0)
            batch[name][key] = value
    # Same single-bond projection as the factorial baseline, kept local.
    ligand = batch['ligand_data']
    mol = Chem.SDMolSupplier(sources['ligand']['path'], removeHs=True)[0]
    assert mol is not None
    edges = ligand['edge_index'].T[ligand['edge_mask']].cpu()
    keep = np.array([mol.GetBondBetweenAtoms(int(u), int(v)).GetBondType() == Chem.BondType.SINGLE
                     for u, v in edges], dtype=bool)
    selected = ligand['edge_mask'].nonzero().flatten()
    ligand['edge_mask'][selected[torch.as_tensor(~keep, device='cuda')]] = False
    masks = ligand['mask_rotate']
    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    ligand['mask_rotate'] = np.asarray(masks)[keep].copy()
    return batch, {'cache': str(cache_path), 'cache_sha256': sha(cache_path),
                   'source_signature_verified': True, 'sources': sources}


def load_baseline(path, artifacts):
    """Strict EMA inference load; no optimizer/trainer or historical build helper."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['checkpoint_version'] == 1
    assert payload['architecture'] == 'residual_consensus_factorial_v1'
    assert payload['experiment_contract'] == dict(chemistry='ccd', torsion='single_bond', adaptation='frozen', seed=1729)
    runtime = payload['runtime_config']
    os.environ['NAN_CLAMP'] = str(runtime['activation_clamp'])
    os.environ['NAN_CLAMP_MODE'] = runtime['activation_clamp_mode']
    expert = build_diffdock_expert(str(artifacts / 'checkpoints/score_model'), 'cuda')
    identity = validate_expert_base_identity(payload, expert)
    fuse = ConsensusFuse(**payload['model_config'])
    model = ExperimentalConsensus(expert, fuse).to('cuda')
    model.trifuse.load_state_dict(payload['ema_trifuse_state_dict'], strict=True)
    model.eval()
    return model, {'sha256': sha(path), 'path': str(path), 'weights': 'EMA',
                   'runtime_config': runtime, 'expert_identity': identity,
                   'experiment_contract': payload['experiment_contract']}


class ReplayInitializer:
    """Reload complete initialization and post-initialization RNG state per call."""
    no_torsion = False

    def __init__(self, path):
        self.path = path
        self.hashes = None

    def initialize(self, *args, **kwargs):
        saved = torch.load(self.path, map_location='cpu', weights_only=False)
        state = tuple(t.clone().to('cuda') for t in saved['state'][:3])
        self.hashes = [coordinate_hash(t) for t in state]
        assert self.hashes == saved['hashes'], 'disk replay must be bitwise identical'
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'])
        torch.cuda.set_rng_state(saved['rng_cuda'], device=0)
        return (*state, copy.deepcopy(saved['state'][3]))


def install_index_add_audit(records):
    """Probe actual inference reduction inputs; restore the original method later."""
    original = torch.Tensor.index_add_
    seen = set()

    def audited(target, dim, index, source, *args, **kwargs):
        key = (tuple(target.shape), tuple(source.shape))
        if (len(seen) < 5 and key not in seen and target.is_cuda
                and source.is_floating_point() and source.ndim > 1):
            seen.add(key)
            mode = torch.are_deterministic_algorithms_enabled()
            row = {'op': 'torch.Tensor.index_add_', 'target_shape': list(target.shape),
                   'source_shape': list(source.shape), 'dtype': str(source.dtype),
                   'input_sha256': coordinate_hash(source), 'trials': 10}
            try:
                for deterministic in (False, True):
                    torch.use_deterministic_algorithms(deterministic)
                    outputs = [original(target.clone(), dim, index, source, *args, **kwargs)
                               for _ in range(10)]
                    row[str(deterministic)] = {
                        'bitwise_equal': all(torch.equal(outputs[0], x) for x in outputs[1:]),
                        'max_abs_difference': max(float((outputs[0] - x).abs().max()) for x in outputs[1:])}
            finally:
                torch.use_deterministic_algorithms(mode)
            records.append(row)
        return original(target, dim, index, source, *args, **kwargs)

    torch.Tensor.index_add_ = audited
    return original


def install_deterministic_index_add():
    """Isolate the reduction's effect without changing other inference operators."""
    original = torch.Tensor.index_add_

    def deterministic(target, *args, **kwargs):
        mode = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            return original(target, *args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(mode)

    torch.Tensor.index_add_ = deterministic
    return original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, default=Path('/home/wl/workspace/CoopDiff/results/handoff_artifacts'))
    ap.add_argument('--out', type=Path, default=ROOT / 'results/local_repeatability')
    ap.add_argument('--cases', nargs='+', default=['3CXH_C_E_6PH', '6U9D_Q_R_TPP', '4I0P_D_H_PGE'])
    ap.add_argument('--seed', type=int, default=12001)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--audit-index-add', action='store_true')
    ap.add_argument('--deterministic-index-add', action='store_true')
    ap.add_argument('--initializations-root', type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(args.deterministic)
    assert torch.cuda.device_count() == 1
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.artifacts / 'results/overnight_root_cause_v3_run04/ccd_single_bond_frozen_seed1729/best.pt'
    op_records = []
    if args.audit_index_add and args.deterministic_index_add:
        ap.error('operator audit and selective determinism require separate runs')
    original_index_add = install_index_add_audit(op_records) if args.audit_index_add else None
    if args.deterministic_index_add:
        original_index_add = install_deterministic_index_add()
    model, provenance = load_baseline(checkpoint, args.artifacts)
    score_args = yaml.full_load((args.artifacts / 'checkpoints/score_model/model_parameters.yml').read_text())
    entries = json.loads((args.artifacts / 'results/overnight_root_cause_v3_plan/indexes/ternary_val_ccd.json').read_text())['entries']
    entries = {e['case']: e for e in entries}
    report = {'scope': 'same baseline EMA, one saved initialization per case, two repeated trajectories; approximate DockQ only',
              'checkpoint': provenance, 'seed': args.seed, 'steps': args.steps,
              'environment': {k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRICOOPDIFF_CACHE_DIR')},
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'index_add_audit': op_records, 'deterministic_index_add_only': args.deterministic_index_add, 'cases': []}
    for case in args.cases:
        started = time.monotonic()
        batch, cache_info = load_case(entries[case], args.artifacts, score_args)
        case_dir = args.out / case
        case_dir.mkdir(exist_ok=True)
        init_path = case_dir / 'initialization.pt'
        if args.initializations_root:
            init_path = args.initializations_root / case / 'initialization.pt'
            saved = torch.load(init_path, map_location='cpu', weights_only=False)
            hashes = saved['hashes']
        else:
            # Fail rather than overwrite an existing saved experiment.
            if init_path.exists():
                raise FileExistsError(f'{init_path}: choose a fresh --out to rerun')
            state = TernaryInitializer().initialize(batch['e3_data']['coords'][0], batch['poi_data']['coords'][0], batch['ligand_data'], seed=args.seed)
            hashes = [coordinate_hash(t) for t in state[:3]]
            torch.save({'state': (*[t.detach().cpu().clone() for t in state[:3]], copy.deepcopy(state[3])),
                        'hashes': hashes, 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                        'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state(0)}, init_path)
        row = {'case': case, **cache_info, 'initialization_sha256': sha(init_path), 'initialization_path': str(init_path),
               'initialization_coordinate_hashes': hashes, 'repeats': []}
        results = []
        for repeat in range(2):
            replay = ReplayInitializer(init_path)
            with torch.no_grad():
                result = sample_single(model, SDPIFlow(), replay, **copy.deepcopy(batch),
                                       steps=args.steps, seed=args.seed, device='cuda',
                                       return_trajectory=True, return_trace=True,
                                       translation_velocity_scale=1.0, rotation_velocity_scale=1.0)
            torch.save(result, case_dir / f'repeat_{repeat + 1}.pt')
            lig, poi = result['ligand_coords'], result['poi_coords']
            native_lig, native_poi, e3 = [batch[k]['coords'][0].cpu().numpy() for k in ('ligand_data', 'poi_data', 'e3_data')]
            dock = compute_deepternary_dockq(native_lig, e3, native_poi, lig, e3, poi)
            row['repeats'].append({'repeat': repeat + 1, 'initialization_coordinate_hashes': replay.hashes,
                'valid': result['valid'], 'completed_steps': len(result['trajectory']),
                'endpoint_hashes': [coordinate_hash(lig), coordinate_hash(poi)],
                'ligand_center_error_A': float(np.linalg.norm(lig.mean(0) - native_lig.mean(0))),
                'poi_center_error_A': float(np.linalg.norm(poi.mean(0) - native_poi.mean(0))),
                'protac_aligned_rmsd_A': compute_rmsd(lig, native_lig, align_first=True),
                'approx_dockq': dock['dockq'] if dock else None})
            results.append(result)
            print(json.dumps({'case': case, **row['repeats'][-1]}), flush=True)
        comparisons = []
        for step, (left, right) in enumerate(zip(results[0]['trajectory'], results[1]['trajectory']), 1):
            comparisons.append({'step': step, 'bitwise_equal': all(np.array_equal(a, b) for a, b in zip(left, right)),
                'ligand_max_abs_A': float(np.max(np.abs(left[0] - right[0]))),
                'poi_max_abs_A': float(np.max(np.abs(left[1] - right[1])))})
        row['per_step_comparison'] = comparisons
        row['passed'] = (all(r['valid'] and r['completed_steps'] == args.steps for r in row['repeats'])
                         and len(comparisons) == args.steps and all(c['bitwise_equal'] for c in comparisons))
        row['elapsed_seconds'] = time.monotonic() - started
        report['cases'].append(row)
        report['passed'] = all(r['passed'] for r in report['cases'])
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'{case}: repeatability passed={row["passed"]}, seconds={row["elapsed_seconds"]:.1f}', flush=True)
    if original_index_add is not None:
        torch.Tensor.index_add_ = original_index_add
    assert report['passed'], 'repeatability failed; inspect per-step comparisons'


if __name__ == '__main__':
    main()
