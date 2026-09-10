#!/usr/bin/env python
"""Fixed-coupling experimental baseline audit; no training or blind docking."""

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/wl/workspace/CoopDiff')
OUT = ROOT / 'results/local_velocity_audit'
ARTIFACTS = ORIGINAL / 'results/handoff_artifacts'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '5')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['HF_HUB_CACHE'] = str(ORIGINAL / 'hf_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRICOOPDIFF_CACHE_DIR'] = str(ORIGINAL / 'results/runtime_cache/diffdock')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ORIGINAL)]

import numpy as np
import torch

import tricoopdiff.diffdock_compat  # noqa: F401,E402
from audit_velocity_directions import vector_metrics
from factorial_arm import PairedTorsionFlow, restrict_non_single
from featurize_common import _to_device_and_batch
from train_entry import _load_score_model_args
from tricoopdiff.consensus import ConsensusFuse
from tricoopdiff.consensus_experiment import ExperimentalConsensus, ExperimentalConsensusTrainer
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility

CASES = ('3CXH_C_E_6PH', '6U9D_Q_R_TPP', '4I0P_D_H_PGE')
SEEDS = (12001, 12002)
TIMES = (.1, .5, .9)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_cpu(item) for item in value)
    return copy.deepcopy(value)


def state_hash(state):
    digest = hashlib.sha256()
    for tensor in state[:3]:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_batch(entry, plan, score_args):
    """Verify transferred v7 cache against original path names and local bytes."""
    mapped = {key: str(ARTIFACTS / entry[key].removeprefix('/tmp/coopdiff/'))
              for key in ('protein1', 'protein2', 'ligand')}
    digest = hashlib.sha256()
    for key in ('protein1', 'protein2', 'ligand'):
        digest.update(entry[key].encode())
        digest.update(Path(mapped[key]).read_bytes())
    digest.update(json.dumps(vars(score_args), sort_keys=True, default=str).encode())
    cache_path = plan / 'cache_ccd' / (entry['case'] + '.pt')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    if cache['cache_version'] != 7 or cache['source_signature'] != digest.hexdigest():
        raise ValueError(f"cache provenance mismatch: {entry['case']}")
    batch = {'case': entry['case'], **{
        key: _to_device_and_batch(cache[key], 'cuda')
        for key in ('e3_data', 'poi_data', 'ligand_data')}}
    return restrict_non_single(batch, mapped['ligand']), {
        'case': entry['case'], 'cache': str(cache_path),
        'cache_sha256': sha256(cache_path), 'source_signature_verified': True,
        'logical_paths': {key: entry[key] for key in mapped}, 'mapped_paths': mapped}


def build(checkpoint, expert_dir):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['experiment_contract'] == {
        'chemistry': 'ccd', 'torsion': 'single_bond', 'adaptation': 'frozen', 'seed': 1729}
    expert = build_diffdock_expert(str(expert_dir), 'cuda')
    model = ExperimentalConsensus(expert, ConsensusFuse(**saved['model_config']),
                                  saved['expert_lora_config'])
    runtime = saved['runtime_config']
    trainer = ExperimentalConsensusTrainer(
        model, PairedTorsionFlow(), device='cuda',
        experiment_contract=saved['experiment_contract'], lr=1e-4,
        lora_lr=1e-5, lora_early_lr=5e-6, lora_warmup_steps=0,
        **{key: runtime[key] for key in (
            'activation_clamp', 'activation_clamp_mode', 'residual_weight', 'ema_decay')})
    trainer.load_checkpoint(checkpoint)
    model.eval()
    return trainer, {key: saved[key] for key in (
        'architecture', 'checkpoint_version', 'experiment_contract', 'runtime_config',
        'expert_base_identity', 'training_state', 'global_step')}


def velocity(trainer, batch, state, time, dt):
    output = trainer._forward_state(
        batch['e3_data'], batch['poi_data'], batch['ligand_data'],
        *state[:3], time, dt)
    return trainer.diffusion.velocities(
        trainer.diffusion.normalized_predictions(output, time),
        batch['e3_data']['coords'], batch['poi_data']['coords'])


def components(predicted, target):
    pairs = {f'{branch}.{kind}': (predicted[branch][kind], target[branch][kind])
             for branch in ('e3_ligand', 'poi_ligand') for kind in ('tr', 'rot')}
    pairs['torsion'] = (predicted['torsion'], target['torsion'])
    result = {}
    for name, (pred, truth) in pairs.items():
        assert torch.isfinite(pred).all() and torch.isfinite(truth).all()
        values = vector_metrics(pred, truth)
        values['norm_ratio'] = (values['prediction_norm'] / values['target_norm']
                                if values['target_norm'] > 0 else None)
        values['prediction'] = pred.detach().cpu().reshape(-1).tolist()
        values['target'] = truth.detach().cpu().reshape(-1).tolist()
        result[name] = values
    return result


def drift(actual, expected):
    """Unaligned per-atom RMSD in the fixed E3 frame, not per-coordinate RMS."""
    result = {}
    for name, a, b in zip(('ligand', 'poi'), actual[:2], expected[:2]):
        result[f'{name}_rmsd_A'] = float((a - b).square().sum(-1).mean().sqrt())
        result[f'{name}_center_A'] = float(torch.linalg.vector_norm(a.mean(1) - b.mean(1)))
    result['ligand_internal_distance_rmse_A'] = float((
        torch.pdist(actual[0][0]) - torch.pdist(expected[0][0])).square().mean().sqrt())
    assert all(np.isfinite(value) for value in result.values())
    return result


def frame_comparison(teacher, rollout):
    """Transport row vectors via v_local = v_teacher @ B_teacher.

    path() maps POI-local row vectors using B_teacher.T. Keeping those local
    components at a different body B_rollout therefore gives
    v_transport = v_teacher @ B_teacher @ B_rollout.T. This is a frame
    comparison, not a derivation of an off-path conditional flow target.
    """
    teacher_body, rollout_body = teacher[2][0], rollout[2][0]
    result = {}
    for kind in ('tr', 'rot'):
        original = teacher[3]['poi_ligand'][kind]
        local = original @ teacher_body
        transported = local @ rollout_body.T
        result[kind] = vector_metrics(original, transported)
        result[kind]['local_roundtrip_max_error'] = float(
            (transported @ rollout_body - local).abs().max())
        result[kind]['norm_preservation_error'] = float(
            (transported.norm() - original.norm()).abs())
    relative = rollout_body @ teacher_body.T
    result['body_angle_rad'] = float(torch.acos(((relative.trace() - 1) / 2).clamp(-1, 1)))
    return result


def summarize(rows):
    result = []
    for weights in ('ema', 'raw'):
        for time in TIMES:
            selected = [r for r in rows if r['weights'] == weights and r['time'] == time]
            for branch in selected[0]['components']:
                values = [r['components'][branch] for r in selected]
                result.append({'weights': weights, 'time': time, 'branch': branch,
                               'n': len(values), **{
                    'mean_' + key: float(np.mean([v[key] for v in values if v[key] is not None]))
                    if any(v[key] is not None for v in values) else None
                    for key in ('cosine', 'prediction_norm', 'target_norm', 'norm_ratio',
                                'signed_projection_ratio', 'error_norm')}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', action='store_true', help='Reload saved couplings for a repeat run')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    reproducibility = configure_inference_reproducibility(True)
    assert torch.cuda.device_count() == 1
    assert str(ROOT) in sys.modules['tricoopdiff'].__file__
    plan = ARTIFACTS / 'results/overnight_root_cause_v3_plan'
    checkpoint = ARTIFACTS / 'results/overnight_root_cause_v3_run04/ccd_single_bond_frozen_seed1729/best.pt'
    expert_dir = ARTIFACTS / 'checkpoints/score_model'
    entries = {e['case']: e for e in json.loads((plan / 'indexes/ternary_val_ccd.json').read_text())['entries']}
    score_args = _load_score_model_args(str(expert_dir))
    batches, provenance = [], []
    for case in CASES:
        batch, evidence = load_batch(entries[case], plan, score_args)
        batches.append(batch)
        provenance.append(evidence)
        print(f'Loaded verified cache: {case}', flush=True)
    trainer, metadata = build(checkpoint, expert_dir)
    print(f'Loaded experimental checkpoint on {torch.cuda.get_device_name(0)}', flush=True)
    flow = trainer.diffusion
    saved_pairs = (torch.load(out / 'fixed_pairs.pt', map_location='cpu', weights_only=False)
                   if args.repeat else {})
    pairs = []
    for batch in batches:
        for seed in SEEDS:
            key = f"{batch['case']}:{seed}"
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            coupling = (saved_pairs[key]['coupling'] if args.repeat else flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'],
                no_torsion=bool(trainer.model.expert.model.no_torsion)))
            initial = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling,
                                torch.tensor([1.], device='cuda'))
            if args.repeat:
                assert state_hash(initial) == saved_pairs[key]['initial_sha256']
            else:
                saved_pairs[key] = {'batch': tree_cpu(batch), 'coupling': tree_cpu(coupling),
                                    'initial': tree_cpu(initial[:3]), 'initial_sha256': state_hash(initial)}
            pairs.append((batch, seed, coupling, initial))
    if not args.repeat:
        torch.save(saved_pairs, out / 'fixed_pairs.pt')
    rows, rollout, local_checks, frame_rows = [], [], [], []
    with torch.no_grad():
        for weights in ('ema', 'raw'):
            context = trainer.ema_parameters() if weights == 'ema' else nullcontext()
            with context:
                for batch, seed, coupling, initial in pairs:
                    identity = {'weights': weights, 'case': batch['case'], 'seed': seed}
                    for value in TIMES:
                        time = torch.tensor([value], device='cuda')
                        dt = time.new_tensor([.05])
                        state = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
                        pred = velocity(trainer, batch, state, time, dt)
                        rows.append({**identity, 'time': value, 'conditioning_dt': .05,
                                     'state_sha256': state_hash(state),
                                     'components': components(pred, state[3])})
                        if weights == 'ema':
                            for h in (.001, .0005):
                                delta = time.new_tensor([h])
                                expected = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling, time - delta)
                                oracle = flow.step(*state[:3], state[3], delta, batch['ligand_data'])
                                step = flow.step(*state[:3], pred, delta, batch['ligand_data'])
                                local_checks.append({**identity, 'time': value, 'dt': h,
                                    'oracle_euler_vs_teacher': drift(oracle, expected),
                                    'predicted_euler_vs_teacher': drift(step, expected),
                                    'zero_velocity_vs_teacher': drift(state, expected)})
                    state = tuple(x.clone() for x in initial[:3])
                    for step_index in range(11):
                        value = 1. - step_index * .05
                        time = torch.tensor([value], device='cuda')
                        expected = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
                        frame_rows.append({**identity, 'step': step_index, 'time': value,
                                           'comparison': frame_comparison(expected, state)})
                        rollout.append({**identity, 'step': step_index, 'time': value,
                                        'state_sha256': state_hash(state),
                                        'vs_teacher': drift(state, expected),
                                        'vs_native': drift(state, (batch['ligand_data']['coords'], batch['poi_data']['coords']))})
                        if step_index < 10:
                            dt = time.new_tensor([.05])
                            pred = velocity(trainer, batch, state, time, dt)
                            state = flow.step(*state[:3], pred, dt, batch['ligand_data'])
                    print(f'{weights} {batch["case"]} seed={seed}: teacher + 10 steps complete', flush=True)
    payload = {
        'scope': '3 fixed CCD validation cases x 2 couplings; teacher velocities and short free rollout',
        'limitations': 'Crystal-reference teacher t=1 start, not blind ETKDG inference. Conditional target cosine is not marginal-flow correctness. Drift is not proof the flow objective is wrong. No DockQ measured.',
        'units': {'translation': 'angstrom / unit time', 'rotation_and_torsion': 'radian / unit time'},
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_metadata': metadata, 'cases': CASES, 'seeds': SEEDS,
        'environment': {key: os.environ[key] for key in (
            'CUDA_VISIBLE_DEVICES', 'TRICOOPDIFF_CACHE_DIR', 'HF_HUB_CACHE',
            'HF_HUB_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        'gpu_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
        'reproducibility': reproducibility, 'provenance': provenance,
        'summary': summarize(rows), 'teacher_rows': rows, 'rollout_rows': rollout,
        'local_step_checks': local_checks,
        'poi_frame_rows': frame_rows,
        'poi_frame_scope': 'On these free-rollout states, compare unchanged teacher-global target to transport preserving teacher-local components: v_new = v_teacher @ B_teacher @ B_rollout.T. This does not establish the intended off-path supervision contract or a correct off-path flow target.',
    }
    destination = out / ('repeat.json' if args.repeat else 'audit.json')
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    if args.repeat:
        previous = json.loads((out / 'audit.json').read_text())
        checks = {key: previous[key] == payload[key] for key in (
            'teacher_rows', 'rollout_rows', 'local_step_checks')}
        (out / 'repeat_check.json').write_text(json.dumps(checks, indent=2) + '\n')
        assert all(checks.values()), checks
    print(f'Wrote {destination}', flush=True)


if __name__ == '__main__':
    main()
