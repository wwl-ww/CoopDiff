"""Oracle diagnostic: optimize motion controls, never a prediction benchmark.

Uses production SDPI initialization and the real symmetric rigid/torsion updater.
The unit-bound run is an expert-scale reference, NOT a bound on v39 repair.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tricoopdiff.diffdock_compat  # noqa: F401
import torch
from featurize_common import featurize_complex
from train_entry import _load_index, _load_score_model_args, validation_metric_context
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.models.operator_flow import advance
from tricoopdiff.models.constraint_motion import (
    horizontal_torsion_jacobian, point_jacobian, solve_velocity, terminal_distance_rate,
)
from tricoopdiff.sdpi_flow import SDPIFlow


def bound(vector):
    rigid = vector[..., :12].reshape(*vector.shape[:-1], 4, 3)
    norm = torch.linalg.vector_norm(rigid, dim=-1, keepdim=True)
    rigid = rigid / (1 + norm.square()).sqrt()
    return torch.cat((rigid.flatten(-2), vector[..., 12:].tanh()), -1)


def oracle_repair(state, target, e3, scales, ld, time_value, ridge=.01, saturate=True):
    """Native material distances, deterministic residue anchors, real WLS rules.

    This isolates the geometry solver from learned distance/anchor errors; it is
    not the exact learned anchor assignment or full expert mixture.
    """
    ligand, poi, _ = state
    coords = (e3[0], poi[0], ligand[0])
    desired = (e3[0], target[1][0], target[0][0])
    centers = torch.stack([x.mean(0) for x in coords])
    width = 12 + int(ld['edge_mask'].sum())
    torsion = horizontal_torsion_jacobian(ligand[0], ld['edge_index'], ld['edge_mask'], ld['mask_rotate'])
    matrices, rates, weights, saturation = [], [], [], []
    for receiver, anchor in ((2, 0), (2, 1), (1, 0)):
        ids = torch.linspace(0, len(coords[anchor]) - 1, min(16, len(coords[anchor]))).long()
        left, right = coords[receiver], coords[anchor][ids]
        ja = point_jacobian(left, receiver, centers, scales, width)
        if receiver == 2:
            ja[:, :, 12:] = torsion
        jb = point_jacobian(right, anchor, centers, scales, width)
        delta = right[None] - left[:, None]
        distance = (delta.square().sum(-1) + .01).sqrt() / 10
        target_distance = ((desired[anchor][ids][None] - desired[receiver][:, None]).square().sum(-1) + .01).sqrt() / 10
        direction = delta / (delta.square().sum(-1, keepdim=True) + .01).sqrt()
        matrix = torch.einsum('ijc,ijcd->ijd', direction, jb[None] - ja[:, None]) / 10
        raw_rate = (distance - target_distance) / time_value.clamp_min(.2)
        rate = terminal_distance_rate(distance, target_distance, time_value) if saturate else raw_rate
        matrices.append(matrix.reshape(-1, width))
        rates.append(rate.flatten())
        weights.append(torch.ones_like(rate).flatten() / rate.numel() / 3)
        saturation.append(float((raw_rate.abs() > 2).double().mean()))
    velocity, _ = solve_velocity(torch.cat(matrices), torch.cat(rates), torch.cat(weights), ridge=ridge)
    matrix, weight = torch.cat(matrices), torch.cat(weights)
    spectrum = torch.linalg.eigvalsh(matrix.T @ (weight[:, None] * matrix))
    residual = torch.cat(rates)
    diagnostics = {'weighted_rate_residual': float((weight * (matrix @ velocity - residual).square()).sum().sqrt()),
                   'rate_saturation_fraction': sum(saturation) / 3,
                   'normal_eigenvalues': spectrum.tolist(),
                   'ridge_shrinkage_eigenvalues': (spectrum / (spectrum + ridge)).tolist()}
    return velocity, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/model45_root_cause/control')
    parser.add_argument('--case', default='1AFQ_B_C_0FG')
    parser.add_argument('--seeds', type=int, nargs='+', default=[73, 74])
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--initial-state', help='saved CUDA initializer state, loaded onto CPU')
    parser.add_argument('--modes', nargs='+', default=['unrestricted', 'unit_bound', 'oracle'])
    parser.add_argument('--oracle-cycles', type=int, default=1)
    parser.add_argument('--oracle-gain', type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, _, cases = _load_index('results/model45_partition/ternary_train.json')
    item = cases[args.case]
    ed, pd, ld, _ = featurize_complex(args.case, item['protein1'], item['protein2'], item['ligand'],
        _load_score_model_args('checkpoints/score_model'), 'cpu',
        cache_dir='/mnt/disk/wl/MoRE-tri/CoopDiff/results/featurize_cache_proposal_oracle',
        cache_read_only=True, cache_only=True)
    metric = validation_metric_context(item, '/mnt/disk/wl/MoRE-tri/CoopDiff/DeepTernary/data/TernaryDB/pdbs')
    no_torsion = bool(getattr(_load_score_model_args('checkpoints/score_model'), 'no_torsion', False))
    if no_torsion:
        raise ValueError('this diagnostic currently requires enabled ligand torsions')
    flow = SDPIFlow(inference_steps=args.steps)
    scales = flow.scales(ed['coords'], pd['coords']).double()
    e3 = ed['coords'].double()
    report = {'config': vars(args), 'prediction_claim': False, 'no_torsion': no_torsion,
              'initialization_device': 'saved_state' if args.initial_state else 'cpu',
              'scales_angstrom': scales.tolist(), 'runs': [], 'limitations': [
                  'native goals are used directly: oracle diagnostic only',
                  'unit bounds are expert-scale reference, not v39 total field bounds',
                  'oracle repair uses fixed 16 residue anchors, not learned material assignments',
                  'a failed local control optimization does not prove unreachability']}
    def save():
        (out / 'summary.json').write_text(json.dumps(report, indent=2))
    def metrics(state, target):
        ligand, poi, _ = state
        return {'ligand_target_rmsd': float((ligand - target[0]).square().sum(-1).mean().sqrt()),
                'poi_native_rmsd': float((poi - target[1]).square().sum(-1).mean().sqrt()),
                **compute_fullatom_dockq(ligand[0].detach().numpy(), poi[0].detach().numpy(),
                                        e3[0].numpy(), **metric)}
    for seed in args.seeds:
        if args.initial_state:
            saved = torch.load(args.initial_state, map_location='cpu', weights_only=False)
            if saved['seed'] != seed or saved['case'] != args.case:
                raise ValueError('saved initial state case/seed mismatch')
            initial = saved['state'][:3]
        else:
            initial = TernaryInitializer(no_torsion=False).initialize(ed['coords'][0], pd['coords'][0], ld, seed=seed)[:3]
        projected, projection = reachable_ligand_target(ld, initial[0], initial[1], pd['coords'])
        initial = tuple(x.double() for x in initial)
        target = (torch.as_tensor(projected).double().reshape_as(initial[0]), pd['coords'].double())
        width = 12 + int(ld['edge_mask'].sum())
        projection = json.loads(json.dumps(projection, default=lambda x: x.item() if hasattr(x, 'item') else x.tolist()))
        base = {'seed': seed, 'initial': metrics(initial, target), 'projection': projection}
        print(json.dumps({'event': 'initial', **base}), flush=True)
        for mode in (m for m in ('unrestricted', 'unit_bound') if m in args.modes):
            started = time.monotonic()
            controls = torch.nn.Parameter(torch.zeros(args.steps, width, dtype=torch.double))
            optimizer = torch.optim.LBFGS([controls], lr=1., max_iter=args.iterations,
                max_eval=args.iterations * 2, tolerance_grad=1e-8, tolerance_change=1e-12,
                line_search_fn='strong_wolfe')
            progress = []
            def rollout():
                state = initial
                velocities = bound(controls) if mode == 'unit_bound' else controls
                for vector in velocities:
                    state = advance(state, vector, scales.new_tensor(1 / args.steps), scales, ld)
                return state
            def closure():
                optimizer.zero_grad()
                state = rollout()
                loss = ((state[0] - target[0]) / scales[0]).square().mean() + ((state[1] - target[1]) / scales[1]).square().mean()
                loss.backward()
                if not torch.isfinite(controls.grad).all():
                    raise FloatingPointError('nonfinite control gradient')
                progress.append({'evaluation': len(progress) + 1, 'loss': float(loss.detach()),
                                 'gradient_norm': float(controls.grad.norm())})
                if len(progress) % 20 == 0:
                    print(json.dumps({'seed': seed, 'mode': mode, **progress[-1]}), flush=True)
                return loss
            optimizer.step(closure)
            with torch.no_grad():
                final = rollout()
                velocity = bound(controls) if mode == 'unit_bound' else controls
            record = {**base, 'mode': mode, 'final': metrics(final, target),
                      'elapsed_seconds': time.monotonic() - started, 'progress': progress,
                      'maximum_normalized_rigid_norm': float(velocity[:, :12].reshape(-1, 3).norm(dim=-1).max()),
                      'maximum_normalized_torsion': float(velocity[:, 12:].abs().max()) if width > 12 else 0}
            torch.save({'initial': initial, 'target': target, 'final': final,
                        'controls': velocity.detach()}, out / f'seed{seed}_{mode}.pt')
            report['runs'].append(record)
            save()
            print(json.dumps({'event': 'complete', 'seed': seed, 'mode': mode, 'final': record['final']}), flush=True)
        if 'oracle' not in args.modes:
            continue
        for ridge, saturate in ((.01, True), (.0001, True), (.01, False), (.0001, False)):
            state = initial
            trace = []
            with torch.no_grad():
                for t, dt in list(flow.reverse_times(initial[0])) * args.oracle_cycles:
                    vector, diagnostics = oracle_repair(state, target, e3, scales, ld, t[0], ridge, saturate)
                    state = advance(state, vector * args.oracle_gain, dt, scales, ld)
                    trace.append({'time': float(t), **diagnostics,
                                  'velocity_norm': float(vector.norm())})
            record = {**base, 'mode': 'oracle_distance_repair', 'ridge': ridge, 'saturate': saturate,
                      'final': metrics(state, target), 'trace': trace}
            report['runs'].append(record)
            save()
            print(json.dumps({'event': 'complete', 'seed': seed, 'mode': record['mode'],
                              'ridge': ridge, 'saturate': saturate, 'final': record['final']}), flush=True)
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
