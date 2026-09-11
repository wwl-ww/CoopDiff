#!/usr/bin/env python
"""Measure parameter-gradient conflicts among fixed endpoint translations."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat
from local_fixed_translation_overfit import build_trainer, fixed_state
from local_paired_pilot import load_batches, sha256
from train_entry import _load_score_model_args
from tricoopdiff.checkpoint_contract import validate_expert_base_identity
from tricoopdiff.consensus import bounded_vector
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility


def parameter_gradient(loss, parameters, *, retain_graph=False):
    """Flatten one loss gradient over a stable trainable parameter list."""
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]).detach().cpu()


def cosine_matrix(vectors):
    """Return pairwise cosine values for nonzero flattened vectors."""
    matrix = torch.stack(vectors).double()
    norms = torch.linalg.vector_norm(matrix, dim=-1)
    if bool((norms <= 0).any()):
        raise ValueError('gradient vectors must be nonzero')
    return (matrix @ matrix.T / (norms[:, None] * norms[None, :])).numpy()


def vector_cosine(left, right):
    """Cosine between finite singleton 3-vectors."""
    denominator = (torch.linalg.vector_norm(left)
                   * torch.linalg.vector_norm(right)).clamp_min(1e-12)
    return float(((left * right).sum() / denominator).item())


def nearest_contact_direction(ligand, receptor, neighbors=4):
    """Invariant-weighted equivariant direction from ligand atoms to a surface."""
    distances = torch.cdist(ligand[0], receptor[0])
    k = min(neighbors, receptor.shape[1])
    values, indices = torch.topk(distances, k, dim=1, largest=False, sorted=True)
    displacements = receptor[0][indices] - ligand[0][:, None]
    bounded = displacements / (
        10 + torch.linalg.vector_norm(displacements, dim=-1, keepdim=True))
    weights = torch.softmax(-values / 4, dim=1)
    return (bounded * weights[..., None]).sum(dim=1).mean(dim=0, keepdim=True)


def translation_basis_diagnostic(output, batch, ligand, poi, targets):
    """Compare the current global basis with explicit local contact directions."""
    raw = [
        output['binary_proposals'][branch][component]
        for branch in ('e3_ligand', 'poi_ligand')
        for component in ('tr', 'rot')
    ]
    centers = [batch['e3_data']['coords'].mean(1), poi.mean(1), ligand.mean(1)]
    center_directions = [
        bounded_vector((centers[index] - centers[2]) / 10)
        for index in (0, 1)
    ]
    old_vectors = [bounded_vector(value) for value in raw] + center_directions + [
        torch.cross(center_directions[0], center_directions[1], dim=-1),
        torch.cross(bounded_vector(raw[0]), bounded_vector(raw[2]), dim=-1),
    ]
    contact_vectors = [
        nearest_contact_direction(ligand, batch['e3_data']['coords']),
        nearest_contact_direction(ligand, poi),
    ]
    old_basis = torch.cat(old_vectors, dim=0)
    augmented_basis = torch.cat((*old_vectors, *contact_vectors), dim=0)
    result = {}
    for branch, contact in zip(('e3_ligand', 'poi_ligand'), contact_vectors):
        target = targets[branch]['tr']
        result[branch] = {
            'contact_target_cosine': vector_cosine(contact, target),
            'center_target_cosine': vector_cosine(
                center_directions[0 if branch == 'e3_ligand' else 1], target),
            'expert_target_cosine': vector_cosine(
                raw[0 if branch == 'e3_ligand' else 2], target),
            'old_basis_max_abs_cosine': max(
                abs(vector_cosine(vector[None], target)) for vector in old_basis),
            'augmented_basis_max_abs_cosine': max(
                abs(vector_cosine(vector[None], target)) for vector in augmented_basis),
        }
    for name, basis in (('old', old_basis), ('contact_augmented', augmented_basis)):
        singular = torch.linalg.svdvals(basis.double())
        result[f'{name}_basis_condition'] = float(
            (singular.max() / singular.min().clamp_min(1e-12)).item())
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', nargs='+', default=(
        '3CXH_C_E_6PH', '4I0P_D_H_PGE', '4IL6_C_I_LMT'))
    ap.add_argument('--times', nargs='+', type=float, default=(.8, .5, .3))
    ap.add_argument('--seed', type=int, default=27201)
    ap.add_argument('--couplings-per-state', type=int, default=1)
    args = ap.parse_args()
    if args.couplings_per_state < 1:
        raise ValueError('couplings-per-state must be positive')
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    policy = configure_inference_reproducibility(True)
    device = 'cuda'
    score_args = _load_score_model_args(str(args.artifacts / 'checkpoints/score_model'))
    batches = load_batches(args.artifacts, 'val', 8, device, score_args)
    by_case = {batch['case']: batch for batch in batches}
    missing = set(args.cases) - set(by_case)
    if missing:
        raise ValueError(f'cases not in validation index: {sorted(missing)}')
    base = torch.load(args.base, map_location='cpu', weights_only=False)
    expert = build_diffdock_expert(str(args.artifacts / 'checkpoints/score_model'), device)
    validate_expert_base_identity(base, expert)
    trainer = build_trainer(base, expert, device)
    parameters = [p for group in trainer.optimizer.param_groups
                  for p in group['params'] if p.requires_grad]
    labels, joint_gradients, within_state, basis_diagnostics = [], [], [], []
    target_vectors, context_vectors = [], []
    captured = {}
    def capture_context(_module, _inputs, output):
        captured['context'] = output.mean(dim=1).detach().cpu().reshape(-1)
    hook = trainer.model.trifuse.transformer.register_forward_hook(capture_context)
    for case_index, case in enumerate(args.cases):
        batch = copy.deepcopy(by_case[case])
        for time_value in args.times:
            for coupling_index in range(args.couplings_per_state):
                run_seed = (args.seed + case_index * 100 + round(time_value * 10)
                            + coupling_index * 10_000)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)
                torch.cuda.manual_seed_all(run_seed)
                coupling = trainer.diffusion.sample_coupling(
                    batch['e3_data']['coords'], batch['poi_data']['coords'],
                    batch['ligand_data'])
                time = torch.tensor([time_value], device=device)
                state = fixed_state(trainer, batch, coupling, time)
                ligand, poi, body, targets, _ = state
                output = trainer._forward_state(
                    batch['e3_data'], batch['poi_data'], batch['ligand_data'],
                    ligand, poi, body, time, time.new_full((1,), .05))
                context_vectors.append(captured['context'])
                basis_diagnostics.append(translation_basis_diagnostic(
                    output, batch, ligand, poi, targets))
                normalized = trainer.diffusion.normalized_predictions(output, time)
                velocities = trainer.diffusion.velocities(
                    normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
                scales = trainer.diffusion.scales(
                    batch['e3_data']['coords'], batch['poi_data']['coords'])
                target_vectors.append(torch.cat((
                    (targets['e3_ligand']['tr'] / scales[0]).reshape(-1),
                    (targets['poi_ligand']['tr'] / scales[1]).reshape(-1),
                )).detach().cpu())
                branch_losses = [
                    ((velocities[branch]['tr'] - targets[branch]['tr']) / scale)
                    .square().mean()
                    for branch, scale in zip(
                        ('e3_ligand', 'poi_ligand'), scales)
                ]
                branch_gradients = [
                    parameter_gradient(branch_losses[0], parameters, retain_graph=True),
                    parameter_gradient(branch_losses[1], parameters),
                ]
                joint_gradients.append(branch_gradients[0] + branch_gradients[1])
                local = cosine_matrix(branch_gradients)
                within_state.append(float(local[0, 1]))
                labels.append({
                    'case': case, 'time': time_value, 'seed': run_seed,
                    'coupling_index': coupling_index,
                })
                print(json.dumps({
                    **labels[-1], 'branch_gradient_cosine': within_state[-1],
                }), flush=True)
    hook.remove()
    matrix = cosine_matrix(joint_gradients)
    target_matrix = cosine_matrix(target_vectors)
    context_matrix = cosine_matrix(context_vectors)
    upper = matrix[np.triu_indices(len(matrix), 1)]
    same_state, other_state = [], []
    same_target, same_context = [], []
    for i, j in zip(*np.triu_indices(len(matrix), 1)):
        is_same = (labels[i]['case'] == labels[j]['case']
                   and labels[i]['time'] == labels[j]['time'])
        target = same_state if is_same else other_state
        target.append(float(matrix[i, j]))
        if is_same:
            same_target.append(float(target_matrix[i, j]))
            same_context.append(float(context_matrix[i, j]))
    def summary(values):
        if not values:
            return {'count': 0}
        values = np.asarray(values)
        return {
            'mean': float(values.mean()), 'median': float(np.median(values)),
            'minimum': float(values.min()), 'maximum': float(values.max()),
            'negative_count': int((values < 0).sum()), 'count': len(values),
        }
    report = {
        'status': 'complete', 'contract': 'translation_gradient_conflicts_v3',
        'base': str(args.base), 'base_sha256': sha256(args.base),
        'couplings_per_state': args.couplings_per_state,
        'reproducibility': policy, 'labels': labels,
        'within_state_branch_gradient_cosine': within_state,
        'translation_basis_diagnostics': basis_diagnostics,
        'joint_state_gradient_cosine_matrix': matrix.tolist(),
        'translation_target_cosine_matrix': target_matrix.tolist(),
        'trifuse_context_cosine_matrix': context_matrix.tolist(),
        'joint_state_off_diagonal': {
            'mean': float(upper.mean()), 'median': float(np.median(upper)),
            'minimum': float(upper.min()), 'maximum': float(upper.max()),
            'negative_count': int((upper < 0).sum()), 'count': len(upper),
        },
        'same_case_time_different_coupling': summary(same_state),
        'same_case_time_target_cosine': summary(same_target),
        'same_case_time_context_cosine': summary(same_context),
        'other_state_pairs': summary(other_state),
    }
    args.out.joinpath('report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
