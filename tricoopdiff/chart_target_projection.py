"""Explicit supervision-only projection onto an unchanged deployment chart.

This is NOT exact native target encoding. The returned endpoint is a local
least-squares fit, accompanied by its residual; it must never be substituted
silently into the legacy native-supervision training contract.
"""

from dataclasses import dataclass
import math

import torch

from .bounded_binary_flow import BoundedBinaryTorsionFlow
from .chemical_torsion_prior import transfer_dihedrals


POLICY = 'fixed_deployment_chart_supervision_multistart_projection_v2'


@dataclass
class ProjectedChartTarget:
    state: torch.Tensor
    ligand: torch.Tensor
    poi: torch.Tensor
    audit: dict


def project_chart_target(flow, ligand, poi, edge_index, *, max_iterations=100,
                         restarts=8, seed=7315):
    """Fit only existing torsion/offset DOFs, never reference internal geometry.

    The loss weights ligand and POI equally by mean squared atom distance in
    the supplied E3 frame. There is no free global alignment or native-driven
    reference selection. Convergence/global optimality is not guaranteed.
    """
    if not isinstance(flow, BoundedBinaryTorsionFlow):
        raise ValueError('projection currently requires a bounded binary chart')
    if (type(max_iterations) is not int or max_iterations < 1
            or type(restarts) is not int or restarts < 1
            or type(seed) is not int or seed < 0):
        raise ValueError('projection needs positive iteration/restart budgets and a nonnegative seed')
    for target, reference in ((ligand, flow.strict.ligand), (poi, flow.strict.poi)):
        if (target.shape != reference.shape or target.dtype != reference.dtype
                or target.device != reference.device or not torch.isfinite(target).all()):
            raise ValueError('projection target must match chart shape/device/dtype')
    ligand, poi = ligand.detach(), poi.detach()
    state = ligand.new_zeros(flow.size)
    # Only an optimizer start, not a declaration that native belongs to chart.
    state[:flow.torsion_count] = transfer_dihedrals(
        flow.strict.ligand, flow.plan, edge_index, ligand[None])[0]
    best = state.clone()
    evaluations = 0

    def objective(value):
        predicted_ligand, predicted_poi, _ = flow.decode(value)
        return .5 * ((predicted_ligand - ligand).square().sum(-1).mean()
                     + (predicted_poi - poi).square().sum(-1).mean())

    with torch.no_grad():
        initial = float(objective(state))
    best_loss = initial
    generator = torch.Generator(device=state.device).manual_seed(seed)
    starts = [state]
    for _ in range(1, restarts):
        start = state.clone()
        if flow.torsion_count:
            start[:flow.torsion_count] = (torch.rand(
                flow.torsion_count, dtype=state.dtype, device=state.device,
                generator=generator) * 2 - 1) * math.pi
        # Offset latents are dimensionless. Moderate deterministic starts probe
        # both bounded pose directions without beginning at saturated maps.
        start[flow.torsion_count:] = torch.randn(
            flow.size - flow.torsion_count, dtype=state.dtype,
            device=state.device, generator=generator) * .75
        starts.append(start)
    restart_rows = []
    for restart, start in enumerate(starts):
        local_best, local_loss, local_evaluations = start.clone(), float(objective(start)), 0
        with torch.enable_grad():
            parameter = start.clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS([parameter], lr=1., max_iter=max_iterations,
                                         tolerance_grad=1e-8, tolerance_change=1e-10,
                                         line_search_fn='strong_wolfe')

            def closure():
                nonlocal local_best, local_loss, local_evaluations, evaluations
                optimizer.zero_grad(set_to_none=True)
                loss = objective(parameter)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite chart projection objective')
                loss.backward()
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('nonfinite chart projection gradient')
                value = float(loss.detach())
                local_evaluations += 1
                evaluations += 1
                if value < local_loss:
                    local_best, local_loss = parameter.detach().clone(), value
                return loss

            optimizer.step(closure)
        restart_rows.append({'restart': restart, 'initial_mse': float(objective(start)),
                             'final_mse': local_loss, 'evaluations': local_evaluations})
        if local_loss < best_loss:
            best, best_loss = local_best, local_loss
    with torch.no_grad():
        predicted_ligand, predicted_poi, _ = flow.decode(best)
        residual = {}
        for name, prediction, target in (('ligand', predicted_ligand, ligand), ('poi', predicted_poi, poi)):
            errors = (prediction - target).norm(dim=-1)
            residual[name + '_rmsd'] = float(errors.square().mean().sqrt())
            residual[name + '_max_error'] = float(errors.max())
        # Auditable fixed-geometry lower bound: each bond length mismatch
        # implies at least half that mismatch at one of its endpoint atoms.
        edges = torch.as_tensor(edge_index, device=ligand.device)
        u, v = edges
        reference = flow.strict.ligand
        differences = ((reference[u] - reference[v]).norm(dim=-1)
                       - (ligand[u] - ligand[v]).norm(dim=-1)).abs()
        lower_bound = float(differences.max()) / 2 if differences.numel() else 0.
    if not math.isfinite(best_loss):
        raise FloatingPointError('nonfinite projected target')
    return ProjectedChartTarget(best, predicted_ligand.detach(), predicted_poi.detach(), {
        'policy': POLICY, 'target_semantics': 'projected_endpoint_not_native',
        'native_used_for': 'supervision_only', 'reference_modified': False,
        'global_alignment_applied': False, 'global_optimum_claimed': False,
        'initial_equal_body_mse': initial, 'final_equal_body_mse': best_loss,
        'optimizer_evaluations': evaluations, 'max_iterations': max_iterations,
        'restarts': restarts, 'seed': seed, 'restart_results': restart_rows,
        'ligand_max_error_bond_lower_bound': lower_bound, **residual})
