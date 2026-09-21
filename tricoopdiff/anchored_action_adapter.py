"""Learn feasible torsion corrections from actual constrained-body motion.

Interface directions are learned coarse geometric messages, not physical
forces. Their contractions with decoder tangents expose which linker axes can
change protein contacts, including POI motion omitted by ligand-only tangents.
"""

import math

import torch
from torch import nn


ADAPTER_POLICY = 'anchored_body_aware_torsion_residual_v1'
ADAPTER_ARCHITECTURE = 'conditioned_consensus_v8_with_anchored_body_aware_torsion_v1'


class AnchoredActionAdapter(nn.Module):
    policy = ADAPTER_POLICY

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * hidden_dim + 12, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
                                  nn.Linear(hidden_dim // 2, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(flow, state, context, velocity, scales, time):
        count = len(flow.plan.bonds)
        if not count:
            return velocity.new_zeros(0)
        ligand, poi, _ = flow.decode(state)
        ligand, poi = ligand.detach(), poi.detach()
        ligand_force, poi_force = context['ligand_directions'], context['poi_directions']
        if ligand_force.shape != ligand.shape or poi_force.shape != poi.shape:
            raise ValueError('body-aware directions must follow the current ligand/POI atom order')
        scales = torch.as_tensor(scales, dtype=state.dtype, device=state.device).reshape(2)
        length_scale = scales.mean()
        single = hasattr(flow, 'translation_bound')
        translations = [flow.translation_bound] if single else getattr(flow, 'translation_bounds', [0., 0.])
        rotations = [flow.rotation_bound] if single else getattr(flow, 'rotation_bounds', [0., 0.])
        offset_count = 6 if single else (12 if hasattr(flow, 'translation_bounds') else 0)
        offset_norm = (state[count:count + offset_count].detach().square().mean().sqrt()
                       if offset_count else state.new_zeros(()))
        scalars = []
        for index, ((u, v), moving, follows) in enumerate(zip(
                flow.plan.bonds, flow.plan.moving_atoms, flow.plan.moves_poi)):
            axis = ligand[u] - ligand[v]
            if axis.norm() < 1e-8:
                raise ValueError('degenerate body-aware torsion axis')
            axis = axis / axis.norm()
            tangent_l = torch.zeros_like(ligand)
            ids = list(moving)
            tangent_l[ids] = torch.linalg.cross(axis.expand(len(ids), 3), ligand[ids] - ligand[v])
            tangent_p = (torch.linalg.cross(axis.expand_as(poi), poi - ligand[v]) if follows
                         else torch.zeros_like(poi))
            work_l = (ligand_force * tangent_l / length_scale).sum(-1).mean()
            work_p = (poi_force * tangent_p / length_scale).sum(-1).mean()
            constants = state.new_tensor([len(ids) / len(ligand), float(follows), float(time),
                sum(translations) / len(translations) / float(length_scale),
                sum(rotations) / len(rotations) / math.pi, float(single)])
            scalars.append(torch.stack((torch.tanh(work_l), torch.tanh(work_p),
                torch.log1p(tangent_l.square().mean().sqrt() / length_scale),
                torch.log1p(tangent_p.square().mean().sqrt() / length_scale),
                constants[0], constants[1], constants[2], torch.tanh(velocity[index] / math.pi),
                constants[3], constants[4], torch.log1p(offset_norm), constants[5])))
        features = torch.stack(scalars)
        bond = context['bond'][0, list(flow.plan.original_indices)]
        global_context = context['global'].expand(count, -1)
        if bond.shape != global_context.shape or not torch.isfinite(features).all():
            raise ValueError('invalid body-aware feature/context contract')
        return torch.cat((bond, global_context, features), -1)

    def forward(self, flow, state, context, velocity, scales, time):
        if not len(flow.plan.bonds):
            return velocity.new_zeros(0)
        features = self.features(flow, state, context, velocity, scales, time)
        return math.pi * torch.tanh(self.net(features).squeeze(-1))


def install_action_adapter(model, *, policy=ADAPTER_POLICY):
    """Explicit, zero-output migration; preserve base parameters and RNG stream."""
    from .conditional_consensus import ConditionedConsensusFuse
    if not isinstance(model.trifuse, ConditionedConsensusFuse):
        raise ValueError('body-aware action adapter requires the conditioned consensus backbone')
    if hasattr(model.trifuse, 'anchored_action_adapter'):
        raise ValueError('anchored action adapter is already installed')
    from .coupled_linker_action import (CoupledLinkerActionAdapter, COUPLED_ADAPTER_POLICY,
                                        COUPLED_ADAPTER_ARCHITECTURE)
    from .prior_joint_action import PriorJointActionAdapter, PRIOR_ACTION_POLICY, PRIOR_ACTION_ARCHITECTURE
    if policy not in (ADAPTER_POLICY, COUPLED_ADAPTER_POLICY, PRIOR_ACTION_POLICY):
        raise ValueError('unsupported anchored action adapter policy')
    coupled = policy == COUPLED_ADAPTER_POLICY
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1702)
        adapter_type = CoupledLinkerActionAdapter if coupled else AnchoredActionAdapter
        if policy == PRIOR_ACTION_POLICY:
            adapter_type = PriorJointActionAdapter
        adapter = adapter_type(model.trifuse.config['node_dim'])
    reference = next(model.trifuse.parameters())
    model.trifuse.add_module('anchored_action_adapter', adapter.to(reference))
    model.architecture = COUPLED_ADAPTER_ARCHITECTURE if coupled else ADAPTER_ARCHITECTURE
    if policy == PRIOR_ACTION_POLICY:
        model.architecture = PRIOR_ACTION_ARCHITECTURE
