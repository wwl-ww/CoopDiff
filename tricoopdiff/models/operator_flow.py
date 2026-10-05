"""Finite-response control of frozen expert operators on rigid/torsion states.

Probes use a fixed, finite clock increment. No division by epsilon, Lie-bracket
limit, hard routing, coordinate averaging, or biological cooperativity claim.
"""

import torch
from torch import nn

from tricoopdiff.models.constraint_motion import binary_rigid_maps, DISTANCE_SCALE
from tricoopdiff.sdpi_flow import SDPIFlow, INDEPENDENT_RIGID_MODE, SYMMETRIC_TORSION_MODE


OPERATOR_FLOW_CONTRACT = 'independent_stage_finite_repair_v2'
PROBE_STEP = .025
EXPERT_EVALUATIONS_PER_STEP = 6


def pack_velocity(vector):
    """Normalized independent-centroid velocity [12+K] -> integrator schema."""
    result = {'rigid_output_mode': INDEPENDENT_RIGID_MODE,
              'torsion_update_mode': SYMMETRIC_TORSION_MODE, 'torsion': vector[None, 12:]}
    for index, branch in enumerate(('e3_ligand', 'poi_ligand')):
        offset = 6 * index
        result[branch] = {'tr': vector[None, offset:offset + 3],
                          'rot': vector[None, offset + 3:offset + 6],
                          'tor': vector[None, 12:]}
    return result


def unpack_velocity(value):
    return torch.cat([value[branch][component][0]
                      for branch in ('e3_ligand', 'poi_ligand') for component in ('tr', 'rot')]
                     + [value['torsion'][0]])


def lift_binary(proposal, branch, state, e3_coords, scales):
    """Minimum normalized rigid-norm lift, plus the expert's shared torsions.

    B translation is measured at the ligand centroid. Applying it directly to
    the POI would have the wrong sign, lever arm, and receptor scale.
    """
    from tricoopdiff.models.consensus import bounded_vector
    ligand, poi, _ = state
    centers = torch.stack((e3_coords[0].mean(0), poi[0].mean(0), ligand[0].mean(0)))
    mapping = binary_rigid_maps(centers, scales, 12)[branch].double()
    target = torch.cat([bounded_vector(proposal[key])[0] for key in ('tr', 'rot')]).double()
    rigid = mapping.T @ torch.linalg.solve(mapping @ mapping.T, target)
    return torch.cat((rigid.to(ligand), proposal['tor'][0].tanh()))


def limit_step(vector, duration, max_angle=.5):
    """Limit the executed angular increment, preserving the field direction.

    Applies after repair/expert blending as well as to the repair solve. Rigid
    norms and torsion magnitudes are invariant to the global coordinate frame.
    """
    angular = vector[[3, 4, 5, 9, 10, 11]].reshape(2, 3).norm(dim=-1).max()
    if len(vector) > 12:
        angular = torch.maximum(angular, vector[12:].abs().max())
    step_angle = angular * torch.pi * duration.reshape(())
    fraction = (max_angle / step_angle.clamp_min(1e-12)).clamp_max(1.)
    return vector * fraction


def advance(state, vector, duration, scales, ligand_data):
    """Execute an actual rigid/torsion substep with the outer flow's units."""
    if not bool(duration.detach().ne(0).any()) and not duration.requires_grad:
        return state
    velocity = pack_velocity(vector)
    for index, branch in enumerate(('e3_ligand', 'poi_ligand')):
        velocity[branch]['tr'] = velocity[branch]['tr'] * scales[index]
        velocity[branch]['rot'] = velocity[branch]['rot'] * torch.pi
    velocity['torsion'] = velocity['torsion'] * torch.pi
    return SDPIFlow.differentiable_step(*state, velocity, duration.reshape(1), ligand_data)


def response_invariants(base, cross):
    """Thirty invariant strengths/alignments; finite even with zero torsions.

    cross[0] is A after a B probe; cross[1] is B after an A probe. Quantities
    are finite responses at a fixed scale, not numerical derivatives.
    """
    features = []
    for before, after in zip(base, cross):
        a, b = before[:12].reshape(4, 3), after[:12].reshape(4, 3)
        features.append(torch.stack((a.square().sum(-1), b.square().sum(-1),
                                     (a * b).sum(-1)), -1).flatten())
        if before.shape[0] > 12:
            a, b = before[12:], after[12:]
            features.append(torch.stack((a.square().mean(), b.square().mean(), (a * b).mean())))
        else:
            features.append(before.new_zeros(3))
    return torch.cat(features)[None]


def goal_probe_evidence(predictions, states, e3_coords):
    """Nine invariant goal errors/deltas, using predictions only, never labels."""
    features = []
    for item in predictions:
        errors = []
        for ligand, poi, _ in states:
            proteins = (e3_coords[0], poi[0])
            anchors = item['anchor_weights'] @ proteins[item['anchor_type']]
            receiver = ligand[0] if item['receiver_type'] == 2 else poi[0]
            delta = anchors[None] - receiver[:, None]
            distance = (delta.square().sum(-1) + .01).sqrt() / DISTANCE_SCALE
            errors.append((torch.log1p(distance) - torch.log1p(item['distance'].detach())).square().mean())
        features.append(torch.stack((errors[0], errors[1] - errors[0], errors[2] - errors[0])))
    return torch.cat(features)[None]


class OperatorController(nn.Module):
    """Independent expert preferences, positive time split, and repair gains.

    Each stage has its own expert preference, avoiding first-order cancellation
    at a balanced initialization. Positive repair gains can exceed one; experts
    are attenuated by 1/(1+gain), without a convex-combination movement ceiling.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.policy = nn.Sequential(nn.LayerNorm(hidden_dim + 39),
            nn.Linear(hidden_dim + 39, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5))
        # Small nonzero readout lets response features receive gradients from
        # the first optimizer update while starting near balanced control.
        nn.init.normal_(self.policy[-1].weight, std=.01)
        nn.init.zeros_(self.policy[-1].bias)
        with torch.no_grad():
            self.policy[-1].bias[3:].fill_(0.541324854612918)  # inverse softplus(1)

    def forward(self, context, base, cross, predictions, states, e3_coords):
        evidence = torch.cat((response_invariants(base, cross),
                              goal_probe_evidence(predictions, states, e3_coords)), -1)
        logits = self.policy(torch.cat((context, evidence), -1))[0]
        controls = logits[:3].sigmoid()
        gains = torch.nn.functional.softplus(logits[3:])
        return {'order_first': controls[0], 'order_second': controls[1],
                'split': .25 + .5 * controls[2],
                'repair_first': gains[0], 'repair_second': gains[1]}, evidence

    @staticmethod
    def blend(fields, repair, order, repair_weight):
        return (order * fields[0] + (1 - order) * fields[1]) / (1 + repair_weight) + repair_weight * repair
