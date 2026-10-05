"""Independent regression checks for identifiable geometry and executed control."""

from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import tricoopdiff.diffdock_compat  # noqa: F401
from test_consensus import _operator_model_fixture
from tricoopdiff.models.constraint_motion import terminal_geometry_loss
from tricoopdiff.models.operator_flow import OperatorController, advance, goal_probe_evidence
from tricoopdiff.sdpi_flow import SDPIFlow


def test_terminal_distance_head_has_auxiliary_but_no_executed_endpoint_gradient():
    model, _, ed, pd, ld = _operator_model_fixture()
    flow = SDPIFlow()
    output = flow.forward_state(model, ed, pd, ld, ld['coords'], pd['coords'],
                                torch.eye(3)[None], torch.ones(1), torch.tensor([.05]))
    distance = tuple(model.trifuse.constraint_head.distance_head.parameters())
    precision = tuple(model.trifuse.constraint_head.precision_head.parameters())
    state = output['operator_update']['state']
    endpoint = state[0].square().mean() + state[1].square().mean()
    gradients = torch.autograd.grad(endpoint, distance + precision,
                                    allow_unused=True, retain_graph=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in gradients[:len(distance)]), \
        'executed endpoint must not train distance predictions as compensating control commands'
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in gradients[len(distance):]), 'precision control lost its endpoint gradient'
    auxiliary = torch.stack([terminal_geometry_loss(stage, ed['coords'][0], pd['coords'][0],
                                                    ld['coords'])
                             for stage in output['geometry_stages']]).mean()
    gradients = torch.autograd.grad(auxiliary, distance, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients), \
        'distance head must remain learnable from geometric supervision'


def test_probe_goal_evidence_stops_goal_gradient_but_retains_state_gradient():
    ligand = torch.tensor([[[2., 1., 0.], [3., 0., 1.]]], requires_grad=True)
    poi = torch.tensor([[[4., 0., 0.], [4., 1., 0.]]], requires_grad=True)
    fixed = torch.tensor([[[0., 0., 0.], [0., 1., 0.]]])
    goal = torch.ones(2, 2, requires_grad=True)
    prediction = {'distance': goal, 'anchor_weights': torch.eye(2),
                  'anchor_type': 0, 'receiver_type': 2}
    state = ligand, poi, torch.eye(3)[None]
    evidence = goal_probe_evidence([prediction], (state, state, state), fixed)
    goal_gradient, state_gradient = torch.autograd.grad(evidence.sum(), (goal, ligand), allow_unused=True)
    assert goal_gradient is None or goal_gradient.abs().sum() == 0, \
        'policy goal evidence reopened the forbidden endpoint-to-distance path'
    assert state_gradient is not None and torch.isfinite(state_gradient).all() \
        and state_gradient.abs().sum() > 0, 'detach must not suppress geometric state evidence'


def test_independent_order_controls_each_change_the_executed_endpoint_at_first_order():
    # Constant translation fields remove commutators: each order must work at first order.
    dtype = torch.double
    ligand = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=dtype)
    poi = ligand + 3.
    state = ligand, poi, torch.eye(3, dtype=dtype)[None]
    fields = [torch.zeros(12, dtype=dtype), torch.zeros(12, dtype=dtype)]
    fields[0][0], fields[1][6] = 1., 1.
    repair = torch.zeros(12, dtype=dtype)
    orders = torch.tensor([.5, .5], dtype=dtype, requires_grad=True)
    scales = torch.ones(2, dtype=dtype)
    ld = {'coords': ligand}
    middle = advance(state, OperatorController.blend(fields, repair, orders[0],
                     orders.new_tensor(1.)), orders.new_tensor(.025), scales, ld)
    final = advance(middle, OperatorController.blend(fields, repair, orders[1],
                    orders.new_tensor(1.)), orders.new_tensor(.025), scales, ld)
    endpoint = final[0][0, :, 0].mean() - final[1][0, :, 0].mean()
    gradient = torch.autograd.grad(endpoint, orders)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > .001).all(), \
        'each independent order must affect the endpoint without higher-order field variation'
    assert torch.allclose(gradient[0], gradient[1], atol=1e-12), \
        'the second control must not secretly reverse the first preference'

    # Also verify the production wrapper consumes both keys independently.
    model, _, ed, pd, ligand_data = _operator_model_fixture(no_torsion=True)
    controls = torch.tensor([.4, .6], requires_grad=True)
    model.trifuse.operator_controller.forward = lambda *args: (
        {'order_first': controls[0], 'order_second': controls[1],
         'split': controls.new_tensor(.5), 'repair_first': controls.new_tensor(1.),
         'repair_second': controls.new_tensor(1.)}, controls.new_zeros(1, 39))
    result = SDPIFlow().forward_state(model, ed, pd, ligand_data, ligand_data['coords'],
                                     pd['coords'], torch.eye(3)[None], torch.ones(1),
                                     torch.tensor([.05]))
    endpoint = sum(value.square().mean() for value in result['operator_update']['state'][:2])
    gradient = torch.autograd.grad(endpoint, controls)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > 1e-9).all(), \
        'production execution ignored one of the independent order controls'


def test_repair_gain_can_exceed_the_expert_repair_convex_hull():
    fields = [torch.zeros(12), torch.zeros(12)]
    repair = torch.zeros(12)
    repair[0] = 1.
    gain = torch.tensor(2., requires_grad=True)
    velocity = OperatorController.blend(fields, repair, torch.tensor(.5), gain)
    assert velocity[0] > 1., 'repair remains confined to the old convex hull'
    gradient = torch.autograd.grad(velocity[0], gain)[0]
    assert torch.isfinite(gradient) and gradient > 0, 'repair gain must remain learnable'


def test_finite_displacement_is_clock_consistent_and_zero_step_is_safe():
    from tricoopdiff.models.constraint_motion import solve_terminal_velocity
    matrix = torch.eye(12, dtype=torch.double)
    error = torch.linspace(-.3, .4, 12, dtype=torch.double, requires_grad=True)
    precision = torch.ones(12, dtype=torch.double) / 12
    displacements = []
    for duration in (1., .05, 1e-6):
        velocity, _ = solve_terminal_velocity(matrix, error, precision, error.new_tensor(duration))
        assert torch.isfinite(velocity).all()
        displacements.append(velocity * duration)
    assert all(torch.allclose(displacements[0], x, atol=1e-12) for x in displacements[1:]), \
        'finite displacement must not decay with the requested substep duration'
    gradient = torch.autograd.grad(displacements[-1].square().sum(), error)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    zero, _ = solve_terminal_velocity(matrix, error, precision, error.new_zeros(()))
    assert torch.equal(zero, torch.zeros_like(zero)), 'dt=0 must be an exact no-op'
    try:
        solve_terminal_velocity(matrix, error, precision, error.new_tensor(-.01))
    except ValueError:
        pass
    else:
        raise AssertionError('negative substep duration was accepted')


def test_actual_angular_step_remains_bounded_after_large_repair_gain():
    from tricoopdiff.models.operator_flow import limit_step
    fields = [torch.zeros(14, dtype=torch.double), torch.zeros(14, dtype=torch.double)]
    repair = torch.arange(1., 15., dtype=torch.double)
    dt = repair.new_tensor(.05)
    mixed = OperatorController.blend(fields, repair, repair.new_tensor(.5), repair.new_tensor(20.))
    limited = limit_step(mixed, dt)
    angles = torch.cat((limited[3:6].norm()[None], limited[9:12].norm()[None], limited[12:].abs())) * torch.pi * dt
    assert angles.max() <= .5 + 1e-12, 'learned gain escaped the actual-step angular trust bound'
    assert torch.isfinite(limited).all()
    ratios = limited / mixed
    assert torch.allclose(ratios, ratios[0].expand_as(ratios)), 'trust scaling changed the solved direction'


def test_production_geometry_head_recovers_an_oracle_rigid_assembly():
    from tricoopdiff.geometry.rigid import rotation_exp
    from tricoopdiff.models.constraint_motion import ConstraintMotionHead
    from tricoopdiff.models.operator_flow import unpack_velocity
    dtype = torch.double
    e3 = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.], [0., 0., 5.]], dtype=dtype)
    target_poi = e3 * .7 + torch.tensor([7., 2., -1.], dtype=dtype)
    target_ligand = e3 * .3 + torch.tensor([3., 1., 2.], dtype=dtype)
    rotation = rotation_exp(torch.tensor([[.4, -.6, .3]], dtype=dtype))[0]
    ligand = (target_ligand @ rotation.T + torch.tensor([5., -3., 2.], dtype=dtype))[None]
    poi = (target_poi @ rotation.T + torch.tensor([-4., 3., 1.], dtype=dtype))[None]
    state = ligand, poi, rotation[None]
    ld = {'coords': ligand}
    types = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    head = ConstraintMotionHead(4).double()
    native = (e3, target_poi, target_ligand)

    class OracleDistance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.index = 0

        def forward(self, inputs):
            receiver, anchor = ((2, 0), (2, 1), (1, 0))[self.index % 3]
            self.index += 1
            delta = native[anchor][None] - native[receiver][:, None]
            goal = (delta.square().sum(-1) + .01).sqrt() / 10.
            # The production head applies softplus, so return its inverse.
            return torch.log(torch.expm1(goal))[..., None]

    head.distance_head = OracleDistance()
    proposals = {name: {'tor': torch.empty(1, 0, dtype=dtype)}
                 for name in ('e3_ligand', 'poi_ligand')}
    with torch.no_grad():
        for step in range(20):
            coordinates = torch.cat((e3, state[1][0], state[0][0]))
            graph = {'node_coords': coordinates, 'node_types': types}
            joint = {'points': coordinates, 'latent_types': types,
                     'features': torch.zeros(12, 4, dtype=dtype),
                     'latent_features': torch.zeros(12, 4, dtype=dtype),
                     'relation': torch.zeros(12, 12, 4, dtype=dtype),
                     'geometry': torch.zeros(12, 12, 4, dtype=dtype),
                     'material_weights': [torch.eye(4, dtype=dtype), torch.eye(4, dtype=dtype)]}
            clock = torch.zeros(1, 9, dtype=dtype)
            clock[:, 5:8] = .05
            clock[:, -1] = 1. - step / 20.
            output = head(joint, graph, proposals, torch.zeros(1, 4, dtype=dtype), clock)
            state = advance(state, unpack_velocity(output['normalized_velocity']),
                            torch.tensor(.05, dtype=dtype), torch.ones(2, dtype=dtype), ld)
    for actual, target in ((state[0][0], target_ligand), (state[1][0], target_poi)):
        rmsd = (actual - target).square().sum(-1).mean().sqrt()
        assert rmsd < .01, f'oracle distances could not recover rigid assembly: RMSD={rmsd}'


if __name__ == '__main__':
    tests = [(name, value) for name, value in globals().copy().items() if name.startswith('test_')]
    for name, test in tests:
        test()
        print(f'PASS {name}', flush=True)
    print(f'{len(tests)} control-repair checks passed')
