"""One-dimensional diagnostic of terminal-label/control-setpoint conflict.

This is a mathematical reduction, not a protein prediction experiment. It uses
historical v39 terminal_distance_rate and finite-horizon Euler timing. A positive
endpoint gradient at a perfectly labelled goal shows an incentive to move the
predicted goal away from its label to compensate for finite-time underreach.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diagnose_control_reachability import terminal_distance_rate


def diagnose():
    rows = []
    for response in (1.0, 0.5, 0.1):
        target = torch.tensor(1., dtype=torch.float64)
        goal = target.clone().requires_grad_()
        distance = torch.tensor(4., dtype=torch.float64)
        for index in range(20):
            time = distance.new_tensor(1. - index / 20.)
            distance = distance - .05 * response * terminal_distance_rate(distance, goal, time)
        endpoint_loss = (distance - target).square()
        label_loss = torch.nn.functional.smooth_l1_loss(torch.log1p(goal), torch.log1p(target))
        endpoint_gradient = torch.autograd.grad(endpoint_loss, goal, retain_graph=True)[0]
        label_gradient = torch.autograd.grad(label_loss, goal)[0]
        assert endpoint_gradient > 0 and label_gradient == 0
        rows.append({'effective_response': response, 'initial_distance': 4., 'native_goal': 1.,
                     'predicted_goal': 1., 'final_distance': distance.item(),
                     'endpoint_gradient_wrt_goal': endpoint_gradient.item(),
                     'label_gradient_wrt_goal': label_gradient.item(),
                     'gradient_descent_pushes_goal_below_native': True})
    return {'scope': 'scalar finite-horizon diagnostic; not full-model causal attribution',
            'distance_units': 'normalized by 10 Angstrom', 'steps': 20, 'total_time': 1.,
            'response_meaning': 'scalar stand-in for Jacobian projection, ridge and repair gate attenuation',
            'results': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/model45_root_cause/terminal_goal_conflict.json')
    args = parser.parse_args()
    result = diagnose()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
