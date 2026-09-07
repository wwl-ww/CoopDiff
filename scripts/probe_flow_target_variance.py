"""Measure empirical target variance at the identical clean endpoint input.

This is an endpoint-only experiment. Continuous random training times hit t=0
with probability zero, so this is NOT a lower bound on the full training loss.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import tricoopdiff.diffdock_compat  # noqa: F401,E402
from tricoopdiff.sdpi_flow import SDPIFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--samples', type=int, default=128)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('samples must be at least two')
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(destination)
    probes = torch.load(args.probes, map_location='cpu', weights_only=False)
    batches = {batch['case']: batch for batch, _, _ in probes}
    torch.manual_seed(123)
    np.random.seed(123)
    flow = SDPIFlow()
    time = torch.zeros(1)
    rows = []
    for case, batch in batches.items():
        reference = None
        targets = {}
        scales = flow.scales(batch['e3_data']['coords'], batch['poi_data']['coords'])
        for _ in range(args.samples):
            coupling = flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
            ligand, poi, body, target = flow.path(
                batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
            visible = (ligand, poi, body)
            if reference is None:
                reference = visible
            if not all(torch.equal(a, b) for a, b in zip(visible, reference)):
                raise AssertionError('endpoint input changed across random couplings')
            values = {'torsion': target['torsion'] / np.pi}
            for i, branch in enumerate(('e3_ligand', 'poi_ligand')):
                values[f'{branch}_tr'] = target[branch]['tr'] / scales[i]
                values[f'{branch}_rot'] = target[branch]['rot'] / np.pi
            for name, value in values.items():
                targets.setdefault(name, []).append(value.double())
        variances = {name: (torch.stack(values).var(dim=0, unbiased=False).mean().item()
                            if values[0].numel() else 0.)
                     for name, values in targets.items()}
        # Both auxiliary torsion targets are identical to the joint target;
        # the production total weights them 0.125 each.
        weighted = sum(v for k, v in variances.items() if k != 'torsion') + 1.25 * variances['torsion']
        rows.append({'case': case, 'samples': args.samples, 'identical_endpoint_inputs': True,
                     'normalized_target_variance': variances,
                     'empirical_endpoint_minimum_mse': weighted})
        print(case, 'endpoint empirical target variance:', weighted, flush=True)
    destination.write_text(json.dumps({
        'scope': 't=0 identical inputs only; NOT a global training-loss floor',
        'seed': 123, 'cases': rows,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
