"""Check reverse-integration tangents on a saved real diagnostic pair."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import tricoopdiff.diffdock_compat  # noqa: F401,E402
from tricoopdiff.sdpi_flow import SDPIFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', required=True, help='Trusted local fixed_pair.pt')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    saved = torch.load(args.pair, map_location='cpu', weights_only=False)
    batch, coupling = saved['batch'], saved['coupling']
    ligand = {**batch['ligand_data'], 'coords': batch['ligand_data']['coords'].double()}
    poi = batch['poi_data']['coords'].double()
    flow = SDPIFlow()
    rows = []
    for value in (.1, .5, .9):
        time = torch.tensor([value], dtype=torch.float64)
        lig, rec, body, target = flow.path(ligand, poi, coupling, time)
        errors = []
        for h in (1e-3, 5e-4):
            dt = time.new_tensor([h])
            actual = flow.step(lig, rec, body, target, dt, ligand)
            expected = flow.path(ligand, poi, coupling, time - dt)
            errors.append(max((actual[i] - expected[i]).abs().max().item()
                              for i in (0, 1)))
        passed = errors[-1] < 1e-3 and errors[-1] < .4 * errors[0] + 1e-9
        rows.append({'time': value, 'cartesian_max_errors_angstrom': errors,
                     'passed': passed})
    report = {'case': batch['case'], 'times': rows,
              'passed': all(row['passed'] for row in rows),
              'scope': 'real-pair joint Cartesian tangent including POI frame and torsion gauge'}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('actual-pair reverse-integration tangent gate failed')


if __name__ == '__main__':
    main()
