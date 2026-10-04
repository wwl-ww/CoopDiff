"""Positive metric control for bound/native-derived internal panel templates only.

Reconstruct the original input pose and require near-perfect exact DockQ.
This is a label-side diagnostic, never a generator, ranker input or benchmark result.
Do not apply the near-perfect expectation to unbound or independently docked inputs.
"""

import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluation_common import evaluation_input_signature, sha256_file
from tricoopdiff.metrics.dockq import (
    _featurized_ca_coords, _ligand_records, compute_fullatom_dockq,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-refs', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--expect-native-frame', action='store_true', required=True,
                        help='Explicitly attest inputs are bound reference splits, not unbound inputs')
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    if args.out.exists():
        raise FileExistsError('choose a new reference-control report')
    entries = [e for path in args.label_refs for e in json.loads(path.read_text())['entries']]
    if not entries or len({e['case'] for e in entries}) != len(entries):
        raise ValueError('reference panel must have unique, nonempty cases')
    rows = []
    for entry in entries:
        row = {'case': entry['case']}
        try:
            row['input_signature'] = evaluation_input_signature(
                {k: entry[k] for k in ('protein1', 'protein2', 'ligand')}, entry['native_complex_pdb'])
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                metrics = compute_fullatom_dockq(
                    _ligand_records(entry['ligand'])[2], _featurized_ca_coords(entry['protein2']),
                    _featurized_ca_coords(entry['protein1']), entry['protein1'], entry['protein2'],
                    entry['ligand'], entry['native_complex_pdb'], project_root / 'DeepTernary')
            if not all(math.isfinite(metrics[k]) for k in ('dockq', 'fnat', 'irms', 'lrms')):
                raise ValueError('nonfinite reference metric')
            row.update(metrics=metrics, passed=metrics['dockq'] >= .999)
        except Exception as error:
            row.update(passed=False, error=f'{type(error).__name__}: {error}')
        rows.append(row)
        print(f"CONTROL {entry['case']}: {row['passed']}", flush=True)
    passed = sum(row['passed'] for row in rows)
    payload = {'protocol': 'bound_reference_metric_control_v1', 'cases': len(rows), 'passed': passed,
               'scope': 'native-frame positive control only; not generated candidates or ranker labels',
               'references_sha256': {str(path): sha256_file(path) for path in args.label_refs},
               'source_sha256': {str(path): sha256_file(path) for path in [Path(__file__),
                   ROOT / 'tricoopdiff/metrics/dockq.py'] + sorted(
                       (project_root / 'DeepTernary/DockQ').rglob('*.py'))}, 'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return int(passed != len(rows))


if __name__ == '__main__':
    sys.exit(main())
