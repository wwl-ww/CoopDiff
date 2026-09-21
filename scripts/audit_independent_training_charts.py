"""Audit deployment-built charts against native supervision, without training.

Generic training complexes supply synthetic protein-local binary evidence, not
independent experimental binaries. ETKDG references use the deployment builder;
native cross-protein placement is used only after chart construction for audit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: F401
from evaluation_runner import EvaluationRunner
from featurize_common import featurize_complex
from tricoopdiff.anchored_fragments import select_training_fragments
from tricoopdiff.binary_pose_constraints import BinaryPoseSide
from tricoopdiff.independent_training_reference import build_independent_training_reference_bank
from tricoopdiff.chart_target_projection import project_chart_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('index', 'cache-dir', 'project-root', 'parent', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--binary-candidates', type=int, default=24)
    parser.add_argument('--fit-iterations', type=int, default=100)
    parser.add_argument('--fit-restarts', type=int, default=8)
    parser.add_argument('--all-references', action='store_true',
                        help='Project every valid chart; selection remains native-blind')
    parser.add_argument('--assembly-flow', action='store_true',
                        help='Use existing unbounded AssemblyFlow for reachability diagnostics only')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = json.loads(args.index.read_text())
    excluded = set(payload.get('metadata', {}).get('excluded_benchmark_pdb_ids', []))
    entries = {row['case']: row for row in payload['entries']}
    cases = args.cases or sorted(entries)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error('max-cases must be positive')
        cases = cases[:args.max_cases]
    if (not excluded or len(set(cases)) != len(cases)
            or any(case not in entries or case.split('_')[0].upper() in excluded for case in cases)):
        raise ValueError('require distinct benchmark-excluded indexed training cases')
    runner = EvaluationRunner(args.project_root, args.parent.resolve(), device='cpu',
                              allow_conditioned_checkpoint=True)
    report = dict(scope='training_input_audit_not_benchmark', seed=args.seed,
                  index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(),
                  binary_evidence='synthetic_native_local_pairs_not_independent_experimental_binaries',
                  native_cross_protein_placement_used_for='supervision_and_residual_audit_only',
                  binary_candidates=args.binary_candidates, fit_iterations=args.fit_iterations,
                  fit_restarts=args.fit_restarts, cases=[])
    for case in cases:
        entry = entries[case]
        e3, poi, ligand, _ = featurize_complex(case, entry['protein1'], entry['protein2'], entry['ligand'],
            runner.score_model_args, 'cpu', cache_dir=str(args.cache_dir), cache_read_only=True, cache_wait_seconds=0)
        generator = torch.Generator().manual_seed(args.seed)
        selection = select_training_fragments(ligand['mol'], ligand['edge_index'], ligand['edge_mask'],
            ligand['coords'][0], (e3['coords'][0], poi['coords'][0]), generator=generator)
        row = dict(case=case)
        if selection is None:
            row.update(status='rejected', reason='no_chemical_cut_pair')
        else:
            sides = []
            for protein, mapping in zip((e3['coords'][0], poi['coords'][0]), selection['atom_maps']):
                center = protein.mean(0)
                sides.append(BinaryPoseSide(protein - center,
                                            ligand['coords'][0, list(mapping)] - center, (mapping,)))
            try:
                bank, bank_audit = build_independent_training_reference_bank(
                    sides, e3['coords'][0], poi['coords'][0], ligand, seed=args.seed,
                    binary_bounds={'translation': [.5, .5], 'rotation': [.2, .2]},
                    evidence_provenance='synthetic_native_local_pairs', max_candidates=args.binary_candidates)
            except ValueError as error:
                row.update(status='rejected', reason=str(error))
            else:
                if args.assembly_flow:
                    from tricoopdiff.assembly_flow import AssemblyFlow
                    bank = [(AssemblyFlow(flow.strict, flow.end1, flow.end2), selected)
                            for flow, selected in bank]
                rows = []
                for index, (flow, selected) in enumerate(bank if args.all_references else bank[:1]):
                    try:
                        _, exact = flow.encode_target(ligand['coords'][0], poi['coords'][0], ligand['edge_index'])
                    except ValueError as error:
                        exact = {'reachable': False, 'reason': str(error)}
                    else:
                        exact = {'reachable': True, **exact}
                    fit = project_chart_target(flow, ligand['coords'][0], poi['coords'][0], ligand['edge_index'],
                                               max_iterations=args.fit_iterations, restarts=args.fit_restarts,
                                               seed=int.from_bytes(hashlib.sha256(
                                                   f'{args.seed}:projection:{case}:{index}'.encode()).digest()[:8], 'big'))
                    rows.append({'reference_index': index, 'selected': selected,
                                 'exact_target': exact, 'projection': fit.audit})
                selected_row = min(rows, key=lambda row: row['projection']['final_equal_body_mse'])
                row.update(status='audited', reference_index=selected_row['reference_index'], bank=bank_audit,
                           fragment_augmentation=selection['audit'], reference_count=len(bank),
                           all_reference_audits=rows if args.all_references else None,
                           exact_target=selected_row['exact_target'], projection=selected_row['projection'])
        report['cases'].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    audited = [row for row in report['cases'] if row['status'] == 'audited']
    report['summary'] = {
        'requested': len(cases), 'audited': len(audited),
        'rejected': len(cases) - len(audited),
        'exactly_reachable': sum(row['exact_target']['reachable'] for row in audited),
        'all_reference_diagnostic': bool(args.all_references),
        'projection_residuals': {name: sorted(row['projection'][name] for row in audited)
                                 for name in ('ligand_rmsd', 'poi_rmsd', 'ligand_max_error', 'poi_max_error')},
        'acceptance_threshold': None,
        'acceptance_threshold_status': 'not_frozen_do_not_train',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
