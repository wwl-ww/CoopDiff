"""Check native-free geometric feasibility of additional binary core hypotheses.

Existing trusted sides are preserved. Missing sides can be proposed from exact
or explicitly tautomer-normalized conserved chemistry. Outputs remain audit-only
and never overwrite production manifests or use ternary coordinates for fits.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import tricoopdiff.diffdock_compat  # noqa: F401
from featurize_common import featurize_complex
from train_entry import _load_score_model_args
from make_protac_pocket_manifests import receptor_node_coordinates
from tricoopdiff.condition_manifest import load_binary_pose_sides, load_condition_manifest
from tricoopdiff.binary_pose_constraints import BinaryPoseInitializer
from tricoopdiff.binary_chart_bank import build_binary_chart_bank
from tricoopdiff.conserved_core_conditions import augment_conserved_core_conditions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('chemistry-report', 'case-root', 'manifest-dir', 'cache-dir', 'expert-model-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--candidates', type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.seed < 0 or args.candidates < 1 or len(set(args.cases)) != len(args.cases):
        parser.error('require nonnegative seed, positive budget and distinct cases')
    torch.set_num_threads(1)
    records = {r['case']: r for r in json.loads(args.chemistry_report.read_text())['cases']}
    if any(case not in records or case != Path(case).name for case in args.cases):
        raise ValueError('unknown or unsafe case identifier')
    args.out.mkdir(parents=True)
    score_args = _load_score_model_args(str(args.expert_model_dir))
    for case in args.cases:
        directory = args.case_root / case
        full_path = directory / 'ligand.sdf'
        if sha(full_path) != records[case]['full_ligand_sha256']:
            raise ValueError('full graph source hash mismatch')
        e3, poi, ligand, _ = featurize_complex(case,
            str((directory / 'unbound_protein1.pdb').resolve()),
            str((directory / 'unbound_protein2.pdb').resolve()), str(full_path.resolve()),
            score_args, 'cpu', cache_dir=str(args.cache_dir), cache_read_only=True, cache_wait_seconds=0)
        full = Chem.Mol(ligand['mol'])
        full.RemoveAllConformers()
        sides, _ = load_binary_pose_sides(args.manifest_dir / f'{case}.json', case=case,
            molecule=full, protein_sizes=[e3['coords'].shape[1], poi['coords'].shape[1]])
        features, _ = load_condition_manifest(args.manifest_dir / f'{case}.json', case=case,
            molecule=full, protein_sizes=[e3['coords'].shape[1], poi['coords'].shape[1]])
        # Use exactly the inference constructor, including cleanup of cached
        # nonstereogenic tags before atom-identity labels are added.
        _, sides, evidence = augment_conserved_core_conditions(features, sides, full,
            case=case, case_dir=directory, chemistry_report=args.chemistry_report,
            read_protein_coordinates=receptor_node_coordinates)
        report = {'case': case, 'audit_only': True, 'deployment_eligible': False,
                  'seed': args.seed, 'candidate_budget': args.candidates, 'evidence': evidence,
                  'native_ternary_coordinates_used': False, 'conditioning_sides': [s is not None for s in sides]}
        if all(s is not None for s in sides):
            try:
                charts, audit = build_binary_chart_bank(BinaryPoseInitializer(*sides, max_candidates=args.candidates),
                    e3['coords'][0], poi['coords'][0], ligand, seed=args.seed)
                report.update(status='geometrically_admissible', reference_bank=audit,
                              active_torsions=[len(flow.plan.bonds) for flow, _ in charts])
            except ValueError as error:
                report.update(status='geometry_rejected', error=str(error))
        else:
            report['status'] = 'still_single_anchor'
        with (args.out / f'{case}.json').open('x') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({k: report[k] for k in ('case', 'status', 'conditioning_sides')}), flush=True)


if __name__ == '__main__':
    main()
