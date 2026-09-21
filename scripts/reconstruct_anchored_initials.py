"""Recover an existing run's exact initial pool without another model forward.

Require saved final-state reconstruction to verify the native-free reference
bank. Scoring, if desired, is a separate post-hoc command.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prediction', 'runtime-root', 'cache-dir', 'condition-manifest', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--case-dir', type=Path)
    parser.add_argument('--conserved-core-chemistry-report', type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('initial-pool reconstruction must not overwrite an artifact')
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.runtime_root.resolve()))
    sys.path.insert(1, str(args.runtime_root.resolve() / 'scripts'))
    from tricoopdiff.binary_chart_bank import build_binary_chart_bank
    from tricoopdiff.binary_pose_constraints import BinaryPoseInitializer
    from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
    from tricoopdiff.condition_manifest import load_binary_pose_sides, load_condition_manifest
    from tricoopdiff.single_anchor_initialization import initialize_single_anchor

    report = json.loads(args.prediction.read_text())
    if report['condition_manifest_sha256'] != hashlib.sha256(args.condition_manifest.read_bytes()).hexdigest():
        raise ValueError('binary conditioning manifest changed since prediction')
    cache = torch.load(args.cache_dir / f'{report["case"]}.pt', map_location='cpu', weights_only=False)
    e3, poi, ligand = cache['e3_data'], cache['poi_data'], cache['ligand_data']
    if not torch.allclose(e3['coords'], torch.tensor(report['e3_coords']), atol=1e-5, rtol=0):
        raise ValueError('cached E3 frame differs from the saved prediction')
    sides, _ = load_binary_pose_sides(args.condition_manifest, case=report['case'], molecule=Chem.Mol(ligand['mol']),
        protein_sizes=[len(e3['coords']), len(poi['coords'])])
    if report['conditioning_task'] in ('dual_core', 'dual_contact_core'):
        from tricoopdiff.core_conditions import load_core_conditions
        _, sides, audit = load_core_conditions(args.condition_manifest, case=report['case'],
            molecule=Chem.Mol(ligand['mol']), protein_sizes=[len(e3['coords']), len(poi['coords'])],
            all_contact_blocks=report['conditioning_task'] == 'dual_contact_core')
        if audit != report['condition_audit']:
            raise ValueError('core-conditioned deployment evidence changed since prediction')
    core_audit = report.get('condition_audit', {}).get('conserved_core_transfer')
    if core_audit:
        if args.case_dir is None or args.conserved_core_chemistry_report is None:
            raise ValueError('conserved-core reconstruction requires original case directory and chemistry report')
        if hashlib.sha256(args.conserved_core_chemistry_report.read_bytes()).hexdigest() != core_audit['chemistry_report_sha256']:
            raise ValueError('conserved-core chemical source changed since prediction')
        from make_protac_pocket_manifests import receptor_node_coordinates
        from tricoopdiff.conserved_core_conditions import augment_conserved_core_conditions
        features, _ = load_condition_manifest(args.condition_manifest, case=report['case'],
            molecule=Chem.Mol(ligand['mol']), protein_sizes=[len(e3['coords']), len(poi['coords'])])
        _, sides, reproduced_audit = augment_conserved_core_conditions(features, sides, Chem.Mol(ligand['mol']),
            case=report['case'], case_dir=args.case_dir, chemistry_report=args.conserved_core_chemistry_report,
            read_protein_coordinates=receptor_node_coordinates)
        if reproduced_audit != core_audit:
            raise ValueError('conserved-core evidence differs from the saved prediction')
    elif args.conserved_core_chemistry_report is not None:
        raise ValueError('cannot add new core conditions to an existing ordinary prediction')
    rows = report.get('predictions', [report])
    bounds = rows[0]['bounded_binary_audit']['bounds']
    single = report['conditioning_task'] == 'single_anchor'
    if single:
        flow, audit = initialize_single_anchor(sides[0], e3['coords'], poi['coords'], ligand,
            seed=report['seed'], max_candidates=report['initialization']['candidate_budget'],
            translation_bound=bounds['translation'][0], rotation_bound=bounds['rotation'][0])
        bank = [(flow, audit)]
    else:
        if report.get('reference_policy') != 'bank':
            raise ValueError('this reconstruction tool requires the explicit dual reference bank')
        bank, _ = build_binary_chart_bank(BinaryPoseInitializer(*sides,
            max_candidates=report['reference_bank_audit']['candidate_budget']),
            e3['coords'], poi['coords'], ligand, seed=report['seed'])
    predictions, maximum = [], 0.
    for row in rows:
        reference = 0 if single else row['reference_index']
        flow, audit = bank[reference]
        if not single:
            if audit['candidate_index'] != row['reference_audit']['candidate_index']:
                raise ValueError('reference-bank membership/order changed')
            if report['conditioning_task'] == 'dual_assembly':
                from tricoopdiff.assembly_flow import AssemblyFlow
                flow = AssemblyFlow(flow, *audit['selected_atom_maps'])
            else:
                flow = BoundedBinaryTorsionFlow(flow, *audit['selected_atom_maps'],
                    translation_bounds=bounds['translation'], rotation_bounds=bounds['rotation'])
        ref = flow.strict.ligand
        final = flow.decode(torch.tensor(row['angles'], dtype=ref.dtype))
        error = max(float((value - torch.tensor(row[key], dtype=value.dtype)).norm(dim=-1).max())
                    for value, key in zip(final[:2], ('ligand_coords', 'poi_coords')))
        if error > 1e-3:
            raise ValueError(f'cannot reproduce saved final geometry: {error}')
        maximum = max(maximum, error)
        initial = flow.decode(torch.tensor(row['initial_parameters'], dtype=ref.dtype))
        predictions.append({'walker': row['walker'], 'ligand_coords': initial[0].tolist(),
                            'poi_coords': initial[1].tolist(), 'body': initial[2].tolist()})
    result = {key: value for key, value in report.items() if key not in ('predictions', 'ligand_coords', 'poi_coords', 'body')}
    result.update(scope='reconstructed initial pool; not additional trained predictions',
        predictions=predictions, prediction_stage='initial',
        source_prediction_sha256=hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
        saved_final_reconstruction_max_error=maximum, native_score_computed=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(result, handle, allow_nan=False)
    print(json.dumps({'case': report['case'], 'walkers': len(predictions), 'reconstruction_max_error': maximum}), flush=True)


if __name__ == '__main__':
    main()
