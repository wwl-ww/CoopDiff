"""Sample complete-binary structures from one root RNG, without native scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import tricoopdiff.diffdock_compat  # noqa: F401
from featurize_common import featurize_complex
from train_entry import _load_score_model_args
from tricoopdiff.anchored_consensus import sample_anchored
from tricoopdiff.anchored_flow import AnchoredTorsionFlow
from tricoopdiff.anchored_trainer import build_anchored_model, validate_anchored_checkpoint
from tricoopdiff.bianchor_source import anchored_source
from tricoopdiff.binary_pose_constraints import BinaryPoseInitializer
from tricoopdiff.binary_chart_bank import build_binary_chart_bank, reference_conformer_schedule
from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow, ball_map
from tricoopdiff.assembly_flow import AssemblyFlow
from tricoopdiff.single_anchor_initialization import initialize_single_anchor
from tricoopdiff.chemical_prior_contract import deployment_conformers, conditioned_chemical_prior
from tricoopdiff.condition_manifest import load_binary_pose_sides, load_condition_manifest
from tricoopdiff.deployment_condition_guard import require_deployable_condition_manifest
from tricoopdiff.expert_factory import build_diffdock_expert
from tricoopdiff.reproducibility import configure_inference_reproducibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'expert-model-dir', 'case-dir', 'condition-manifest', 'cache-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=7315)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--n-walkers', type=int, default=1)
    parser.add_argument('--binary-candidates', type=int, default=24)
    parser.add_argument('--max-fit-rmsd', type=float, default=.15)
    parser.add_argument('--reference-policy', choices=('bank', 'selected'),
                        help='Chemical dual priors default to a validated reference bank; legacy checkpoints stay selected')
    parser.add_argument('--allow-smoke', action='store_true', help='Explicit implementation check, not accuracy evaluation')
    parser.add_argument('--conserved-core-chemistry-report', type=Path,
                        help='Explicit experimental conserved-core transfer for missing binary sides')
    parser.add_argument('--stratified-mixture', action='store_true',
                        help='Stratify mixed-prior components within the declared walker budget')
    parser.add_argument('--allow-chemical-repeats', action='store_true',
                        help='Explicit shuffled chemical cycles; reuse identical deterministic predictions')
    args = parser.parse_args()
    if args.allow_chemical_repeats and not args.stratified_mixture:
        parser.error('chemical repeats require the explicit stratified mixture protocol')
    if args.out.exists():
        raise FileExistsError('choose a new prediction output')
    if args.seed < 0 or min(args.steps, args.binary_candidates, args.n_walkers) < 1:
        parser.error('seed must be nonnegative and steps/candidates positive')
    require_deployable_condition_manifest(args.condition_manifest)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    validate_anchored_checkpoint(payload)
    if payload.get('conditioning_task') in ('dual_core', 'dual_contact_core') and args.conserved_core_chemistry_report:
        raise ValueError('dual_core currently requires validated manifest sides; analogue augmentation is not yet supported')
    if payload['scope'] == 'implementation_smoke' and not args.allow_smoke:
        raise ValueError('smoke weights require --allow-smoke; they are not accuracy evidence')
    reproducibility = configure_inference_reproducibility(True)
    environment = payload['expert_environment']
    if environment['activation_clamp'] is None:
        os.environ.pop('NAN_CLAMP', None)
    else:
        os.environ['NAN_CLAMP'] = str(environment['activation_clamp'])
    os.environ['NAN_CLAMP_MODE'] = environment['activation_clamp_mode']
    expert = build_diffdock_expert(str(args.expert_model_dir), device=args.device)
    model = build_anchored_model(payload, expert, allow_smoke=args.allow_smoke).to(args.device).eval()
    case = args.case_dir.name
    e3, poi, ligand, dim = featurize_complex(
        case, str((args.case_dir / 'unbound_protein1.pdb').resolve()),
        str((args.case_dir / 'unbound_protein2.pdb').resolve()),
        str((args.case_dir / 'ligand.sdf').resolve()),
        _load_score_model_args(str(args.expert_model_dir)), args.device,
        cache_dir=str(args.cache_dir), cache_read_only=True, cache_wait_seconds=0)
    if dim != payload['model_config']['input_dim']:
        raise ValueError('checkpoint and receptor feature dimensions disagree')
    options = dict(case=case, molecule=Chem.Mol(ligand['mol']),
                   protein_sizes=[e3['coords'].shape[1], poi['coords'].shape[1]])
    conditions, condition_audit = load_condition_manifest(args.condition_manifest, **options)
    sides, _ = load_binary_pose_sides(args.condition_manifest, **options)
    if payload.get('conditioning_task') in ('dual_core', 'dual_contact_core'):
        from tricoopdiff.core_conditions import load_core_conditions
        conditions, sides, condition_audit = load_core_conditions(args.condition_manifest, **options,
            all_contact_blocks=payload['conditioning_task'] == 'dual_contact_core')
    if args.conserved_core_chemistry_report:
        from make_protac_pocket_manifests import receptor_node_coordinates
        from tricoopdiff.conserved_core_conditions import augment_conserved_core_conditions
        conditions, sides, core_audit = augment_conserved_core_conditions(
            conditions, sides, options['molecule'], case=case, case_dir=args.case_dir,
            chemistry_report=args.conserved_core_chemistry_report,
            read_protein_coordinates=receptor_node_coordinates)
        condition_audit = {**condition_audit, 'conserved_core_transfer': core_audit,
                           'available_sides': core_audit['available_sides']}
    bounds = payload.get('binary_bounds')
    assembly = payload.get('conditioning_task') == 'dual_assembly'
    single = payload.get('conditioning_task', 'dual_binary') == 'single_anchor'
    prior_config = payload.get('prior_config')
    reference_policy = args.reference_policy or ('bank' if prior_config and not single else 'selected')
    if reference_policy == 'bank' and (single or prior_config is None):
        raise ValueError('reference-bank sampling currently requires a dual chemical-prior checkpoint')
    reference_bank, bank_audit = None, None
    if single:
        if sides[0] is None or sides[1] is not None:
            raise ValueError('single-anchor checkpoint requires exactly the first binary side; no silent side swap')
        flow, initialization = initialize_single_anchor(sides[0], e3['coords'][0], poi['coords'][0], ligand,
            seed=args.seed, max_candidates=args.binary_candidates, max_fit_rmsd=args.max_fit_rmsd,
            translation_bound=bounds['translation'][0], rotation_bound=bounds['rotation'][0])
        strict = flow.strict
    elif assembly:
        # AssemblyFlow deployments do not require two binary pose sides.  Reuse
        # the audited training initializer so missing binary manifests are not
        # misrouted into BinaryPoseInitializer.
        from tricoopdiff.assembly_flow import prepare_assembly_sample
        assembly_sample = prepare_assembly_sample(
            e3, poi, ligand, generator=torch.Generator(device=args.device).manual_seed(args.seed),
            evidence_id=f'deployment:{case}')
        initialization = {'selected_atom_maps': assembly_sample.audit['selected_atom_maps']}
        flow = assembly_sample.flow
        strict = flow.strict
    else:
        initializer = BinaryPoseInitializer(*sides, max_candidates=args.binary_candidates)
        if reference_policy == 'bank':
            reference_bank, bank_audit = build_binary_chart_bank(initializer,
                e3['coords'][0], poi['coords'][0], ligand, seed=args.seed, max_fit_rmsd=args.max_fit_rmsd)
            flow, initialization = reference_bank[0]
        else:
            flow, initialization = AnchoredTorsionFlow.from_binary_initializer(initializer,
                e3['coords'][0], poi['coords'][0], ligand, seed=args.seed, max_fit_rmsd=args.max_fit_rmsd)
        strict = flow
        if bounds:
            flow = BoundedBinaryTorsionFlow(strict, *initialization['selected_atom_maps'],
                                           translation_bounds=bounds['translation'], rotation_bounds=bounds['rotation'])
        elif assembly:
            flow = AssemblyFlow(strict, *initialization['selected_atom_maps'])
    conformers, conformer_audit = (deployment_conformers(ligand['mol'], prior_config)
                                  if prior_config else (None, None))
    left, right = initialization['selected_atom_maps']
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.stratified_mixture:
        from tricoopdiff.stratified_anchored_prior import (stratified_reference_schedule,
            sample_stratified_component, exact_prediction_key, STRATIFIED_POLICY, CYCLIC_POLICY)
        if not reference_bank or not prior_config.get('uniform_torsion_probability', 0.):
            raise ValueError('stratified mixture requires a dual reference bank and mixed-prior checkpoint')
        schedule = stratified_reference_schedule(len(reference_bank), len(conformers), args.n_walkers,
            prior_config['uniform_torsion_probability'], generator=generator,
            allow_repeats=args.allow_chemical_repeats)
    else:
        schedule = (reference_conformer_schedule(len(reference_bank), len(conformers), args.n_walkers,
                                                 generator=generator) if reference_bank else None)
    predictions = []
    prediction_cache = {}
    for walker in range(args.n_walkers):
        reference_index, conformer_index = None, None
        component = None
        if reference_bank:
            if args.stratified_mixture:
                reference_index, conformer_index, component = schedule[walker]
            else:
                reference_index, conformer_index = schedule[walker]
            strict, initialization = reference_bank[reference_index]
            left, right = initialization['selected_atom_maps']
            flow = (BoundedBinaryTorsionFlow(strict, left, right,
                    translation_bounds=bounds['translation'], rotation_bounds=bounds['rotation']) if bounds else strict)
            if assembly:
                flow = AssemblyFlow(strict, left, right)
        if prior_config:
            pool = conformers if conformer_index is None else conformers[conformer_index:conformer_index + 1]
            if args.stratified_mixture:
                prior, prior_audit = sample_stratified_component(flow, ligand['edge_index'], pool, component,
                                                                generator=generator, config=prior_config)
            else:
                prior, prior_audit = conditioned_chemical_prior(flow, ligand['edge_index'], pool,
                                                               generator=generator, config=prior_config)
            if conformer_index is not None and prior_audit.get('component') != 'uniform':
                prior_audit.update(conformer_index=conformer_index, conformer_count=len(conformers))
        else:
            prior, prior_audit = flow.sample_prior(generator=generator), None
        cache_key = exact_prediction_key(reference_index, prior) if args.allow_chemical_repeats else None
        reused_from = None
        if cache_key is not None and cache_key in prediction_cache:
            reused_from, result = prediction_cache[cache_key]
        else:
            result = sample_anchored(model, flow, prior, e3, poi, ligand,
                                     steps=args.steps, conditions=conditions, source=anchored_source(model))
            if cache_key is not None:
                prediction_cache[cache_key] = walker, result
        if args.allow_chemical_repeats:
            prior_audit['sampling_design'] = CYCLIC_POLICY
        anchor_error = float((result['ligand_coords'][left] - strict.ligand[left]).norm(dim=-1).max())
        binary_error = (None if single else float((torch.cdist(result['ligand_coords'][right], result['poi_coords'])
                             - torch.cdist(strict.ligand[right], strict.poi)).abs().max()))
        if not bounds and not assembly and (anchor_error > 1e-4 or binary_error > 1e-3):
            raise ValueError('sample violated its initial binary geometry')
        bounded_audit = None
        if assembly:
            _, reconstruction = flow.encode_target(result['ligand_coords'], result['poi_coords'], ligand['edge_index'])
            bounded_audit = {'bounds': None, 'policy': flow.version, 'reconstruction': reconstruction,
                             'local_binary_pose_is_soft_condition': True}
        if bounds:
            _, reconstruction = flow.encode_target(result['ligand_coords'], result['poi_coords'], ligand['edge_index'])
            side_count = 1 if single else 2
            offsets = result['angles'][flow.torsion_count:flow.torsion_count + 6 * side_count].reshape(side_count, 2, 3)
            bounded_audit = {'bounds': bounds, 'reconstruction': reconstruction,
                             'translation_norms': [float(ball_map(offsets[i, 0], bounds['translation'][i]).norm())
                                                   for i in range(side_count)],
                             'rotation_norms_radians': [float(ball_map(offsets[i, 1], bounds['rotation'][i]).norm())
                                                       for i in range(side_count)]}
        predictions.append({'walker': walker, 'initial_parameters': prior.detach().cpu().tolist(),
                            'reused_prediction_from_walker': reused_from,
                            'prior_audit': prior_audit,
                            'reference_index': reference_index, 'reference_audit': initialization if reference_bank else None,
                            'bounded_binary_audit': bounded_audit,
                            'anchor_max_error': anchor_error, 'binary_distance_max_error': binary_error,
                            **{key: value.detach().cpu().tolist() for key, value in result.items()
                               if torch.is_tensor(value)}})
        print(json.dumps({'case': case, 'walker': walker, 'completed': walker + 1,
                          'total': args.n_walkers}), flush=True)
    report = {'case': case, 'seed': args.seed, 'steps': args.steps, 'n_walkers': args.n_walkers,
              'model_trajectories_computed': sum(row['reused_prediction_from_walker'] is None for row in predictions),
              'ensemble_policy': (CYCLIC_POLICY if args.allow_chemical_repeats else
                                  STRATIFIED_POLICY if args.stratified_mixture else
                                  'validated_reference_bank_mixed_torsion_priors_v1'
                                  if reference_bank and prior_config.get('uniform_torsion_probability', 0.) else
                                  'validated_reference_bank_distinct_chemical_pairs_v1' if reference_bank else
                                  'one_shared_binary_reference_independent_parameter_priors_v1'),
              'reference_policy': reference_policy, 'reference_bank_audit': bank_audit,
              'scope': 'unscored structure prediction', 'checkpoint_scope': payload['scope'],
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'condition_manifest_sha256': hashlib.sha256(args.condition_manifest.read_bytes()).hexdigest(),
              'generative_process': flow.version, 'source_policy': payload['source_policy'],
              'initialization_source': anchored_source(model),
              'action_policy': payload['action_policy'],
              'action_adapter': payload.get('action_adapter'),
              'conditioning_task': payload.get('conditioning_task', 'dual_binary'),
              'prior_config': prior_config, 'chemical_conformer_audit': conformer_audit,
              'reproducibility': reproducibility, 'condition_audit': condition_audit,
              'initialization': bank_audit if reference_bank else initialization, 'native_score_computed': False,
              'coordinate_frame': 'featurized E3 frame; E3 coordinates included',
              'e3_coords': e3['coords'][0].detach().cpu().tolist(),
              **(predictions[0] if args.n_walkers == 1 else {'predictions': predictions})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({'case': case, 'scope': payload['scope'], 'steps': args.steps,
                      'n_walkers': args.n_walkers,
                      'max_anchor_displacement': max(row['anchor_max_error'] for row in predictions),
                      'max_binary_distance_change': (None if single else
                          max(row['binary_distance_max_error'] for row in predictions))}), flush=True)


if __name__ == '__main__':
    main()
