#!/usr/bin/env python
"""Read-only artifact loading and fixed-case full-atom DockQ evaluation.

Only writes beneath results/local_exact_gate. Artifact paths are CLI inputs;
legacy /tmp/coopdiff paths are relocated in memory, never on disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from contextlib import nullcontext

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '4')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    """Keep evaluator integrity gates active under ``python -O``."""
    if not condition:
        raise ValueError(message)


def relocate(path, root):
    legacy = Path('/tmp/coopdiff')
    path = Path(path)
    return root / path.relative_to(legacy) if path.is_relative_to(legacy) else path


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--deepternary', type=Path, required=True)
    parser.add_argument('--support-root', type=Path, required=True,
                        help='Original checkout containing torch_cluster/scatter stubs')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--expert-dir', type=Path)
    parser.add_argument('--cases', nargs='+', default=[
        '3CXH_C_E_6PH', '6U9D_Q_R_TPP', '4I0P_D_H_PGE'])
    parser.add_argument('--seed', type=int, default=12001)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable strict reproducibility before constructing CUDA models')
    parser.add_argument('--weights', choices=('ema', 'raw'), default='ema')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/local_exact_gate')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT / 'results/local_exact_gate'):
        raise ValueError('Output must remain under owned results/local_exact_gate')
    if len(set(args.cases)) < 3:
        raise ValueError('Gate requires at least three distinct fixed cases')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'report.json').exists():
        raise FileExistsError('Preserve previous report; choose a fresh --out subdirectory')
    os.environ['TMPDIR'] = str(args.out / 'tmp')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    os.environ['HF_HUB_CACHE'] = str(args.support_root / 'hf_cache')
    os.environ.setdefault('TRICOOPDIFF_CACHE_DIR', str(
        args.support_root / 'results/runtime_cache/diffdock'))
    cache_root = Path(os.environ['TRICOOPDIFF_CACHE_DIR'])
    for name in ('.p.npy', '.score.npy', '.so3_omegas_array4.npy',
                 '.so3_cdf_vals4.npy', '.so3_score_norms4.npy',
                 '.so3_exp_score_norms4.npy'):
        if not (cache_root / name).is_file():
            raise FileNotFoundError(cache_root / name)
    sys.path.insert(1, str(args.support_root))
    import numpy as np
    import torch
    import yaml
    from rdkit import Chem
    import tricoopdiff.diffdock_compat  # must precede vendored model imports
    from tricoopdiff.consensus import ConsensusFuse
    from tricoopdiff.checkpoint_contract import validate_expert_base_identity
    from tricoopdiff.consensus_experiment import (
        EndpointTranslationRolloutTrainer, ExperimentalConsensus,
        ExperimentalConsensusTrainer, RolloutAugmentedTrainer)
    from tricoopdiff.deepternary_metrics import (
        _records, _featurized_ca_coords, _fit_rigid, _replace_coords)
    from tricoopdiff.expert_factory import build_diffdock_expert
    from tricoopdiff.initialization import TernaryInitializer
    from tricoopdiff.sampler import sample_single
    from tricoopdiff.sdpi_flow import SDPIFlow
    from tricoopdiff.reproducibility import configure_inference_reproducibility
    sys.path.insert(1, str(args.deepternary.resolve()))
    from DockQ.dockq_util import cal_dockq
    import DockQ.dockq_util as dockq_impl

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    numerical_policy = configure_inference_reproducibility(args.deterministic)
    require(torch.cuda.device_count() == 1, 'exact evaluator requires one visible GPU')
    require(Path(tricoopdiff.diffdock_compat.__file__).is_relative_to(ROOT),
            'DiffDock compatibility module resolved outside this worktree')
    plan = args.artifacts / 'results/overnight_root_cause_v3_plan'
    checkpoint = args.checkpoint or (args.artifacts / 'results/'
        'overnight_root_cause_v3_run04/ccd_single_bond_frozen_seed1729/best.pt')
    index = args.index or plan / 'indexes/ternary_val_ccd.json'
    feature_cache = args.feature_cache or plan / 'cache_ccd'
    expert_dir = args.expert_dir or args.artifacts / 'checkpoints/score_model'
    score_args = yaml.full_load((expert_dir / 'model_parameters.yml').read_text())
    entries = {e['case']: e for e in json.loads(index.read_text())['entries']}
    report = dict(status='running', cases=args.cases, seed=args.seed, steps=args.steps,
                  model='ExperimentalConsensus', weights=args.weights, training_performed=False,
                  reproducibility=numerical_policy,
                  metric='DeepTernary/DockQ/dockq_util.py::cal_dockq',
                  metric_note='Full-atom contacts; upstream backbone iRMS/LRMS convention',
                  checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  index=str(index), index_sha256=sha256(index),
                  dockq_source_sha256=sha256(dockq_impl.__file__),
                  fnat_sha256=sha256(args.deepternary / 'DockQ/fnat'),
                  command=sys.argv, script_sha256=sha256(__file__),
                  environment={k: os.environ[k] for k in (
                      'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE',
                      'TRICOOPDIFF_CACHE_DIR')}, gpu=torch.cuda.get_device_name(0),
                  native_checks=[], baseline=[])
    save_json(args.out / 'report.json', report)

    def score(predicted, native):
        values = list(map(float, cal_dockq(str(predicted), str(native))))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite exact DockQ output')
        return dict(zip(('fnat', 'irms', 'lrms', 'dockq'), values))

    def atom_keys(lines):
        return [(l[:6], l[12:27], l[76:78]) for l in lines
                if l.startswith(('ATOM', 'HETATM'))]

    def write_pdb(path, lines):
        path.write_text('\n'.join(lines) + '\nEND\n')

    def rebuild(info, ligand, poi, path):
        # Undo the cached shared E3 frame, then lift POI's rigid motion to all atoms.
        rotation, translation = info['frame']
        poi_pdb = poi @ rotation + translation
        pr, pt = _fit_rigid(info['p2_ca'], poi_pdb)
        error = float(np.max(np.abs(info['p2_ca'] @ pr + pt - poi_pdb)))
        if error > 0.002:
            raise ValueError(f'POI is not rigid to tolerance: {error}')
        p2 = _replace_coords(info['p2'], list(range(len(info['p2']))),
                             info['p2_xyz'] @ pr + pt)
        lig = _replace_coords(info['lig'], list(range(len(info['lig']))),
                             ligand @ rotation + translation, chain=info['chain'])
        lines = info['p1'] + p2 + lig
        if atom_keys(lines) != atom_keys(info['native_lines']):
            raise ValueError('Full-atom identities/order changed during writeback')
        write_pdb(path, lines)
        return error

    batches, infos = {}, {}
    try:
        for case in args.cases:
            folder = args.out / case
            folder.mkdir(exist_ok=True)
            original = entries[case]
            entry = {k: relocate(original[k], args.artifacts)
                     for k in ('protein1', 'protein2', 'ligand')}
            cache_path = feature_cache / f'{case}.pt'
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
            require(cached['cache_version'] == 7, 'Only corrected graph cache v7 allowed')
            # Recompute the original fingerprint with original path spelling and
            # current file bytes: relocating the path must not conceal stale data.
            digest = hashlib.sha256()
            for key in ('protein1', 'protein2', 'ligand'):
                digest.update(original[key].encode())
                digest.update(entry[key].read_bytes())
            digest.update(json.dumps(score_args, sort_keys=True, default=str).encode())
            require(digest.hexdigest() == cached['source_signature'],
                    'Source/cache fingerprint mismatch')
            batch = {}
            for key in ('e3_data', 'poi_data', 'ligand_data'):
                batch[key] = {k: (v.to('cuda').unsqueeze(0) if k in ('coords', 'features')
                                  else v.to('cuda')) if torch.is_tensor(v) else v
                              for k, v in cached[key].items()}
            mol = Chem.SDMolSupplier(str(entry['ligand']), removeHs=True)[0]
            require(mol is not None, 'cached ligand molecule is missing')
            lig = batch['ligand_data']
            require([a.GetSymbol() for a in mol.GetAtoms()] == [
                a.GetSymbol() for a in Chem.Mol(lig['mol']).GetAtoms()],
                'Cached ligand atom order differs')
            edges = lig['edge_index'].T[lig['edge_mask']].cpu().tolist()
            keep = np.array([mol.GetBondBetweenAtoms(u, v).GetBondType() ==
                             Chem.BondType.SINGLE for u, v in edges], dtype=bool)
            selected = lig['edge_mask'].nonzero().flatten()
            lig['edge_mask'] = lig['edge_mask'].clone()
            lig['edge_mask'][selected[torch.tensor(~keep, device='cuda')]] = False
            masks = lig['mask_rotate']
            if isinstance(masks, (list, tuple)):
                masks = masks[0]
            lig['mask_rotate'] = np.asarray(masks)[keep]
            p1 = _records(entry['protein1'], 'ATOM')
            p2 = _records(entry['protein2'], 'ATOM')
            # Original PDB ligand preserves native residue and atom identifiers;
            # CCD restoration is explicitly verified to retain this atom order.
            lp = _records(entry['protein1'].with_name('ligand.pdb'), 'HETATM')
            ligand_lines = [lp[0][i] for i in lp[1]]
            require([l[76:78].strip() for l in ligand_lines] == [
                a.GetSymbol().upper() for a in mol.GetAtoms()],
                'ligand PDB and CCD elements differ')
            require(np.max(np.abs(lp[2] - mol.GetConformer().GetPositions())) < 0.002,
                    'ligand PDB and CCD coordinates differ')
            e3 = cached['e3_data']['coords'].numpy()
            native_poi = cached['poi_data']['coords'].numpy()
            native_lig = cached['ligand_data']['coords'].numpy()
            p1_ca = _featurized_ca_coords(entry['protein1'])
            p2_ca = _featurized_ca_coords(entry['protein2'])
            fr, ft = _fit_rigid(e3, p1_ca)
            errors = dict(e3=float(np.max(np.abs(e3 @ fr + ft - p1_ca))),
                          poi=float(np.max(np.abs(native_poi @ fr + ft - p2_ca))),
                          ligand=float(np.max(np.abs(native_lig @ fr + ft - lp[2]))))
            require(max(errors.values()) < 0.002,
                    f'Cache/template frame mismatch: {errors}')
            p1_lines = [p1[0][i] for i in p1[1]]
            chain = p1_lines[0][21]
            ligand_lines = [l[:21] + chain + l[22:] for l in ligand_lines]
            info = dict(p1=p1_lines, p2=[p2[0][i] for i in p2[1]],
                        p2_xyz=p2[2], p2_ca=p2_ca, lig=ligand_lines,
                        chain=chain, frame=(fr, ft))
            info['native_lines'] = info['p1'] + info['p2'] + info['lig']
            chains = sorted({l[21] for l in info['native_lines']})
            require(len(chains) == 2, f'Expected precisely two metric chains: {chains}')
            native_source = args.deepternary / 'data/TernaryDB/pdbs' / case / 'gt_complex.pdb'
            if not native_source.is_file():
                raise FileNotFoundError(native_source)
            native_path = folder / 'native.pdb'
            native_path.write_bytes(native_source.read_bytes())
            identity_path = folder / 'native_reconstructed.pdb'
            rebuild(info, native_lig, native_poi, identity_path)
            direct = score(native_path, native_path)
            identity = score(identity_path, native_path)
            require(identity['dockq'] > 0.99999 and identity['fnat'] == 1.0,
                    f'native DockQ identity failed: {identity}')
            require(max(identity['irms'], identity['lrms']) < 0.002,
                    f'native RMS identity failed: {identity}')
            check = dict(case=case, source_signature_verified=True,
                         native_source=str(native_source), native_sha256=sha256(native_source),
                         cache_sha256=sha256(cache_path), cache_version=7,
                         frame_max_abs_error_A=errors, chains=chains,
                         protein_atom_counts=[len(info['p1']), len(info['p2'])],
                         ligand_atoms=len(info['lig']), atom_identity_order_verified=True,
                         single_bond_torsions=int(lig['edge_mask'].sum()),
                         direct_identity=direct, reconstructed_identity=identity)
            report['native_checks'].append(check)
            save_json(args.out / 'report.json', report)
            batches[case], infos[case] = batch, info
            print('NATIVE PASS', case, json.dumps(identity), flush=True)

        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(payload['architecture'] == 'residual_consensus_factorial_v1',
                'unsupported checkpoint architecture')
        contract = payload['experiment_contract']
        require(contract['chemistry'] == 'ccd' and contract['torsion'] == 'single_bond',
                'checkpoint chemistry/torsion contract differs')
        require(contract['adaptation'] == 'frozen', 'checkpoint expert is not frozen')
        expert = build_diffdock_expert(str(expert_dir), 'cuda')
        model = ExperimentalConsensus(expert, ConsensusFuse(**payload['model_config']),
                                      payload['expert_lora_config'])
        runtime = payload['runtime_config']
        if runtime.get('offpath_target') == 'e3_ligand_endpoint_translation_v1':
            trainer_class = EndpointTranslationRolloutTrainer
            variant = dict(rollout_probability=runtime['rollout_probability'],
                           rollout_steps=runtime['rollout_steps'],
                           rollout_min_time=runtime['rollout_min_time'],
                           endpoint_blend=runtime['endpoint_blend'],
                           endpoint_correction_ratio=runtime['endpoint_correction_ratio'])
        elif runtime.get('offpath_target') == 'poi_body_frame_transported_coupling_v1':
            from tricoopdiff.offpath_experiments import BodyFrameTransportRolloutTrainer
            trainer_class = BodyFrameTransportRolloutTrainer
            variant = dict(rollout_probability=runtime['rollout_probability'],
                           rollout_steps=runtime['rollout_steps'],
                           rollout_min_time=runtime['rollout_min_time'],
                           contract_mode=runtime['offpath_target'])
        elif runtime.get('offpath_target') == 'poi_body_frame_transported_gauge_correct_v2':
            from tricoopdiff.offpath_experiments import (
                GaugeCorrectBodyFrameTransportRolloutTrainer,
            )
            trainer_class = GaugeCorrectBodyFrameTransportRolloutTrainer
            variant = dict(rollout_probability=runtime['rollout_probability'],
                           rollout_steps=runtime['rollout_steps'],
                           rollout_min_time=runtime['rollout_min_time'],
                           contract_mode=runtime['offpath_target'])
        elif runtime.get('offpath_target') == 'joint_rigid_endpoint_kabsch_bounded_v1':
            from tricoopdiff.offpath_experiments import BoundedJointEndpointRolloutTrainer
            trainer_class = BoundedJointEndpointRolloutTrainer
            variant = dict(
                rollout_probability=runtime['rollout_probability'],
                rollout_steps=runtime['rollout_steps'],
                rollout_min_time=runtime['rollout_min_time'],
                endpoint_blend=runtime['endpoint_blend'],
                endpoint_translation_cap=runtime['endpoint_translation_cap'],
                endpoint_rotation_cap=runtime['endpoint_rotation_cap'],
                contract_mode=runtime['offpath_target'])
        elif runtime.get('trajectory_objective') in {
                'differentiable_sdpi_trajectory_v1',
                'differentiable_sdpi_staged_poi_v2',
                'differentiable_sdpi_staged_poi_distance_v2'}:
            from tricoopdiff.trajectory_experiment import (
                StagedPOITrajectoryTrainer, TrajectoryConsistencyTrainer)
            staged = runtime['trajectory_objective'] != 'differentiable_sdpi_trajectory_v1'
            trainer_class = (StagedPOITrajectoryTrainer if staged
                             else TrajectoryConsistencyTrainer)
            variant = dict(
                trajectory_steps=runtime['trajectory_steps'],
                trajectory_weight=runtime['trajectory_weight'],
                trajectory_min_time=runtime['trajectory_min_time'],
                contract_mode=runtime['trajectory_objective'])
            if staged:
                variant.update(
                    trajectory_max_time=runtime['trajectory_max_time'],
                    distance_weight_cap=runtime['trajectory_distance_weight_cap'])
        elif 'rollout_probability' in runtime:
            trainer_class = RolloutAugmentedTrainer
            variant = dict(rollout_probability=runtime['rollout_probability'],
                           rollout_steps=runtime['rollout_steps'])
        else:
            trainer_class = ExperimentalConsensusTrainer
            variant = {}
        trainer = trainer_class(
            model, SDPIFlow(), device='cuda',
            experiment_contract=contract,
            lr=runtime['optimizer_groups']['trifuse']['lr'], lora_lr=1e-5,
            lora_early_lr=5e-6, lora_warmup_steps=0,
            activation_clamp=runtime['activation_clamp'],
            activation_clamp_mode=runtime['activation_clamp_mode'],
            residual_weight=runtime['residual_weight'], ema_decay=runtime['ema_decay'],
            **variant)
        expected_runtime = trainer.runtime_contract()
        if expected_runtime == runtime:
            trainer.load_checkpoint(checkpoint)
        else:
            # Early local transported pilots were written while the explicit
            # duplicate contract key was being added. Its value is already
            # preserved by offpath_target; accept only this exact omission.
            compatible = dict(expected_runtime)
            duplicate = compatible.pop('offpath_contract_mode', None)
            if (compatible != runtime or duplicate != runtime.get('offpath_target')
                    or runtime.get('offpath_target') != 'poi_body_frame_transported_coupling_v1'):
                raise ValueError(f'evaluator trainer runtime mismatch: expected={expected_runtime!r} saved={runtime!r}')
            validate_expert_base_identity(payload, model.expert)
            model.trifuse.load_state_dict(payload['trifuse_state_dict'])
            trainer.ema = {k: v.clone() for k, v in payload['ema_trifuse_state_dict'].items()}
            report['checkpoint_runtime_migration'] = 'accepted missing duplicate offpath_contract_mode key'
        report['checkpoint_contract'] = {k: payload[k] for k in (
            'architecture', 'checkpoint_version', 'experiment_contract',
            'runtime_config', 'expert_base_identity', 'global_step')}
        print('CHECKPOINT PASS: experimental contract, expert identity, EMA', flush=True)

        class RecordingInitializer(TernaryInitializer):
            def initialize(self, *a, **kw):
                result = super().initialize(*a, **kw)
                self.initial = [x.detach().cpu().numpy().copy() for x in result[:3]]
                return result

        weight_context = trainer.ema_parameters() if args.weights == 'ema' else nullcontext()
        with weight_context, torch.no_grad():
            model.eval()
            for case in args.cases:
                started = time.monotonic()
                print('ROLLOUT START', case, 'seed', args.seed, 'steps', args.steps, flush=True)
                batch, info = batches[case], infos[case]
                initializer = RecordingInitializer()
                result = sample_single(model, trainer.diffusion, initializer,
                                       **batch, steps=args.steps, device='cuda',
                                       seed=args.seed, return_trajectory=True)
                folder = args.out / case
                np.savez_compressed(folder / 'coordinates.npz',
                    initial_ligand=initializer.initial[0], initial_poi=initializer.initial[1],
                    initial_body_rotation=initializer.initial[2],
                    final_ligand=result['ligand_coords'], final_poi=result['poi_coords'],
                    trajectory_ligand=np.array([x[0] for x in result['trajectory']]),
                    trajectory_poi=np.array([x[1] for x in result['trajectory']]))
                save_json(folder / 'initialization.json', result['initialization'])
                require(result['valid'] and len(result['trajectory']) == args.steps,
                        result['termination_reason'])
                initial_hash = hashlib.sha256(b''.join(x.tobytes() for x in initializer.initial)).hexdigest()
                repeat = TernaryInitializer().initialize(
                    batch['e3_data']['coords'][0], batch['poi_data']['coords'][0],
                    batch['ligand_data'], seed=args.seed)
                require(all(np.array_equal(old, new.cpu().numpy())
                            for old, new in zip(initializer.initial, repeat[:3])),
                        'Initialization not reproducible')
                pred_path = folder / 'baseline.pdb'
                fit_error = rebuild(info, result['ligand_coords'], result['poi_coords'], pred_path)
                metrics = score(pred_path, folder / 'native.pdb')
                row = dict(case=case, seed=args.seed, steps=args.steps, valid=True,
                           **metrics, poi_rigid_fit_max_abs_error_A=fit_error,
                           initial_state_sha256=initial_hash, initialization_repeat_exact=True,
                           ligand_center_error_A=float(np.linalg.norm(result['ligand_coords'].mean(0) -
                               batch['ligand_data']['coords'][0].cpu().numpy().mean(0))),
                           poi_center_error_A=float(np.linalg.norm(result['poi_coords'].mean(0) -
                               batch['poi_data']['coords'][0].cpu().numpy().mean(0))),
                           seconds=time.monotonic() - started)
                report['baseline'].append(row)
                save_json(args.out / 'report.json', report)
                print('ROLLOUT PASS', json.dumps(row), flush=True)
        report['status'] = 'complete'
        report['mean'] = {k: float(np.mean([r[k] for r in report['baseline']]))
                          for k in ('dockq', 'fnat', 'irms', 'lrms')}
        report['limitations'] = [f'{len(args.cases)} fixed cases and one seed are a diagnostic baseline.',
                                'No training or quality improvement is claimed.',
                                'Initialization was repeated exactly; full rollout was run once per case.']
        save_json(args.out / 'report.json', report)
        print('COMPLETE', json.dumps(report['mean']), flush=True)
    except BaseException as error:
        report['status'] = 'blocked'
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        save_json(args.out / 'report.json', report)
        raise


if __name__ == '__main__':
    main()
