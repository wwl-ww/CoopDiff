#!/usr/bin/env python
"""Evaluate fixed coupled banks with checkpoint weights; never resume optimization."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

import tricoopdiff.diffdock_compat
from train_entry import (build_trainer, TernaryDataLoader, _load_score_model_args,
                         validation_metric_context, seed_training)
from tricoopdiff.runtime import evaluation_context, sha256_file
from tricoopdiff.models.expert import validate_expert_base_identity
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from native_contact_evaluation import native_contact_preflight, contact_strata
from featurize_common import esm_encoder_identity


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--bank-root', type=Path,
        default=Path('/root/.local/share/coopdiff/small_sample_seed73/coupled_pilot'))
    parser.add_argument('--annotate-existing', action='store_true',
        help='Add official native-contact strata to existing reports without rerunning models')
    parser.add_argument('--validate-only', action='store_true',
        help='Build model and verify fixed bank/checkpoint coverage without rollouts')
    parser.add_argument('--reuse-update0', action='store_true',
        help='Reuse bank-root update0 results only after exact checkpoint/state equality')
    parser.add_argument('--config', type=Path,
        help='Matching training config; defaults to fixed-bank pilot config')
    parser.add_argument('--unseen-data', type=Path,
        help='Explicit cross-complex index; defaults to config or small-sample index')
    parser.add_argument('--training-bank-root', type=Path,
        help='Pinned training bank when evaluating separately prepared transfer groups')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--groups', default='same_complex,independent_center,unseen_complex',
        help='Explicit comma-separated validation groups; every selected group is required')
    parser.add_argument('--weights', default='raw,ema', choices=('raw', 'ema', 'raw,ema'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--binary-condition-mode', choices=('all','off','early','late'), default='all',
        help='Fixed per-call FiLM policy; controller still receives both proteins')
    parser.add_argument('--memory-fraction', type=float, default=.15)
    opt = parser.parse_args()
    opt.groups = tuple(opt.groups.split(','))
    if (not opt.groups or len(set(opt.groups)) != len(opt.groups)
            or not set(opt.groups) <= {'same_complex', 'independent_center', 'unseen_complex'}):
        parser.error('groups must contain unique supported validation groups')
    if not 0 < opt.memory_fraction <= 1:
        parser.error('memory-fraction must be in (0,1]')
    if opt.training_bank_root and 'same_complex' in opt.groups:
        parser.error('a separate training bank is only allowed for transfer groups')
    return opt


def fixed_contact_audit(opt):
    config=json.loads((opt.config or opt.bank_root/'config.json').read_text())
    entries={}
    indexes = ['train.json']
    if 'unseen_complex' in opt.groups:
        indexes.append('unseen_complex_val.json')
    for name in indexes:
        for entry in json.loads((ROOT/'results/small_sample_seed73'/name).read_text())['entries']:
            context=validation_metric_context(entry,config['native_root'])
            context['deepternary_root']=config.get('deepternary_root','/root/.local/share/coopdiff/DeepTernary')
            entries[entry['case']]=native_contact_preflight(context)
    return entries


def bank_records(value):
    """Accept historical one-state records and explicit multiple-state lists."""
    records = value if isinstance(value, list) else [value]
    if not records or any(not isinstance(row, dict) or 'coupling' not in row for row in records):
        raise ValueError('fixed bank requires a nonempty record or record list per case')
    identities = [(row['center_sha256'], row['seed']) for row in records]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate fixed validation state within one case')
    return records


def evaluate_group(trainer, loader, banks, args, group, weights, contact_audit):
    records = []
    with evaluation_context(trainer.model):
        for batch in loader:
            for state_index, bank in enumerate(bank_records(banks[batch['case']])):
                context = validation_metric_context(loader.entries_by_case[batch['case']], args.native_root)
                context['deepternary_root'] = args.deepternary_root
                result = trainer.compute_loss(**batch, source=0,
                    coupling_override=bank['coupling'], diagnostics=True)
                ligand, native, target = (result['predicted_ligand'],
                    batch['ligand_data']['coords'], result['target_ligand'])
                metric = compute_fullatom_dockq(ligand[0].cpu().numpy(),
                    result['predicted_poi'][0].cpu().numpy(),
                    batch['e3_data']['coords'][0].cpu().numpy(), **context)
                records.append({'case':batch['case'], **contact_audit[batch['case']], 'initial_seed':bank['seed'],
                    'center_sha256':bank['center_sha256'],
                    'state_index':state_index, 'center_index':bank.get('center_index',0),
                    'validation_seed_base':bank.get('validation_seed_base'),
                    'initialization_mode':bank['initialization_provenance']['mode'],
                    'initial_dockq':bank['initial_dockq']['dockq'],
                    'center_dockq':bank['center_dockq']['dockq'], **metric,
                    'loss_total':result['loss_total'].item(),
                    'loss_endpoint_main':result['loss_endpoint_main'].item(),
                    'ligand_target_rmsd_angstrom':result['ligand_target_rmsd_angstrom'].item(),
                    'ligand_crystal_rmsd_angstrom':(ligand-native).square().sum(-1).mean().sqrt().item(),
                    'projection_residual_rmsd_angstrom':(target-native).square().sum(-1).mean().sqrt().item(),
                    'loss_components':{key:value.item() for key,value in result['components'].items()}})
    expected = sum(len(bank_records(value)) for value in banks.values())
    if len(banks) != 8 or len(records) != expected:
        raise ValueError(f'{group}: expected all 8 cases and {expected} states, got {len(records)} states')
    for row in records:
        if any(not math.isfinite(row[key]) for key in ('dockq','loss_total',
                'ligand_target_rmsd_angstrom','ligand_crystal_rmsd_angstrom')):
            raise FloatingPointError(f"nonfinite evaluation: {row['case']}")
    dockq = torch.tensor([row['dockq'] for row in records], dtype=torch.float64)
    def mean(key, rows=records):
        return sum(row[key] for row in rows)/len(rows)
    modes = {}
    for mode in sorted({row['initialization_mode'] for row in records}):
        rows = [row for row in records if row['initialization_mode']==mode]
        modes[mode] = {'count':len(rows), 'dockq_mean':mean('dockq',rows),
                      'initial_dockq_mean':mean('initial_dockq',rows),
                      'ligand_crystal_rmsd_mean':mean('ligand_crystal_rmsd_angstrom',rows),
                      'ligand_reachable_rmsd_mean':mean('ligand_target_rmsd_angstrom',rows)}
    return {'event':'evaluation', 'global_step':trainer.global_step,
        'group':group, 'weights':weights, 'count':len(records),
        'case_count':len(banks), 'state_count':len(records),
        'seed_base_summaries':{str(seed):{'state_count':len(rows),
            'case_count':len({row['case'] for row in rows}), 'dockq_mean':mean('dockq',rows),
            'initial_dockq_mean':mean('initial_dockq',rows)}
            for seed in sorted({row['validation_seed_base'] for row in records
                                if row['validation_seed_base'] is not None})
            for rows in [[row for row in records if row['validation_seed_base']==seed]]},
        'dockq_mean':dockq.mean().item(), 'dockq_median':dockq.quantile(.5).item(),
        'dockq_above_023':int((dockq>.23).sum()), 'loss_mean':mean('loss_total'),
        'ligand_crystal_rmsd_mean':mean('ligand_crystal_rmsd_angstrom'),
        'ligand_reachable_rmsd_mean':mean('ligand_target_rmsd_angstrom'),
        'projection_residual_rmsd_mean':mean('projection_residual_rmsd_angstrom'),
        'initial_dockq_mean':mean('initial_dockq'), 'center_dockq_mean':mean('center_dockq'),
        'mode_summaries':modes, **contact_strata(records), 'per_case':records}


def reuse_update0(opt):
    config = json.loads((opt.config or opt.bank_root/'config.json').read_text())
    banks = torch.load(opt.bank_root/'validation_starts.pt', map_location='cpu', weights_only=False)
    centers = torch.load(opt.bank_root/'predicted_centers.pt', map_location='cpu', weights_only=False)
    if not set(opt.groups) <= set(banks):
        raise ValueError('requested fixed validation groups are missing')
    from train_distributed import audit_coupled_centers
    required = tuple({'same_complex':'train', 'independent_center':'independent_center',
                      'unseen_complex':'unseen_complex'}[group] for group in opt.groups)
    if audit_coupled_centers(centers, config['cache_dir'], required)['invalid_count']:
        raise ValueError('cannot reuse evaluations from an invalid center bank')
    old_path = opt.bank_root/'checkpoint_update0.pt'
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    new = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    for payload in (old,new):
        if payload['global_step']!=0:
            raise ValueError('update0 reuse requires both checkpoint steps equal 0')
    for field in ('model_config','runtime_config','expert_base_identity'):
        if old[field] != new[field]:
            raise ValueError(f'update0 reuse {field} differs')
    tensor_counts = {}
    for field in ('trifuse_state_dict','ema_trifuse_state_dict'):
        left,right = old[field],new[field]
        if left.keys()!=right.keys() or any(not torch.equal(left[k],right[k]) for k in left):
            raise ValueError(f'update0 reuse requires exact tensors: {field}')
        tensor_counts[field] = len(left)
    output = opt.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    provenance = {'checkpoint':str(opt.checkpoint.resolve()),
        'checkpoint_sha256':sha256_file(opt.checkpoint),
        'reused_source_checkpoint':str(old_path.resolve()),
        'reused_source_checkpoint_sha256':sha256_file(old_path),
        'validation_bank_sha256':sha256_file(opt.bank_root/'validation_starts.pt'),
        'center_bank_sha256':sha256_file(opt.bank_root/'predicted_centers.pt'),
        'exact_state_tensor_counts':tensor_counts, 'runtime_config_exact':True,
        'optimizer_state_loaded':False, 'center_regeneration':False}
    contact_audit = fixed_contact_audit(opt)
    for weights in opt.weights.split(','):
        for group in opt.groups:
            source = opt.bank_root/f'eval_update0_{group}_{weights}.json'
            report = json.loads(source.read_text())
            expected = sum(len(bank_records(row)) for row in banks[group].values())
            if report['global_step']!=0 or report['count']!=expected:
                raise ValueError('invalid source update0 report')
            for row in report['per_case']:
                row.update(contact_audit[row['case']])
            report.update(contact_strata(report['per_case']))
            report['native_contact_audit']={'official_fnat_5A':True,'all_cases_retained':True}
            report['provenance'] = {**provenance,'reused_source_evaluation_sha256':sha256_file(source)}
            destination = output/source.name
            if destination.exists() and json.loads(destination.read_text()) != report:
                raise ValueError(f'refusing different existing report: {destination}')
            destination.write_text(json.dumps(report,indent=2))
    (output/'update0_reuse_audit.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'event':'complete_reused_exact_update0',**provenance}),flush=True)


def main():
    opt = options()
    if opt.annotate_existing:
        contact_audit=fixed_contact_audit(opt)
        for path in opt.output_dir.glob('eval_update*_*.json'):
            report=json.loads(path.read_text())
            for row in report['per_case']:
                row.update(contact_audit[row['case']])
            report.update(contact_strata(report['per_case']))
            report['native_contact_audit']={'official_fnat_5A':True,'all_cases_retained':True}
            path.write_text(json.dumps(report,indent=2))
        (opt.output_dir/'native_contact_preflight.json').write_text(json.dumps(contact_audit,indent=2))
        print(json.dumps({'event':'existing_reports_annotated','zero_native_contacts':
            [case for case,audit in contact_audit.items() if audit['zero_native_contacts']]}),flush=True)
        return
    if opt.reuse_update0:
        reuse_update0(opt)
        return
    # Keep the production relative-path contract while making every persisted
    # input explicit. No initializer or candidate generator is imported here.
    os.chdir(ROOT)
    config = json.loads((opt.config or opt.bank_root/'config.json').read_text())
    args = SimpleNamespace(**config)
    # Build from the saved architecture/runtime policy, never today's factory
    # defaults. This path still never calls trainer.load_checkpoint or loads
    # optimizer state; only strict raw/EMA weights are loaded below.
    args.resume = str(opt.checkpoint.resolve())
    args.explicit_loss_options = set()
    args.device = opt.device
    args.deepternary_root = config.get('deepternary_root', '/root/.local/share/coopdiff/DeepTernary')
    device = torch.device(opt.device)
    if device.type=='cuda':
        device = torch.device('cuda', device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(opt.memory_fraction, device)
    seed_training(73, device)
    bank_path, center_path = opt.bank_root/'validation_starts.pt', opt.bank_root/'predicted_centers.pt'
    banks = torch.load(bank_path, map_location='cpu', weights_only=False)
    centers = torch.load(center_path, map_location='cpu', weights_only=False)
    if not set(opt.groups) <= set(banks):
        raise ValueError('requested fixed validation groups are missing')
    for group in opt.groups:
        for case, value in banks[group].items():
            for bank in bank_records(value):
                index = bank.get('center_index', 0)
                center = centers['train'][case][index] if group=='same_complex' else centers[group][case]
                if bank['center_sha256'] != center['sha256']:
                    raise ValueError(f'{case}: validation-bank/center identity mismatch')
    from train_distributed import audit_coupled_centers
    required = tuple({'same_complex':'train', 'independent_center':'independent_center',
                      'unseen_complex':'unseen_complex'}[group] for group in opt.groups)
    audit = audit_coupled_centers(centers, args.cache_dir, required)
    if audit['invalid_count']:
        raise ValueError('fixed evaluation center bank contains invalid geometry')
    score_args = _load_score_model_args(args.expert_model_dir)
    encoder_identity = esm_encoder_identity()
    print(json.dumps({'event':'esm_identity_pinned', 'scope':'startup_full_content_read_only'}), flush=True)
    available_paths = {'same_complex':args.val_data, 'independent_center':args.val_data,
        'unseen_complex':str(opt.unseen_data) if opt.unseen_data else getattr(args, 'unseen_complex_val',
                                str(ROOT/'results/small_sample_seed73/unseen_complex_val.json'))}
    loader_paths = {group:available_paths[group] for group in opt.groups}
    probe = TernaryDataLoader(args.val_data, score_args, shuffle=False,
        device=str(device), cache_dir=args.cache_dir, cache_read_only=True, cache_wait_seconds=0,
        expected_esm_encoder_identity=encoder_identity)
    first = next(iter(probe))
    input_dim = int(first['e3_data']['features'].shape[-1])
    del first, probe
    seed_training(73, device)
    model, _, trainer = build_trainer(args, device, input_dim=input_dim)
    # Bind the explicit per-call policy once for this whole evaluation. Never
    # change it during a rollout or activation-checkpoint recomputation.
    from functools import partial
    bound_condition = partial(model.trifuse.binary_condition,
                              ablation_mode=opt.binary_condition_mode)
    condition_calls = [0, 0, 0]
    def observed_condition(*values, stage_index=0):
        condition_calls[stage_index] += 1
        return bound_condition(*values, stage_index=stage_index)
    model.trifuse.binary_condition = observed_condition
    model.to(device)
    trainer.set_phase(args.phase)
    payload = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
    initialization = payload.get('distributed_state', {}).get('initialization_protocol', {})
    if initialization.get('name') == 'coupled_frozen_centers_epoch_v1':
        training_center_path = (opt.training_bank_root/'predicted_centers.pt'
                                if opt.training_bank_root else center_path)
        training_validation_path = (opt.training_bank_root/'validation_starts.pt'
                                    if opt.training_bank_root else bank_path)
        if (initialization['center_bank_sha256'] != sha256_file(training_center_path)
                or initialization.get('validation_bank_sha256') != sha256_file(training_validation_path)):
            raise ValueError('checkpoint was trained with different fixed center/validation banks')
        if opt.training_bank_root:
            original_centers = torch.load(training_center_path, map_location='cpu', weights_only=False)
            identities = lambda bank: {case:[record['sha256'] for record in rows]
                                        for case,rows in bank['train'].items()}
            if identities(original_centers) != identities(centers):
                raise ValueError('transfer bank does not retain the pinned training centers')
    if payload['model_config'] != model.trifuse.config:
        raise ValueError('checkpoint model configuration differs')
    validate_expert_base_identity(payload, model.expert)
    if not payload.get('ema_trifuse_state_dict'):
        raise ValueError('checkpoint missing EMA weights')
    trainer.global_step = int(payload['global_step'])
    if trainer.optimizer.state:
        raise RuntimeError('evaluation construction unexpectedly populated optimizer state')
    provenance = {'checkpoint':str(opt.checkpoint.resolve()),
        'checkpoint_sha256':sha256_file(opt.checkpoint),
        'validation_bank_sha256':sha256_file(bank_path), 'center_bank_sha256':sha256_file(center_path),
        'config_sha256':sha256_file(opt.config or opt.bank_root/'config.json'),
        'script_sha256':sha256_file(__file__), 'optimizer_state_loaded':False,
        'center_regeneration':False, 'selected_groups':list(opt.groups),
        'available_groups':sorted(banks), 'device':str(device),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    provenance['binary_condition_mode'] = opt.binary_condition_mode
    provenance['ablation_scope'] = 'expert_FiLM_only_controller_keeps_opposite_protein'
    provenance['evaluation_indexes'] = {group:{'path':str(Path(path).resolve()),
        'sha256':sha256_file(path)} for group,path in loader_paths.items()}
    if opt.training_bank_root:
        provenance['training_bank_identity'] = {
            'root':str(opt.training_bank_root.resolve()),
            'center_sha256':sha256_file(opt.training_bank_root/'predicted_centers.pt'),
            'validation_sha256':sha256_file(opt.training_bank_root/'validation_starts.pt')}
    output = opt.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    missing_caches = [case for group in opt.groups for case in banks[group]
                      if not (Path(args.cache_dir)/(case+'.pt')).is_file()]
    if missing_caches:
        raise FileNotFoundError(f'fixed evaluation requires every prepared cache: {missing_caches}')
    loaders = {group:TernaryDataLoader(path, score_args, shuffle=False,
        device=str(device), cache_dir=args.cache_dir, expert=model.expert,
        cache_read_only=True, cached_only=False, cache_wait_seconds=0,
        expected_esm_encoder_identity=encoder_identity) for group,path in loader_paths.items()}
    for group, loader in loaders.items():
        if set(loader.entries_by_case) != set(banks[group]):
            raise ValueError(f'{group}: case coverage differs from fixed bank')
    contact_audit = fixed_contact_audit(opt)
    (output/'native_contact_preflight.json').write_text(json.dumps(contact_audit,indent=2))
    if opt.validate_only:
        model.trifuse.load_state_dict(payload['trifuse_state_dict'],strict=True)
        model.trifuse.load_state_dict(payload['ema_trifuse_state_dict'],strict=True)
        print(json.dumps({'event':'validation_preflight_passed','global_step':trainer.global_step,
            'groups':{key:{'case_count':len(value.entries),
                'state_count':sum(len(bank_records(row)) for row in banks[key].values())}
                for key,value in loaders.items()},
            'optimizer_state_loaded':False,
            'cuda_initialized':torch.cuda.is_initialized(),
            'interface_error_scale':trainer.interface_error_scale,
            'model_config_preserved':payload['model_config']==model.trifuse.config}),flush=True)
        return
    for weights in opt.weights.split(','):
        state = payload['trifuse_state_dict' if weights=='raw' else 'ema_trifuse_state_dict']
        model.trifuse.load_state_dict(state, strict=True)
        film_norms = [float(layer.weight.detach().norm().cpu()) + float(layer.bias.detach().norm().cpu())
                      for layer in model.trifuse.binary_film]
        for group, loader in loaders.items():
            path = output/f'eval_update{trainer.global_step}_{group}_{weights}.json'
            if path.exists():
                old = json.loads(path.read_text())
                if old.get('provenance') != provenance:
                    raise ValueError(f'refusing to overwrite different evaluation: {path}')
                print(json.dumps({'event':'skip_existing','path':str(path)}),flush=True)
                continue
            before_calls = list(condition_calls)
            report = evaluate_group(trainer, loader, banks[group], args, group, weights, contact_audit)
            report['provenance'] = provenance
            report['film_parameter_norms_by_stage'] = film_norms
            report['condition_forward_calls_by_stage'] = [after - before
                for after, before in zip(condition_calls, before_calls)]
            active_stages = {'all':(0,1,2), 'off':(), 'early':(0,), 'late':(1,2)}[opt.binary_condition_mode]
            report['film_enabled_forward_calls_by_stage'] = [count if stage in active_stages else 0
                for stage,count in enumerate(report['condition_forward_calls_by_stage'])]
            report['film_heads_all_zero'] = all(value == 0 for value in film_norms)
            report['condition_call_scope'] = 'both expert branches; includes auxiliary near-target loss rollouts'
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(report,indent=2))
            temp.replace(path)
            print(json.dumps({key:value for key,value in report.items()
                if key not in ('per_case','provenance')}),flush=True)
    if trainer.optimizer.state:
        raise RuntimeError('evaluation changed optimizer state')
    print(json.dumps({'event':'complete','global_step':trainer.global_step,
                      'weights':opt.weights, 'optimizer_state_loaded':False,
            'cuda_initialized':torch.cuda.is_initialized(),
            'interface_error_scale':trainer.interface_error_scale,
            'model_config_preserved':payload['model_config']==model.trifuse.config}),flush=True)


if __name__=='__main__':
    main()
