"""Fixed-init single-case Matcha assembly overfit, with exact protein DockQ.

This is an explicitly isolated research checkpoint format, not a production
training checkpoint. Only the ternary module is optimized; Matcha stays frozen.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tricoopdiff.diffdock_compat
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import reachable_ligand_target
from tricoopdiff.metrics.dockq import compute_fullatom_dockq
from tricoopdiff.models.assembly import TernaryAssembly, TernaryAssemblyModel, production_model_config
from tricoopdiff.sdpi_flow import SDPIFlow
from tricoopdiff.train import (AssemblyTrainer, clip_grad_norm_finite, NonFiniteTrainingError,
                              LIGAND_INTERFACE_CONTRACT, LIGAND_INTERFACE_LOSS_KEYS)
from tricoopdiff.runtime import (atomic_torch_save, sha256_file, evaluation_context,
                                capture_rng_state, restore_rng_state)

CASE = '4G8A_B_C_LP5'
FORMAT = 'matcha_fixed_case_research_resume_v2'


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(move(item, device) for item in value)
    return value


def common_descent_gradients(losses, parameters):
    """Unit bisector of two exact gradients, including unused zero slots.

    Its dot product with either nonzero gradient is nonnegative. This is a
    local first-order property, not a substitute for executed-rollout checks.
    """
    gradients = [torch.autograd.grad(loss, parameters, retain_graph=index == 0,
                                     allow_unused=True) for index, loss in enumerate(losses)]
    gradients = [[torch.zeros_like(p) if g is None else g for p, g in zip(parameters, group)]
                 for group in gradients]
    if any(not torch.isfinite(g).all() for group in gradients for g in group):
        raise NonFiniteTrainingError('non-finite individual endpoint gradient')
    norms = [torch.stack([g.double().square().sum() for g in group]).sum().sqrt()
             for group in gradients]
    dot = sum((a.double() * b.double()).sum() for a, b in zip(*gradients))
    direction = [a.double() / norms[0].clamp_min(1e-30)
                 + b.double() / norms[1].clamp_min(1e-30) for a, b in zip(*gradients)]
    norm = torch.stack([g.square().sum() for g in direction]).sum().sqrt()
    if not torch.isfinite(norm) or norm < 1e-12:
        raise RuntimeError('no nonzero first-order common descent direction')
    for parameter, gradient in zip(parameters, direction):
        parameter.grad = (gradient / norm).to(parameter)
    return dict(ligand_gradient_norm=float(norms[0]), protein_gradient_norm=float(norms[1]),
                gradient_cosine=float(dot / (norms[0] * norms[1]).clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--weights-from', help='Explicit weights-only transfer; resets optimizer and counters')
    parser.add_argument('--updates', type=int, default=300, help='Total optimizer updates, including resumed updates')
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--lr', type=float, help='Fresh-run default 1e-4; required explicitly for weights-only transfer')
    parser.add_argument('--node-dim', type=int, default=128)
    parser.add_argument('--steric-weight', type=float, default=.05)
    parser.add_argument('--optimizer', choices=('adam', 'pareto'), default='adam',
                        help='pareto: exact two-objective gradients with real-rollout backtracking')
    parser.add_argument('--goal-dockq', type=float, default=.90)
    goal = parser.add_mutually_exclusive_group()
    goal.add_argument('--goal-loss', type=float, default=.01)
    goal.add_argument('--goal-ligand-rmsd', type=float,
                      help='Physical Å target, jointly required with goal DockQ, instead of total-loss threshold')
    parser.add_argument('--source-dir', default='results/matcha_qualification_source')
    parser.add_argument('--weights-dir', default='results/matcha_qualification_weights')
    args = parser.parse_args()
    if args.resume and args.weights_from:
        parser.error('--resume and --weights-from are mutually exclusive')
    if args.weights_from and args.lr is None:
        parser.error('weights-only transfer resets Adam; choose --lr explicitly, do not treat it as resume')
    args.lr = 1e-4 if args.lr is None else args.lr
    if (args.updates < 0 or args.eval_every < 1 or args.save_every < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not 0 <= args.goal_dockq <= 1 or not math.isfinite(args.goal_loss) or args.goal_loss < 0
            or (args.goal_ligand_rmsd is not None
                and (not math.isfinite(args.goal_ligand_rmsd) or args.goal_ligand_rmsd <= 0))):
        parser.error('invalid update budget, interval, learning rate, or goal threshold')
    out = Path(args.out).resolve()
    if not args.resume and (out/'config.json').exists():
        parser.error('output already contains a run; use --resume or a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    device = 'cuda'
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    transfer = torch.load(args.weights_from, map_location='cpu', weights_only=False) if args.weights_from else None
    if transfer is not None and (transfer.get('case') != CASE or transfer.get('seed') != 73):
        raise ValueError('weights source must belong to the same fixed-case experiment')
    if checkpoint is not None:
        if checkpoint.get('format') != FORMAT or checkpoint['case'] != CASE or checkpoint['seed'] != 73:
            raise ValueError('checkpoint does not match the fixed research experiment')
    case_dir = ROOT / f'data/ternary_deepternary/train/{CASE}'
    p1, p2 = case_dir/'protein1.pdb', case_dir/'protein2.pdb'
    cache_path = ROOT/f'results/featurize_cache/{CASE}.pt'
    input_manifest = {str(path.relative_to(ROOT)): sha256_file(path)
                      for path in (p1, p2, case_dir/'ligand.pdb', cache_path)}
    if checkpoint is not None and checkpoint['input_manifest'] != input_manifest:
        raise ValueError('resume indexed structures or feature cache changed')
    if transfer is not None and transfer.get('input_manifest', input_manifest) != input_manifest:
        raise ValueError('fixed-case transfer indexed structures or feature cache changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    ed, pd, ld = [move(copy.deepcopy(cache[key]), device) for key in ('e3_data', 'poi_data', 'ligand_data')]
    for data in (ed, pd, ld):
        for key in ('coords', 'features'):
            if data[key].ndim == 2:
                data[key] = data[key].unsqueeze(0)
    if isinstance(ld['mol'], bytes):
        ld['mol'] = Chem.Mol(ld['mol'])
    fixed = checkpoint if checkpoint is not None else transfer
    if fixed is None:
        ligand, poi, body, provenance = TernaryInitializer(no_torsion=False).initialize(
            ed['coords'][0], pd['coords'][0], ld, seed=73)
        target, projection = reachable_ligand_target(ld, ligand, poi, pd['coords'], no_torsion=False)
        coupling = dict(initial_state=dict(ligand=ligand.detach().cpu(), poi=poi.detach().cpu(),
                                           body_frame=body.detach().cpu()),
                        target_ligand=torch.as_tensor(target).detach().cpu())
    else:
        coupling = fixed['fixed_coupling']
        provenance = fixed['initialization_provenance']
        projection = fixed['target_projection']
    # Native structure is used only by loss / evaluation, not as an expert input.
    native_path = out/'native_proteins.pdb'
    records = []
    for path, chain in ((p1, 'A'), (p2, 'B')):
        records.extend(line[:21]+chain+line[22:] for line in path.read_text().splitlines()
                       if line.startswith('ATOM'))
    native_path.write_text('\n'.join(records)+'\n')
    metric_context = dict(protein1_pdb=str(p1), protein2_pdb=str(p2),
        ligand_pdb=str(case_dir/'ligand.pdb'), native_complex_pdb=str(native_path),
        deepternary_root=str(ROOT/'DeepTernary'))

    from tricoopdiff.models.expert import FrozenMatcha, expert_runtime_identity
    expert = FrozenMatcha(source_dir=str(ROOT/args.source_dir), weights_dir=str(ROOT/args.weights_dir))
    expert = expert.to(device)
    ed, pd, ld = expert.prepare_case(ed, pd, ld, (str(p1), str(p2)))
    config = production_model_config(args.node_dim, int(ed['features'].shape[-1]),
        expert_scalar_dim=expert.node_scalar_dim, expert_vector_dim=expert.node_vector_dim,
        expert_backend='matcha')
    model = TernaryAssemblyModel(expert, TernaryAssembly(**config), expert_input_grad=True,
                                checkpoint_ternary=True).to(device)
    trainer = AssemblyTrainer(model, SDPIFlow(inference_steps=20), lr=args.lr,
        residual_weight=0., interface_loss_weight=1., steric_loss_weight=args.steric_weight,
        training_source_mix='sdpi_only', training_endpoint_protocol='none', device=device)
    params = [p for p in model.trifuse.parameters() if p.requires_grad]
    if args.optimizer == 'pareto':
        trainer.optimizer = torch.optim.SGD(params, lr=args.lr)
    manifest = expert_runtime_identity(expert)
    runtime = dict(lr=args.lr, steric_weight=args.steric_weight, steps=20,
                   optimizer=args.optimizer,
                   source='sdpi', endpoint_weight=.1, interface_weight=1., near_target_aux=False,
                   ligand_geometry_scale_angstrom=1.,
                   protein_geometry_scale_angstrom=1.,
                   ligand_interface_contract=dict(LIGAND_INTERFACE_CONTRACT),
                   gradient_clip=10. if args.optimizer == 'adam' else None, expert_input_grad=True,
                   code_identity={str(path.relative_to(ROOT)): sha256_file(path) for path in (
                       Path(__file__).resolve(), ROOT/'tricoopdiff/models/assembly.py',
                       ROOT/'tricoopdiff/train.py', ROOT/'tricoopdiff/sdpi_flow.py')})
    step = 0
    if transfer is not None:
        if (transfer['model_config'] != config or
                transfer['backend_manifest']['stage_weights_sha256'] != manifest['stage_weights_sha256']):
            raise ValueError('weights transfer requires the current architecture and pretrained stages')
        model.trifuse.load_state_dict(transfer['trainable_weights'], strict=True)
        trainer.ema = {name: value.detach().clone() for name, value in model.trifuse.state_dict().items()}
        print(json.dumps(dict(kind='weights_only_transfer', source=args.weights_from,
                              source_step=transfer['step'], optimizer_reset=True)), flush=True)

    if checkpoint is not None:
        if checkpoint['model_config'] != config or checkpoint['runtime'] != runtime:
            raise ValueError('resume architecture/training contract mismatch')
        if checkpoint['backend_manifest'] != manifest:
            raise ValueError('resume pretrained backend identity changed')
        model.trifuse.load_state_dict(checkpoint['trainable_weights'], strict=True)
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('resume update counter must be a nonnegative integer')
        restore_rng_state(checkpoint['rng'])
    batch = dict(e3_data=ed, poi_data=pd, ligand_data=ld, case=CASE,
                 coupling_override=coupling, diagnostics=True, source=0)
    (out/'config.json').write_text(json.dumps(dict(case=CASE, seed=73, model_config=config,
        runtime=runtime, backend_manifest=manifest, input_manifest=input_manifest, checkpoint_format=FORMAT,
        goal=dict(dockq=args.goal_dockq, ligand_rmsd_angstrom=args.goal_ligand_rmsd,
                  loss=None if args.goal_ligand_rmsd is not None else args.goal_loss),
        source_note='LP5 phosphorylated lipid pressure case, not PROTAC'), indent=2))

    def save(filename='progress.pt'):
        payload = dict(format=FORMAT, case=CASE, seed=73, step=step,
            model_config=config, runtime=runtime, backend_manifest=manifest, input_manifest=input_manifest,
            weights_source=args.weights_from if checkpoint is None else checkpoint.get('weights_source'),
            trainable_weights=model.trifuse.state_dict(), optimizer=trainer.optimizer.state_dict(),
            rng=capture_rng_state(), fixed_coupling=coupling, initialization_provenance=provenance,
            target_projection=projection)
        atomic_torch_save(payload, out/filename)

    def evaluate(record=True):
        with evaluation_context(model), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            result = trainer.compute_loss(**batch)
        metric = compute_fullatom_dockq(result['predicted_ligand'][0].cpu().numpy(),
            result['predicted_poi'][0].cpu().numpy(), ed['coords'][0].cpu().numpy(), **metric_context)
        values = dict(kind='evaluation', step=step, loss=float(result['loss_total']), **metric,
            ligand_interface_pairs=result['ligand_interface_pairs'],
            ligand_target_rmsd_angstrom=float(result['ligand_target_rmsd_angstrom']),
            **{name: float(value) for name, value in result['components'].items()})
        if record:
            print(json.dumps(values), flush=True)
            with (out/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(values)+'\n')
            torch.save(dict(ligand=result['predicted_ligand'].cpu(), poi=result['predicted_poi'].cpu()),
                       out/'latest_prediction.pt')
        return values

    def meets_goal(values):
        geometry_ok = (values['ligand_target_rmsd_angstrom'] <= args.goal_ligand_rmsd
                       if args.goal_ligand_rmsd is not None else values['loss'] <= args.goal_loss)
        return values['dockq'] >= args.goal_dockq and geometry_ok

    before = evaluate()
    save()
    reached_goal = meets_goal(before)
    if reached_goal:
        save('goal.pt')
    start = time.time()
    current = before
    while step < args.updates and not reached_goal:
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            result = trainer.compute_loss(**batch)
            if args.optimizer == 'pareto':
                c = result['components']
                geometry = (c['loss_ligand_geometry'] + args.steric_weight * c.get('loss_clash_geometry', 0.)
                            + LIGAND_INTERFACE_CONTRACT['per_interface_weight']
                            * sum(c[key] for key in LIGAND_INTERFACE_LOSS_KEYS))
                protein = sum(c[key] for key in ('loss_protein_geometry', 'loss_interface_geometry',
                                                 'loss_contact_geometry'))
                gradient_info = common_descent_gradients((geometry, protein), params)
            else:
                result['loss_total'].backward()
        if args.optimizer == 'pareto':
            previous = current
            old = [p.detach().clone() for p in params]
            previous_protein = sum(previous[key] for key in ('loss_protein_geometry',
                                    'loss_interface_geometry', 'loss_contact_geometry'))
            trial_lr = min(args.lr, trainer.optimizer.param_groups[0]['lr'] * 1.5)
            accepted = False
            for trials in range(16):
                with torch.no_grad():
                    for p, value in zip(params, old):
                        p.copy_(value)
                trainer.optimizer.param_groups[0]['lr'] = trial_lr
                trainer.optimizer.step()
                try:
                    candidate = evaluate(record=False)
                except (FloatingPointError, NonFiniteTrainingError):
                    candidate = None
                if candidate is not None:
                    candidate_protein = sum(candidate[key] for key in ('loss_protein_geometry',
                                             'loss_interface_geometry', 'loss_contact_geometry'))
                    accepted = (
                        candidate['ligand_target_rmsd_angstrom'] <= previous['ligand_target_rmsd_angstrom'] + 1e-6
                        and candidate_protein <= previous_protein + 1e-6
                        and candidate['dockq'] >= min(args.goal_dockq, previous['dockq']) - 1e-6
                        and candidate['loss'] < previous['loss'] - 1e-7 * max(1., previous['loss']))
                if accepted:
                    current = candidate
                    break
                trial_lr *= .5
            if not accepted:
                with torch.no_grad():
                    for p, value in zip(params, old):
                        p.copy_(value)
                save('line_search_stalled.pt')
                raise RuntimeError('no verified joint improvement; parameters restored, not counted as an update')
            gradient = 1.
            gradient_info.update(line_search_trials=trials + 1, accepted_lr=trial_lr,
                                 dockq=current['dockq'], ligand_target_rmsd_angstrom=current['ligand_target_rmsd_angstrom'],
                                 loss_before=float(result['loss_total'].detach()), loss_after=current['loss'])
        else:
            try:
                gradient = float(clip_grad_norm_finite(params, 10.))
            except NonFiniteTrainingError:
                save('nonfinite_before_update.pt')
                raise
            trainer.optimizer.step()
            gradient_info = {}
        step += 1
        row = dict(kind='train', step=step, loss=float(result['loss_total'].detach()),
                   grad_norm=gradient, elapsed_seconds=time.time()-start, **gradient_info)
        print(json.dumps(row), flush=True)
        with (out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if step % args.save_every == 0:
            save()
        if step % args.eval_every == 0 or step == args.updates:
            current = evaluate()
            if meets_goal(current):
                reached_goal = True
                save('goal.pt')
                break
    save()
    print(json.dumps(dict(kind='finished', completed_updates=step,
                         goal_reached=reached_goal,
                         resumed_from=checkpoint['step'] if checkpoint else None)), flush=True)


if __name__ == '__main__':
    main()
