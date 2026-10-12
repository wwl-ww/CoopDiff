#!/usr/bin/env python
"""Candidate-driven ternary assembly: generate, train/resume, and predict."""

import argparse
from argparse import Namespace
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tricoopdiff.diffdock_compat  # noqa: E402,F401
from featurize_common import featurize_complex  # noqa: E402
from train_entry import _load_index, _load_score_model_args, TernaryDataLoader  # noqa: E402
from tricoopdiff.initialization.candidates import (  # noqa: E402
    load_assembly_candidate, generate_joint_matcha_ligand,
)
from tricoopdiff.models.assembly import (  # noqa: E402
    CANDIDATE_FRAMEWORK_VERSION, TernaryAssembly, TernaryAssemblyModel,
    production_model_config, CANDIDATE_GEOMETRY_CONTRACT, CANDIDATE_QUALITY_CONTRACT,
    CANDIDATE_CONDITION_CONTRACT,
)
from tricoopdiff.models.expert import FrozenMatcha  # noqa: E402
from tricoopdiff.runtime import (  # noqa: E402
    atomic_json, atomic_torch_save, capture_rng_state, restore_rng_state,
    sha256_file, preserve_rng,
)
from tricoopdiff.sdpi_flow import SDPIFlow  # noqa: E402
from tricoopdiff.train import AssemblyTrainer  # noqa: E402


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'train', 'predict'))
    parser.add_argument('--index', required=True)
    parser.add_argument('--bank', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--score-model-dir', default='checkpoints/score_model')
    parser.add_argument('--matcha-source-dir', default='results/matcha_qualification_source')
    parser.add_argument('--matcha-weights-dir', default='results/matcha_qualification_weights')
    parser.add_argument('--hdock-dir', default='results/hdocklite_install', type=Path)
    parser.add_argument('--fftw-lib-dir', default='/home/wl/micromamba/envs/proteinprep/lib')
    parser.add_argument('--candidates', type=int, default=5)
    parser.add_argument('--hdock-workers', type=int, default=16,
                        help='Parallel CPU docking processes; Matcha generation stays on one GPU')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--generator-steps-per-stage', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--node-dim', type=int, default=128)
    parser.add_argument('--near-target-frequency', type=int, default=4,
                        help='Every N updates add a zero/perturbed near-target task; 0 disables')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', choices=('hdock', 'learned'), default='hdock',
                        help='Keep HDOCK rank by default; learned scoring is an explicit uncalibrated experiment')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--native-root', type=Path,
                        help='Optional post-selection metrics only; never used by the predictor')
    opt = parser.parse_args(argv)
    if min(opt.candidates, opt.steps, opt.generator_steps_per_stage, opt.epochs, opt.hdock_workers) < 1:
        parser.error('candidate counts, steps and epochs must be positive')
    if opt.limit is not None and opt.limit < 1:
        parser.error('limit must be positive')
    if opt.near_target_frequency < 0:
        parser.error('near-target frequency must be nonnegative')
    if opt.mode == 'predict' and not opt.checkpoint:
        parser.error('predict requires --checkpoint')
    if opt.resume and opt.mode != 'train':
        parser.error('resume is a training-only operation')
    if opt.checkpoint and opt.mode != 'predict':
        parser.error('checkpoint is a prediction-only operation')
    return opt


def input_identity(entry):
    return {key: sha256_file(entry[key]) for key in ('protein1', 'protein2', 'ligand')}


def read_payload(path, device='cpu'):
    """Pin the loaded bytes and their digest, even if a producer publishes a new file."""
    contents = Path(path).read_bytes()
    return (torch.load(io.BytesIO(contents), map_location=device, weights_only=False),
            hashlib.sha256(contents).hexdigest())


def load_case(opt, entry, expert, score_args):
    # Offline mode prevents a cache miss from silently downloading models.
    data = featurize_complex(entry['case'], entry['protein1'], entry['protein2'],
        entry['ligand'], score_args, opt.device, cache_dir=opt.cache_dir)
    loader = Namespace(expert=expert, cache_dir=opt.cache_dir, device=torch.device(opt.device))
    e3, poi, ligand = preserve_rng(TernaryDataLoader._prepare_matcha)(loader, entry, data[:3])
    from tricoopdiff.metrics.dockq import _records, _featurized_ca_coords, _fit_rigid
    table = Chem.GetPeriodicTable()
    prepared = []
    for part, key in ((e3, 'protein1'), (poi, 'protein2')):
        lines, indices, atoms = _records(entry[key], 'ATOM')
        ca = _featurized_ca_coords(entry[key])
        rotation, translation = _fit_rigid(ca, part['coords'][0].cpu().numpy())
        elements = []
        for index in indices:
            line = lines[index]
            # Standard protein ATOM names: CA is carbon, not calcium.
            elements.append((line[76:78].strip() if len(line) >= 78 else '') or line[12:16].strip()[0])
        selected = [i for i, element in enumerate(elements) if element.upper() not in ('H', 'D')]
        prepared.append({**part,
            'steric_atoms': part['coords'].new_tensor(atoms[selected] @ rotation+translation)[None],
            'steric_ca_center': part['coords'][0].mean(0).detach(),
            'steric_radii': part['coords'].new_tensor([table.GetRvdw(elements[i].title()) for i in selected])})
    e3, poi = prepared
    return e3, poi, ligand, data[3]


def write_hdock_input(path, destination, chain, rotation):
    records = []
    identities = set()
    for line in Path(path).read_text().splitlines():
        key = (line[22:27], line[12:16])
        if (not line.startswith('ATOM') or line[16] not in (' ', 'A')
                or key in identities):
            continue
        identities.add(key)
        records.append(line)
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in records])
    if not len(records) or not np.isfinite(xyz).all():
        raise ValueError('HDOCK input must contain finite protein atoms')
    xyz = (xyz-xyz.mean(0)) @ rotation
    destination.write_text('\n'.join(l[:16]+' '+l[17:21]+chain+l[22:30]
        + ''.join(f'{v:8.3f}' for v in point)+l[54:]
        for l, point in zip(records, xyz)) + '\nEND\n')


def docking_job(opt, entry):
    """Independent CPU search, with exact input/binary resume identity."""
    directory = opt.bank/entry['case']
    directory.mkdir(parents=True, exist_ok=True)
    seed = (opt.seed+int(hashlib.sha256(entry['case'].encode()).hexdigest()[:8], 16)) % (2**31-1)
    identity = {'inputs': input_identity(entry), 'seed': opt.seed,
                'hdock_sha256': sha256_file(opt.hdock_dir/'hdock')}
    marker = directory/'docking_complete.json'
    if marker.exists() and (directory/'Hdock.out').is_file() and json.loads(marker.read_text()) == identity:
        return entry
    write_hdock_input(entry['protein1'], directory/'r.pdb', 'A', np.eye(3))
    write_hdock_input(entry['protein2'], directory/'l.pdb', 'B',
                      Rotation.random(random_state=seed).as_matrix())
    env = {**os.environ, 'LD_LIBRARY_PATH': opt.fftw_lib_dir,
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (directory/'hdock.log').open('w') as log:
        subprocess.run([str((opt.hdock_dir/'hdock').resolve()), 'r.pdb', 'l.pdb', '-out', 'Hdock.out'],
            cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=14400)
    atomic_json(identity, marker)
    return entry


def generate(opt, entries, expert, score_args):
    # Start bounded CPU searches ahead of the resident GPU generator. Do not
    # fork a process containing initialized CUDA models.
    pending = []
    for entry in entries:
        bank_path = opt.bank/entry['case']/'candidates.pt'
        if bank_path.exists():
            existing = torch.load(bank_path, map_location='cpu', weights_only=False)
            if existing['protocol'] != generation_protocol(opt, entry, expert):
                raise ValueError('candidate bank provenance changed; use a new bank directory')
            print(f"SKIP generated {entry['case']}", flush=True)
        else:
            pending.append(entry)
    with ThreadPoolExecutor(max_workers=opt.hdock_workers) as pool:
        futures = [pool.submit(docking_job, opt, entry) for entry in pending]
        for future in as_completed(futures):
            generate_case(opt, future.result(), expert, score_args)


def generation_protocol(opt, entry, expert):
    return {'format': CANDIDATE_FRAMEWORK_VERSION, 'inputs': input_identity(entry),
        'expert': expert.provenance, 'seed': opt.seed, 'count': opt.candidates,
        'steps_per_stage': opt.generator_steps_per_stage,
        'generator_sha256': sha256_file(ROOT/'tricoopdiff/initialization/candidates.py'),
        'hdock_sha256': sha256_file(opt.hdock_dir/'hdock')}


def generate_case(opt, entry, expert, score_args):
    """Main-thread CUDA generation, after the CPU search completes."""
    directory = opt.bank/entry['case']
    protocol = generation_protocol(opt, entry, expert)
    e3, poi, ligand, _ = load_case(opt, entry, expert, score_args)
    seed = (opt.seed+int(hashlib.sha256(entry['case'].encode()).hexdigest()[:8], 16)) % (2**31-1)
    with (directory/'export.log').open('w') as log:
        subprocess.run([str((opt.hdock_dir/'createpl').resolve()), 'Hdock.out', 'top.pdb',
            '-nmax', str(opt.candidates), '-complex', '-models'], cwd=directory,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    # A chemistry-only template for importing the protein frame; replaced
    # by joint generation before any candidate is saved or scored.
    mol = ligand['mol']
    template = Chem.AddHs(Chem.Mol(mol) if isinstance(mol, bytes) else Chem.Mol(mol))
    template.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(template, params) != 0:
        raise ValueError('template ETKDG failed')
    template = Chem.RemoveHs(template)
    with Chem.SDWriter(str(directory/'template.sdf')) as writer:
        writer.write(template)
    candidates = []
    for rank in range(1, opt.candidates+1):
        candidate = load_assembly_candidate(directory/f'model_{rank}.pdb', directory/'template.sdf',
            e3['coords'], poi['coords'], ligand, candidate_id=entry['case'], rank=rank,
            protein1_path=entry['protein1'], protein2_path=entry['protein2'])
        candidate = generate_joint_matcha_ligand(expert, e3, poi, ligand, candidate,
            seed=(seed+rank*1009) % (2**31-1), steps_per_stage=opt.generator_steps_per_stage)
        candidate['provenance']['template_sha256'] = candidate['provenance'].pop('ligand_pose_sha256')
        candidates.append({key: value.detach().cpu() if torch.is_tensor(value) else value
                           for key, value in candidate.items()})
    atomic_torch_save({'protocol': protocol, 'candidates': candidates}, directory/'candidates.pt')
    print(f"GENERATED {entry['case']} {len(candidates)} candidates", flush=True)


def load_bank(opt, entry, expert):
    path = opt.bank/entry['case']/'candidates.pt'
    bank, digest = read_payload(path)
    protocol = bank['protocol']
    if (protocol['format'] != CANDIDATE_FRAMEWORK_VERSION
            or protocol['inputs'] != input_identity(entry)
            or protocol['expert'] != expert.provenance):
        raise ValueError('candidate bank input, generator or format identity differs')
    ranks = [c['provenance']['protein_candidate_rank'] for c in bank['candidates']]
    if len(ranks) != protocol['count'] or sorted(ranks) != list(range(1, protocol['count']+1)):
        raise ValueError('candidate bank must contain exactly one pose per declared HDOCK rank')
    return bank['candidates'], digest


def model_for(opt, expert, input_dim, payload=None):
    config = (payload['model_config'] if payload else production_model_config(
        opt.node_dim, input_dim, True, expert_backend='matcha',
        expert_scalar_dim=expert.node_scalar_dim, expert_vector_dim=expert.node_vector_dim,
        candidate_driven=True))
    if (input_dim != config['input_dim'] or not config.get('candidate_driven')
            or config.get('candidate_geometry_contract') != CANDIDATE_GEOMETRY_CONTRACT
            or config.get('candidate_quality_contract') != CANDIDATE_QUALITY_CONTRACT
            or config.get('candidate_condition_contract') != CANDIDATE_CONDITION_CONTRACT):
        raise ValueError('checkpoint is not a compatible candidate-driven model')
    if payload and (payload['format'] != CANDIDATE_FRAMEWORK_VERSION
                    or payload['expert_identity'] != expert.provenance):
        raise ValueError('checkpoint format or frozen expert identity differs')
    model = TernaryAssemblyModel(expert, TernaryAssembly(**config), expert_input_grad=True,
                                 checkpoint_ternary=True).to(opt.device)
    if payload:
        model.trifuse.load_state_dict(payload['model_state'], strict=True)
    return model


def train(opt, entries, expert, score_args):
    if not opt.resume and (opt.output/'progress.pt').exists():
        raise ValueError('training output already contains a checkpoint; use --resume or a new directory')
    identity, tasks = {}, []
    for i, entry in enumerate(entries):
        candidates, digest = load_bank(opt, entry, expert)
        identity[entry['case']] = digest
        tasks.extend((i, j) for j in range(len(candidates)))
    # Never preload the full dataset onto CUDA. Retain at most two cases.
    case_cache = OrderedDict()

    def current_case(index):
        if index not in case_cache:
            if len(case_cache) >= 2:
                case_cache.popitem(last=False)
            entry = entries[index]
            case_cache[index] = (load_case(opt, entry, expert, score_args),
                                 load_bank(opt, entry, expert)[0])
        case_cache.move_to_end(index)
        return case_cache[index]
    payload = read_payload(opt.resume, opt.device)[0] if opt.resume else None
    if payload and (payload['bank_identity'] != identity or payload['steps'] != opt.steps or payload['seed'] != opt.seed):
        raise ValueError('resume bank, rollout steps or seed differs')
    if payload and payload['training_index_sha256'] != sha256_file(opt.index):
        raise ValueError('resume training index differs')
    training_options = {'quality_weight': .1, 'steric_weight': .05,
                        'quality_representation': CANDIDATE_QUALITY_CONTRACT,
                        'quality_target': 'native_crystal_log_geometry_v1',
                        'gradient_clip_norm': 10., 'node_dim': opt.node_dim,
                        'gradient_clip_contract': 'independent_motion_quality_groups_v1',
                        'near_target_frequency': opt.near_target_frequency,
                        'steric_contract': 'element_0p75_vdw_cross_and_ligand_gt3bonds_v1'}
    if payload and payload['training_options'] != training_options:
        raise ValueError('resume model width or training objective differs')
    model = model_for(opt, expert, current_case(0)[0][3], payload)
    feature_identity = {entry['case']: sha256_file(Path(opt.cache_dir)/(entry['case']+'.pt'))
                        for entry in entries}
    if payload and payload['feature_identity'] != feature_identity:
        raise ValueError('resume feature cache identity differs')
    trainer = AssemblyTrainer(model, SDPIFlow(inference_steps=opt.steps), lr=opt.lr,
        device=opt.device, residual_weight=0., steric_loss_weight=.05, effective_batch_size=1)
    epoch, position, updates = 0, 0, 0
    if payload:
        trainer.optimizer.load_state_dict(payload['optimizer_state'])
        restore_rng_state(payload['rng_state'])
        epoch, position, updates = payload['epoch'], payload['position'], payload['updates']
    model.train()
    while epoch < opt.epochs:
        permutation = np.random.RandomState(opt.seed+epoch).permutation(len(tasks))
        for cursor in range(position, len(tasks)):
            i, j = tasks[permutation[cursor]]
            entry = entries[i]
            (e3, poi, ligand, _), candidates = current_case(i)
            candidate = candidates[j]
            trainer.optimizer.zero_grad(set_to_none=True)
            near = ({'near_target_zero': (updates // opt.near_target_frequency) % 2 == 0,
                     'near_target_start_t': 1.}
                    if opt.near_target_frequency and updates % opt.near_target_frequency == 0 else {})
            result = trainer.compute_loss(e3, poi, ligand, case=entry['case'], source='hdock_matcha',
                coupling_override={'initial_state': {'ligand': candidate['ligand'],
                    'poi': candidate['poi'], 'body_frame': candidate['rotation']}}, **near)
            result['loss_total'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.trifuse.parameters()):
                raise FloatingPointError('nonfinite candidate training gradient; update aborted')
            norms = {group['name']: float(torch.nn.utils.clip_grad_norm_(
                group['params'], 10., error_if_nonfinite=True))
                for group in trainer.optimizer.param_groups}
            trainer.optimizer.step()
            updates += 1
            record = {'epoch': epoch, 'update': updates, 'case': entry['case'],
                      'candidate': j, 'loss': float(result['loss_total'].detach()),
                      'gradient_norms': norms,
                      'components': {key: value for key, value in result.items()
                                     if key.startswith('loss_') and key != 'loss_total'},
                      'target_projection_rmsd_angstrom': result.get('target_projection_rmsd_angstrom')}
            with (opt.output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            next_epoch, next_position = (epoch+1, 0) if cursor+1 == len(tasks) else (epoch, cursor+1)
            atomic_torch_save({'format': CANDIDATE_FRAMEWORK_VERSION,
                'model_config': model.trifuse.config, 'model_state': model.trifuse.state_dict(),
                'expert_identity': expert.provenance, 'optimizer_state': trainer.optimizer.state_dict(),
                'bank_identity': identity, 'steps': opt.steps, 'seed': opt.seed,
                'training_index_sha256': sha256_file(opt.index),
                'training_options': training_options,
                'feature_identity': feature_identity,
                'epoch': next_epoch, 'position': next_position, 'updates': updates,
                'rng_state': capture_rng_state()}, opt.output/'progress.pt')
            print(json.dumps(record), flush=True)
        epoch += 1
        position = 0


def predict(opt, entries, expert, score_args):
    payload, checkpoint_digest = read_payload(opt.checkpoint, opt.device)
    if payload['training_options'].get('quality_target') != 'native_crystal_log_geometry_v1':
        raise ValueError('retired reachable-target scorer; train the crystal-quality framework')
    if payload['steps'] != opt.steps:
        raise ValueError('deployment must use the trained refinement step count')
    if not payload.get('updates', 0):
        raise ValueError('prediction requires a trained candidate-framework checkpoint')
    model = None
    for entry in entries:
        e3, poi, ligand, input_dim = load_case(opt, entry, expert, score_args)
        candidates, bank_digest = load_bank(opt, entry, expert)
        if model is None:
            model = model_for(opt, expert, input_dim, payload)
        predictions = model.refine_candidates(e3, poi, ligand, candidates, steps=opt.steps)
        scores = [float(p['candidate_score']) for p in predictions]
        learned_winner = int(np.argmax(scores)) if payload.get('updates', 0) else None
        winner = (learned_winner if opt.selection == 'learned' else
                  min(range(len(predictions)), key=lambda i:
                      predictions[i]['provenance']['protein_candidate_rank']))
        directory = opt.output/entry['case']
        directory.mkdir(parents=True, exist_ok=True)
        selection_path = directory/'selection.json'
        if selection_path.exists() and json.loads(selection_path.read_text())['checkpoint_sha256'] != checkpoint_digest:
            raise ValueError('prediction output belongs to another checkpoint; use a new directory')
        atomic_torch_save([{key: value.cpu() if torch.is_tensor(value) else value
                           for key, value in p.items()} for p in predictions], directory/'predictions.pt')
        atomic_json({'case': entry['case'], 'candidate_scores': scores, 'selected_index': winner,
            'selection_policy': opt.selection, 'experimental_learned_index': learned_winner,
            'quality_trained_updates': payload.get('updates', 0), 'bank_sha256': bank_digest,
            'checkpoint_sha256': checkpoint_digest, 'native_used_for_selection': False,
            'quality_calibration_validated': False}, directory/'selection.json')
        # Export physical structures before optional scoring. Predictions and
        # winner were already fixed without access to a native complex.
        from tricoopdiff.metrics.dockq import (
            _records, _featurized_ca_coords, _fit_rigid, _replace_coords,
            compute_fullatom_dockq,
        )
        p1_lines, _, _ = _records(entry['protein1'], 'ATOM')
        p2_lines, p2_indices, p2_xyz = _records(entry['protein2'], 'ATOM')
        ca1, ca2 = _featurized_ca_coords(entry['protein1']), _featurized_ca_coords(entry['protein2'])
        frame_r, frame_t = _fit_rigid(e3['coords'][0].cpu().numpy(), ca1)
        metrics = []
        for index, prediction in enumerate(predictions):
            lig_xyz = prediction['ligand'][0].cpu().numpy() @ frame_r + frame_t
            poi_xyz = prediction['poi'][0].cpu().numpy() @ frame_r + frame_t
            rotation = prediction['rotation'][0].cpu().numpy().T @ frame_r
            translation = poi_xyz.mean(0)-ca2.mean(0) @ rotation
            protein2 = _replace_coords(p2_lines, p2_indices, p2_xyz @ rotation+translation, chain='B')
            mol = ligand['mol']
            mol = Chem.Mol(mol) if isinstance(mol, bytes) else Chem.Mol(mol)
            mol.RemoveAllConformers()
            conformer = Chem.Conformer(mol.GetNumAtoms())
            for atom, point in enumerate(lig_xyz):
                conformer.SetAtomPosition(atom, point.tolist())
            mol.AddConformer(conformer)
            with Chem.SDWriter(str(directory/f'candidate{index+1}_ligand.sdf')) as writer:
                writer.write(mol)
            ligand_lines = [l[:21]+'C'+l[22:] for l in Chem.MolToPDBBlock(mol).splitlines()
                            if l.startswith(('ATOM', 'HETATM'))]
            records = [l[:21]+'A'+l[22:] for l in p1_lines if l.startswith('ATOM')]
            records += [l for l in protein2 if l.startswith('ATOM')]+ligand_lines
            (directory/f'candidate{index+1}.pdb').write_text('\n'.join(records)+'\nEND\n')
            if opt.native_root:
                native_path = opt.native_root/entry['case']/'gt_complex.pdb'
                row = compute_fullatom_dockq(prediction['ligand'][0].cpu().numpy(),
                    prediction['poi'][0].cpu().numpy(), e3['coords'][0].cpu().numpy(),
                    entry['protein1'], entry['protein2'], entry['ligand'], native_path, ROOT/'DeepTernary',
                    pred_poi_rotation=prediction['rotation'][0].cpu().numpy())
                rmsd = torch.linalg.vector_norm(prediction['ligand']-ligand['coords'], dim=-1).square().mean().sqrt()
                row.update(candidate_index=index, ligand_crystal_rmsd_angstrom=float(rmsd),
                           ligand_rmsd_protocol='indexed_atom_order_e3_frame_no_ligand_alignment')
                metrics.append(row)
        if opt.native_root:
            atomic_json({'case': entry['case'], 'selected_index': winner,
                'selected': metrics[winner] if winner is not None else None,
                'candidates': metrics, 'native_used_for_selection': False}, directory/'metrics.json')
        print(f"PREDICTED {entry['case']} selected={winner}", flush=True)


def main(argv=None):
    opt = arguments(argv)
    os.environ['HF_HUB_OFFLINE'] = '1'
    torch.set_num_threads(4)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    entries = _load_index(opt.index)[0]
    if opt.limit:
        entries = entries[:opt.limit]
    if not entries or len({e['case'] for e in entries}) != len(entries):
        raise ValueError('index must contain nonempty unique cases')
    opt.output.mkdir(parents=True, exist_ok=True)
    opt.bank.mkdir(parents=True, exist_ok=True)
    lock_root = opt.bank if opt.mode == 'generate' else opt.output
    with (lock_root/(opt.mode+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another process is writing this pipeline directory') from error
        expert = FrozenMatcha(opt.matcha_source_dir, opt.matcha_weights_dir, opt.device,
                              frame_policy='backbone_rigid_transport_v1')
        score_args = _load_score_model_args(opt.score_model_dir)
        {'generate': generate, 'train': train, 'predict': predict}[opt.mode](opt, entries, expert, score_args)


if __name__ == '__main__':
    main()
