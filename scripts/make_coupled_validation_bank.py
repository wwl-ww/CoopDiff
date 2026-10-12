#!/usr/bin/env python
"""Freeze paired same-center held-out starts: 8 cases x 2 centers = 16 states.

Center 0 uses case-derived seed base 75; center 1 uses base 76. This records
explicit new reconstruction seeds, not a claim to recover the old launcher.
No model, optimizer, native-quality selection or new center generation is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

import tricoopdiff.diffdock_compat
from train_entry import TernaryDataLoader, _load_score_model_args, case_seed, validation_metric_context
from train_distributed import audit_coupled_centers
from run_coupled_small_sample import make_bank, center_hash
from tricoopdiff.runtime import atomic_json, atomic_torch_save, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--center-bank', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--train-data', default='results/small_sample_seed73/train.json')
    parser.add_argument('--cache-dir', default='results/featurize_cache')
    parser.add_argument('--expert-model-dir', default='checkpoints/score_model')
    parser.add_argument('--native-root', default='/root/.local/share/coopdiff/TernaryDB_release/TernaryDB/pdbs')
    parser.add_argument('--deepternary-root', default='/root/.local/share/coopdiff/DeepTernary')
    parser.add_argument('--torsion-std', type=float, default=.15)
    parser.add_argument('--translation-std', type=float, default=.5)
    parser.add_argument('--rotation-std', type=float, default=.05)
    opt = parser.parse_args()
    os.chdir(ROOT)
    output = opt.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'validation_starts.pt').exists():
        raise FileExistsError('validation bank already exists; refusing to replace fixed states')
    centers = torch.load(opt.center_bank, map_location='cpu', weights_only=False)
    audit = audit_coupled_centers(centers, opt.cache_dir, ('train',))
    atomic_json(audit, output/'center_bank_audit.json')
    if audit['invalid_count']:
        raise ValueError('invalid predicted center bank; no validation starts generated')
    loader = TernaryDataLoader(opt.train_data, _load_score_model_args(opt.expert_model_dir),
        shuffle=False, device='cpu', cache_dir=opt.cache_dir, cache_read_only=True)
    if len(loader.entries) != 8 or len(loader) != 8 or set(centers['train']) != set(loader.entries_by_case):
        raise ValueError('paired validation requires exactly the 8 training cases')
    if any(len(records)!=2 for records in centers['train'].values()):
        raise ValueError('paired validation requires two genuine centers per case')
    banks = {'same_complex':{}}
    for batch in loader:
        case = batch['case']
        context = validation_metric_context(loader.entries_by_case[case], opt.native_root)
        context['deepternary_root'] = opt.deepternary_root
        values = []
        for center_index, center in enumerate(centers['train'][case]):
            if center['sha256'] != center_hash(center['center']) or center.get('native_information_used') is not False:
                raise ValueError(f'{case}: center identity or native-free provenance invalid')
            seed_base = 75 + center_index
            seed_label = f'{case}:same_complex:heldout_perturbation:center{center_index}'
            seed = case_seed(seed_label, seed_base)
            bank = make_bank(batch, seed, context, center, opt)
            bank.update(center_index=center_index, validation_seed_base=seed_base,
                        seed_derivation_label=seed_label)
            values.append(bank)
        banks['same_complex'][case] = values
        print(json.dumps({'event':'case_frozen','case':case,'state_count':2,
            'seeds':[value['seed'] for value in values]}),flush=True)
    destination = output/'predicted_centers.pt'
    if destination.exists():
        if sha256_file(destination) != sha256_file(opt.center_bank):
            raise ValueError('output center bank differs from requested center bank')
    else:
        # Preserve byte identity because training checkpoints pin bank hashes.
        import shutil
        shutil.copyfile(opt.center_bank, destination)
    atomic_torch_save(banks, output/'validation_starts.pt')
    atomic_json({'contract':'paired_two_shared_centers_same_case_v1',
        'groups':['same_complex'],'case_count':8,'state_count':16,
        'center_indices':[0,1],'seed_bases':[75,76],
        'seed_derivation':'case_seed(case:same_complex:heldout_perturbation:center{index},75+index)',
        'seed_layout':'paired: center0/base75 and center1/base76',
        'historical_seed_identity_claimed':False,
        'source_center_bank':str(opt.center_bank.resolve()),
        'generator_entry_sha256':sha256_file(__file__),
        'perturb_entry_sha256':sha256_file(ROOT/'scripts/run_coupled_small_sample.py'),
        'center_bank_sha256':sha256_file(destination),
        'validation_bank_sha256':sha256_file(output/'validation_starts.pt'),
        'train_index_sha256':sha256_file(opt.train_data),
        'perturbation':{'torsion_std':opt.torsion_std,'translation_std':opt.translation_std,
                        'rotation_std':opt.rotation_std}}, output/'validation_bank_manifest.json')
    print(json.dumps({'event':'complete','case_count':8,'state_count':16,'cuda_initialized':torch.cuda.is_initialized()}),flush=True)


if __name__=='__main__':
    main()
