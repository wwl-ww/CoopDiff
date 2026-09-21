"""Fixed-cohort development validation; not an oracle benchmark or ranker."""

import hashlib
import json
import math

import torch

from .anchored_training import prepare_augmented_anchored_sample
from .anchored_tasks import prepare_task_sample
from .chemical_prior_contract import apply_training_prior


VALIDATION_POLICY = 'ema_fixed_case_prior_times_mean_loss_v1'
VALIDATION_TIMES = (.25, .5, .75)


def validate_index_pair(training, validation):
    """Check PDB disjointness against the unfiltered training population too."""
    left, right = training.get('metadata', {}), validation.get('metadata', {})
    if left.get('split') != 'train' or right.get('split') != 'val':
        raise ValueError('explicit train and val split metadata required')
    if (not left.get('source_index_sha256')
            or right.get('disjoint_from_index_sha256') != left['source_index_sha256']):
        raise ValueError('validation was not checked against the original full training index')
    if not training.get('entries') or not validation.get('entries'):
        raise ValueError('empty train/validation index')
    train_ids = {row['case'].split('_')[0].upper() for row in training['entries']}
    val_ids = {row['case'].split('_')[0].upper() for row in validation['entries']}
    original_train = set(right.get('excluded_training_pdb_ids', []))
    if not original_train or not train_ids <= original_train:
        raise ValueError('validation provenance omits training PDB IDs')
    excluded = set(left.get('excluded_benchmark_pdb_ids', []))
    if not excluded or excluded != set(right.get('excluded_benchmark_pdb_ids', [])):
        raise ValueError('train/validation benchmark exclusions disagree')
    if (train_ids & val_ids or val_ids & original_train
            or (train_ids | val_ids) & excluded):
        raise ValueError('train/validation/benchmark PDB overlap')
    return {'train_pdbs': len(train_ids), 'validation_pdbs': len(val_ids),
            'original_training_pdbs': len(original_train), 'overlap': []}


def evaluate_panel(trainer, entries, load, *, index_sha256, seed, prior_library=None):
    """Fixed augmentation/prior/times, equal case weight, explicit data rejections.

    Model errors are fatal. A changing eligible cohort cannot improve selection
    by dropping difficult examples; the trainer verifies its fingerprint.
    """
    rows, cohort = [], []
    with trainer.ema_mode():
        for entry in entries:
            e3, poi, ligand, _ = load(entry)
            case = entry['case']
            derived_seed = int.from_bytes(hashlib.sha256(f'{seed}:validation:{case}'.encode()).digest()[:8], 'big')
            generator = torch.Generator(device=ligand['coords'].device).manual_seed(derived_seed)
            try:
                if getattr(trainer, 'conditioning_task', 'dual_binary') == 'dual_binary':
                    sample = prepare_augmented_anchored_sample(
                        e3, poi, ligand, generator=generator, evidence_id=f'validation:{case}',
                        **({'binary_bounds': trainer.binary_bounds} if trainer.binary_bounds else {}))
                else:
                    sample = prepare_task_sample(trainer.conditioning_task,
                        e3, poi, ligand, generator=generator, evidence_id=f'validation:{case}',
                        binary_bounds=trainer.binary_bounds)
                if getattr(trainer, 'prior_config', None) is not None and sample is not None:
                    if prior_library is None:
                        raise RuntimeError('chemical-prior validation requires its fixed library')
                    sample = apply_training_prior(sample, ligand, prior_library.get(case, ligand['mol']),
                        generator=generator, config=trainer.prior_config)
            except ValueError as error:
                sample, reason = None, str(error)
            else:
                reason = 'no_chemical_cut_pair'
            if sample is None:
                row = {'case': case, 'status': 'rejected', 'reason': reason}
                rows.append(row)
                cohort.append(row)
                continue
            losses = [trainer.evaluate(sample, e3, poi, ligand, time=time, dt=.05)
                      for time in VALIDATION_TIMES]
            mean = sum(loss['loss_total'] for loss in losses) / len(losses)
            if not math.isfinite(mean):
                raise FloatingPointError(f'{case}: nonfinite validation loss')
            rows.append({'case': case, 'status': 'evaluated', 'mean_loss': mean, 'losses': losses})
            cohort.append({'case': case, 'status': 'evaluated',
                           'atom_maps': sample.audit['selected_atom_maps'],
                           'torsion_indices': list(sample.flow.plan.original_indices),
                           'prior_angles': sample.prior_angles.detach().cpu().tolist()})
    evaluated = [row['mean_loss'] for row in rows if row['status'] == 'evaluated']
    if not evaluated:
        raise ValueError('no eligible validation samples; cannot select a checkpoint')
    return {'policy': VALIDATION_POLICY, 'index_sha256': index_sha256, 'seed': seed,
            'binary_metric': trainer.binary_metric, 'conditioning_task': trainer.conditioning_task,
            'times': list(VALIDATION_TIMES), 'global_step': trainer.global_step,
            'mean_loss': sum(evaluated) / len(evaluated), 'evaluated': len(evaluated),
            'rejected': len(rows) - len(evaluated), 'cases': rows,
            'cohort_sha256': hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()}
