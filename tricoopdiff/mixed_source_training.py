"""Shared-model SDPI/BiT/BiAnchor training without sharing incompatible flows."""

import hashlib
import json
import math

import numpy as np
import torch

from .bianchor_source import SOURCE_NAMES
from .condition_augmentation import ConditionFeatureDropout
from .conditioned_consensus_train import supervised_condition_batch
from .sdpi_flow import SDPIFlow
from .train import TriCoopDiffTrainer


FORMAT = 'anchored_consensus_training_v12_three_sources'
POLICY = 'least_updated_actual_source_sdpi_bitseed_bianchor_v1'
FULL = ConditionFeatureDropout(0., 0., 0., 0., 0.)
CONTRACT = dict(policy=POLICY, free_flow=SDPIFlow.version,
                endpoint_protocol='online_bitseed_training_c1_m1_v1',
                endpoint_steps=20, residual_weight=1e-3,
                selection='bianchor_development_loss_with_separate_free_source_reports_v1',
                loss_weights=dict(sdpi=1., bitseed=1., bianchor=1.))


def validate_source_training(value, global_step):
    if (not isinstance(value, dict) or value.get('contract') != CONTRACT
            or set(value) != {'contract', 'updates', 'fallbacks'}):
        raise ValueError('unsupported three-source training contract')
    counts = value['updates']
    if (not isinstance(counts, dict) or set(counts) != set(SOURCE_NAMES)
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != global_step
            or type(value['fallbacks']) is not int or not 0 <= value['fallbacks'] <= counts['sdpi']):
        raise ValueError('invalid actual-source update counts')


class FreeSourceObjective:
    """Reuse the production flow loss, owning neither optimizer nor EMA."""

    _training_state = TriCoopDiffTrainer._training_state
    _forward_state = TriCoopDiffTrainer._forward_state
    compute_loss = TriCoopDiffTrainer.compute_loss

    def __init__(self, model):
        self.model = model
        self.device = next(model.trifuse.parameters()).device
        self.diffusion = SDPIFlow(inference_steps=20)
        self.torsion_expert_aux_weight = 0.
        self.residual_weight = CONTRACT['residual_weight']


def prepare_free_batch(batch, requested_source, endpoint_store, *, seed):
    """Construct a real source-specific coupling; fallback is explicitly SDPI."""
    if requested_source not in SOURCE_NAMES[:2]:
        raise ValueError('free flow only supports SDPI and BiTSeed')
    actual = requested_source
    coupling = None
    if requested_source == 'bitseed':
        coupling, record = endpoint_store.coupling(batch['case'], batch=batch)
        actual = record['actual_source']
        if actual not in SOURCE_NAMES[:2]:
            raise ValueError('endpoint returned an unsupported actual source')
    else:
        # SDPI draws via NumPy/SciPy; isolate its seed so resume needs only the
        # existing torch RNG and case cursor, not a hidden global NumPy stream.
        state = np.random.get_state()
        try:
            np.random.seed(seed % (2**32))
            coupling = SDPIFlow.sample_coupling(batch['e3_data']['coords'], batch['poi_data']['coords'],
                                                batch['ligand_data'])
        finally:
            np.random.set_state(state)
    conditions, audit = supervised_condition_batch(batch, seed=seed, dropout=FULL)
    return {**batch, 'source': actual, 'coupling_override': coupling, 'conditions': conditions}, {
        'requested_source': requested_source, 'actual_source': actual,
        'fallback': actual != requested_source, 'condition_audit': audit}


def evaluate_free_sources(trainer, entries, load, endpoint_store, *, seed):
    """Fixed free-flow cohorts; exclude failed BiT endpoints from BiT metrics.

    Preserve the training RNG. Separate losses are diagnostics, not values to
    average with the different anchored objective or use for benchmark ranking.
    """
    device = next(trainer.model.trifuse.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    rows, cohort = [], []
    with torch.random.fork_rng(devices=devices), trainer.ema_mode():
        for entry in entries:
            e3, poi, ligand, _ = load(entry)
            batch = dict(case=entry['case'], e3_data=e3, poi_data=poi, ligand_data=ligand)
            for requested in SOURCE_NAMES[:2]:
                derived = int.from_bytes(hashlib.sha256(
                    f'{seed}:free-validation:{requested}:{entry["case"]}'.encode()).digest()[:8], 'big')
                torch.manual_seed(derived)
                prepared, audit = prepare_free_batch(batch, requested, endpoint_store, seed=derived)
                if prepared['coupling_override'] is None:
                    prepared['coupling_override'] = trainer.free_objective.diffusion.sample_coupling(
                        e3['coords'], poi['coords'], ligand,
                        no_torsion=bool(getattr(trainer.model.expert.model, 'no_torsion', False)))
                coupling = {k: torch.as_tensor(v).detach().cpu().tolist()
                            for k, v in prepared['coupling_override'].items()}
                cohort.append(dict(case=entry['case'], requested=requested,
                                   actual=prepared['source'], coupling=coupling))
                if audit['fallback']:
                    rows.append(dict(case=entry['case'], requested_source=requested,
                                     actual_source=prepared['source'], status='fallback'))
                    continue
                losses = [trainer.evaluate_free(**prepared, t_override=t)['loss_total']
                          for t in (.25, .5, .75)]
                mean = sum(losses) / len(losses)
                if not math.isfinite(mean):
                    raise FloatingPointError('nonfinite free-source validation')
                rows.append(dict(case=entry['case'], requested_source=requested,
                                 actual_source=prepared['source'], status='evaluated', mean_loss=mean))
    means, counts = {}, {}
    for source in SOURCE_NAMES[:2]:
        values = [r['mean_loss'] for r in rows if r['requested_source'] == source and r['status'] == 'evaluated']
        if not values:
            raise ValueError(f'validation has no real {source} samples')
        means[source], counts[source] = sum(values) / len(values), len(values)
    return dict(mean_loss_by_source=means, evaluated_by_source=counts, cases=rows,
                cohort_sha256=hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest())
