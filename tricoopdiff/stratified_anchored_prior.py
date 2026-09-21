"""Fixed-budget mixture scheduling without consuming discrete slots for uniform draws."""

import math

import torch

from .binary_chart_bank import reference_conformer_schedule
from .chemical_prior_contract import chemical_prior_config, conditioned_chemical_prior, validate_prior_config, MIXTURE_POLICY


STRATIFIED_POLICY = 'stratified_chemical_uniform_reference_pool_v1'
CYCLIC_POLICY = 'stratified_chemical_cycles_uniform_reference_pool_v2'


def stratified_reference_schedule(references, conformers, walkers, uniform_probability, *, generator,
                                  allow_repeats=False):
    """Randomize balanced mixture labels; preserve each walker's prior marginal.

    Stochastic rounding preserves the requested expected component fraction for
    arbitrary budgets. Chemical pairs remain unique; uniform torsion draws have
    continuous support and do not use a discrete conformer slot. A genuinely
    insufficient chemical pool is rejected unless repeated cycles are explicitly
    allowed. Each cycle is independently shuffled; neither component mass nor
    per-walker marginal reference/conformer probability changes.
    """
    if (any(type(value) is not int or value < 1 for value in (references, conformers, walkers))
            or not isinstance(uniform_probability, (int, float))
            or not math.isfinite(uniform_probability) or not 0 < uniform_probability < 1):
        raise ValueError('stratified schedule requires positive counts and an interior mixture probability')
    expected = walkers * uniform_probability
    uniform_count = math.floor(expected)
    fraction = expected - uniform_count
    if fraction:
        uniform_count += int(float(torch.rand((), generator=generator, device=generator.device)) < fraction)
    chemical_count = walkers - uniform_count
    capacity = references * conformers
    if chemical_count > capacity and not allow_repeats:
        raise ValueError('chemical budget exceeds distinct binary-reference/conformer combinations')
    chemical = []
    while len(chemical) < chemical_count:
        chemical.extend(reference_conformer_schedule(
            references, conformers, min(capacity, chemical_count - len(chemical)), generator=generator))
    rows = [(reference, conformer, 'chemical') for reference, conformer in chemical]
    order = torch.randperm(references, generator=generator, device=generator.device).cpu().tolist()
    rows.extend((order[index % references], None, 'uniform') for index in range(uniform_count))
    shuffle = torch.randperm(walkers, generator=generator, device=generator.device).cpu().tolist()
    return [rows[index] for index in shuffle]


def exact_prediction_key(reference_index, initial_parameters):
    """Memoize deterministic inference only within one fixed model/case/protocol.

    Different references must never share a result, even when parameters agree.
    Exact equality is intentional: no geometric rounding or approximate merging.
    """
    if initial_parameters.ndim != 1 or not torch.isfinite(initial_parameters).all():
        raise ValueError('prediction reuse requires a finite flat parameter state')
    return reference_index, str(initial_parameters.dtype), tuple(initial_parameters.detach().cpu().tolist())


def sample_stratified_component(flow, edges, conformers, component, *, generator, config):
    """Sample the scheduled component with unchanged checkpoint component laws."""
    validate_prior_config(config)
    if config is None or config['policy'] != MIXTURE_POLICY or component not in ('chemical', 'uniform'):
        raise ValueError('stratified component requires an explicit mixed-prior checkpoint')
    if component == 'chemical':
        conditional_config = chemical_prior_config(seed=config['seed'], count=config['count'])
        state, audit = conditioned_chemical_prior(flow, edges, conformers,
                                                  generator=generator, config=conditional_config)
    else:
        from .single_anchor_flow import SingleAnchorTorsionFlow
        from .bounded_binary_flow import BoundedBinaryTorsionFlow
        state = flow.sample_prior(generator=generator)
        count = len(flow.plan.bonds)
        offsets = 6 if isinstance(flow, SingleAnchorTorsionFlow) else (
            12 if isinstance(flow, BoundedBinaryTorsionFlow) else 0)
        state[count:count + offsets] = 0
        audit = {'policy': MIXTURE_POLICY, 'component': 'uniform', 'conformer_index': None,
                 'conformer_count': len(conformers), 'transferred_torsions': 0,
                 'binary_offset_prior': 'template_origin'}
    from .chemical_prior_contract import apply_pose_prior
    apply_pose_prior(flow, state, generator=generator, config=config)
    audit['binary_offset_prior'] = config['binary_offset_prior']
    return state, {**audit, 'config': dict(config), 'sampling_design': STRATIFIED_POLICY}
