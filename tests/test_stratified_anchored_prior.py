"""Mixed candidate pools must not confuse continuous and discrete support."""

from pathlib import Path
import sys

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from tricoopdiff.stratified_anchored_prior import stratified_reference_schedule, sample_stratified_component


def test_one_reference_sixteen_conformers_supports_thirtytwo_mixed_walkers():
    def draw():
        return stratified_reference_schedule(1, 16, 32, .5, generator=torch.Generator().manual_seed(7315))
    rows = draw()
    assert rows == draw() and len(rows) == 32
    chemical = [(r, c) for r, c, component in rows if component == 'chemical']
    assert len(chemical) == len(set(chemical)) == 16
    assert sum(component == 'uniform' for _, _, component in rows) == 16
    assert all(c is None for _, c, component in rows if component == 'uniform')


def test_discrete_shortfall_is_not_silently_changed_to_uniform():
    try:
        stratified_reference_schedule(1, 2, 32, .5, generator=torch.Generator().manual_seed(7315))
    except ValueError as error:
        assert 'budget' in str(error)
    else:
        raise AssertionError('chemical component was silently discarded')


def test_explicit_chemical_cycles_keep_mass_and_exhaust_before_repeating():
    rows = stratified_reference_schedule(12, 1, 32, .5,
        generator=torch.Generator().manual_seed(7315), allow_repeats=True)
    chemical = [(r, c) for r, c, component in rows if component == 'chemical']
    assert len(chemical) == 16 and len(set(chemical)) == 12
    assert sum(component == 'uniform' for _, _, component in rows) == 16
    assert max(chemical.count(pair) for pair in chemical) == 2


def test_exact_prediction_reuse_does_not_merge_references_or_nearby_states():
    from tricoopdiff.stratified_anchored_prior import exact_prediction_key
    state = torch.zeros(3, dtype=torch.float64)
    key = exact_prediction_key(0, state)
    assert key == exact_prediction_key(0, state.clone())
    assert key != exact_prediction_key(1, state)
    other = state.clone()
    other[0] = 1e-12
    assert key != exact_prediction_key(0, other)
    assert key != exact_prediction_key(0, state.float())


def test_stratification_preserves_component_laws_and_binary_template_origin():
    from test_anchored_torsion import fixture
    from tricoopdiff.anchored_flow import AnchoredTorsionFlow
    from tricoopdiff.bounded_binary_flow import BoundedBinaryTorsionFlow
    from tricoopdiff.chemical_prior_contract import chemical_prior_config, conditioned_chemical_prior
    strict = AnchoredTorsionFlow(*fixture())
    flow = BoundedBinaryTorsionFlow(strict, (0, 1, 2), (5, 6, 7),
        translation_bounds=[.5, .5], rotation_bounds=[.2, .2])
    edges = torch.stack((torch.arange(7), torch.arange(1, 8)))
    conformers = strict.ligand[None]
    config = chemical_prior_config(uniform_probability=.5)
    for component in ('chemical', 'uniform'):
        state, audit = sample_stratified_component(flow, edges, conformers, component,
            generator=torch.Generator().manual_seed(7315), config=config)
        if component == 'chemical':
            expected, _ = conditioned_chemical_prior(flow, edges, conformers,
                generator=torch.Generator().manual_seed(7315), config=chemical_prior_config())
        else:
            expected = flow.sample_prior(generator=torch.Generator().manual_seed(7315))
            expected[3:] = 0
        assert torch.equal(state, expected)
        assert state[3:].count_nonzero() == 0
        assert audit['config'] == config and audit['component'] == component
