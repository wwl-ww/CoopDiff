"""Check deterministic quadrature against analytic and Monte Carlo references."""

from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def test_import_preserves_numpy_rng():
    before = np.random.get_state()
    from tricoopdiff.diffdock.utils import torus
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert torus.SCORE_NORM_VERSION == 'wrapped_normal_quadrature_v1'


def test_score_norm_matches_references():
    from tricoopdiff.diffdock.utils import torus
    rng = np.random.default_rng(42)
    sigmas = np.array([.01, .1, .5, 1., 2., 3.14])
    estimate = torus.score_norm(sigmas)
    # Lookup uses the nearest log-sigma row; compare to that actual sigma.
    indices = np.round(np.clip(
        (np.log(sigmas/np.pi)-np.log(torus.SIGMA_MIN)) /
        np.log(torus.SIGMA_MAX/torus.SIGMA_MIN)*torus.SIGMA_N, 0, torus.SIGMA_N)).astype(int)
    effective = torus.sigma[indices]
    assert np.allclose(estimate[:2], 1/effective[:2]**2, rtol=1e-4)
    draws = rng.normal(size=(200000, len(sigmas))) * effective
    mc = np.mean(torus.score(draws, effective)**2, axis=0)
    assert np.allclose(estimate, mc, rtol=.025), (estimate, mc)
    again = torus.deterministic_score_norm(torus.p_, torus.score_, torus.x, chunk_size=17)
    assert np.array_equal(again, torus.score_norm_)


def test_checkpoint_normalization_provenance():
    from tricoopdiff.diffdock.utils import torus
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert torus.validate_score_norm_version({}) == 'legacy_unrecorded'
        assert len(caught) == 1 and 'cannot be recovered' in str(caught[0].message)
    assert torus.validate_score_norm_version({
        'torus_score_norm_version': torus.SCORE_NORM_VERSION,
    }) == torus.SCORE_NORM_VERSION
    try:
        torus.validate_score_norm_version({'torus_score_norm_version': 'wrong'})
    except ValueError:
        pass
    else:
        raise AssertionError('known normalization mismatches must fail')


if __name__ == '__main__':
    test_import_preserves_numpy_rng()
    test_score_norm_matches_references()
    test_checkpoint_normalization_provenance()
    print('Deterministic torus normalization tests passed')
