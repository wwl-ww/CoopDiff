"""Tests for empirical ensemble free energy and cooperativity analysis."""

import numpy as np


from tricoopdiff.ensemble_analysis import (
    analyze_ensemble,
    cooperativity_from_ddg,
    cooperativity_from_kd,
    cooperativity_ranking,
    energy_population_association,
)


def test_report_diversity_keeps_rigid_pose_displacement():
    from tricoopdiff.evaluate import compute_ensemble_diversity

    pose = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    result = compute_ensemble_diversity([pose, pose + [10., 0., 0.]])
    assert result['mean_pairwise_rmsd'] == 10.
    assert result['n_clusters'] == 2
    assert result['coordinate_frame'] == 'fixed_e3_v1'


def test_ensemble_report_computes_pair_distances_once():
    from unittest.mock import patch
    from tricoopdiff import ensemble_analysis

    with patch.object(ensemble_analysis, 'pairwise_pose_rmsd',
                      wraps=ensemble_analysis.pairwise_pose_rmsd) as distances:
        report = analyze_ensemble([_pose(0.), _pose(5.)])
    assert distances.call_count == 1 and report['n_clusters'] == 2


def _pose(offset):
    return np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]) + offset


def test_ensemble_clusters_and_population_free_energy():
    poses = [_pose(0.0), _pose(0.1), _pose(0.2), _pose(5.0)]
    report = analyze_ensemble(poses, rmsd_cutoff=1.0)
    assert report['n_clusters'] == 2
    assert report['clusters'][0]['count'] == 3
    assert report['clusters'][0]['relative_free_energy_kcal_mol'] == 0.0
    assert report['clusters'][1]['relative_free_energy_kcal_mol'] > 0.0
    assert 'not absolute binding free energy' in report['free_energy_interpretation']


def test_cooperativity_conversions_and_ranking():
    assert cooperativity_from_kd(10.0, 2.0) == 5.0
    assert abs(cooperativity_from_ddg(0.0) - 1.0) < 1e-12
    assert cooperativity_from_ddg(-1.0) > 1.0
    ranking = cooperativity_ranking([-2., -1., 0., 1.], [8., 4., 2., 1.])
    assert abs(ranking['spearman_rho'] - 1.0) < 1e-12
    assert ranking['top_n_hit_rate'] == 1.0


def test_energy_population_uses_cluster_level_frequency():
    labels = [1, 1, 1, 2, 2, 3]
    energies = [-3.1, -3.0, -2.9, -2.0, -1.9, -1.0]
    report = energy_population_association(labels, energies)
    assert report['favorability_population_spearman_rho'] > 0.9
    assert report['dominant_cluster_energy_rank'] == 1


def test_energy_population_marks_undefined_correlation():
    report = energy_population_association([1, 1, 2, 2], [-2., -2., -1., -1.])
    assert report['favorability_population_spearman_rho'] is None
