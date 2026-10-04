"""Check per-seed metric provenance without running neural models."""

from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tricoopdiff.evaluate import evaluate_model
from tricoopdiff.deepternary_metrics import compute_fullatom_dockq, _ligand_records


def test_walker_metrics_preserve_seeds_and_exact_failures():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    samples = [dict(ligand_coords=coords, poi_coords=coords,
                    valid=valid, initialization={}) for valid in (True, False, True)]
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    exact = {'dockq': .5, 'fnat': .4, 'lrms': 2., 'irms': 1.}
    with ExitStack() as stack:
        stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=samples))
        stack.enter_context(patch('tricoopdiff.deepternary_metrics.compute_fullatom_dockq',
                                  side_effect=[exact, RuntimeError('metric failed')]))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_dockq', 'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.ensemble_analysis.analyze_ensemble', return_value={
            'mean_pairwise_rmsd': 0., 'std_pairwise_rmsd': 0., 'n_clusters': 1}))
        report = evaluate_model(
            None, data, data, data, None, initializer=object(), device='cpu',
            n_walkers=3, seed=7, native_complex_pdb='dummy', deepternary_dockq_paths={})
    assert report['ensemble']['invalid_seeds'] == [8]
    assert report['ensemble']['n_exact_dockq_walkers'] == 1
    assert [row['seed'] for row in report['walkers']] == [7, 9]
    assert report['walkers'][0]['dockq'] == .5
    assert 'metric failed' in report['walkers'][1]['exact_dockq_error']
    assert all('pred' not in row and 'pred_poi' not in row for row in report['walkers'])
    assert report['structure']['dockq_best'] == .5


def test_rank_features_are_saved_only_when_requested():
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    samples = [dict(ligand_coords=coords, poi_coords=coords, valid=True,
                    initialization={}, rank_features=np.array([1., 2., 3.]))]
    data = {'coords': torch.tensor(coords).unsqueeze(0),
            'edge_index': torch.tensor([[0, 1], [1, 0]])}
    conditions = {'contract': object()}
    with ExitStack() as stack:
        sampler = stack.enter_context(patch('tricoopdiff.sampler.sample_ensemble', return_value=samples))
        for name in ('compute_deepternary_dockq', 'compute_e3_interface_dockq',
                     'compute_poi_dockq', 'compute_poi_com_dist'):
            stack.enter_context(patch(f'tricoopdiff.evaluate.{name}', return_value=None))
        stack.enter_context(patch('tricoopdiff.ensemble_analysis.analyze_ensemble', return_value={
            'mean_pairwise_rmsd': 0., 'std_pairwise_rmsd': 0., 'n_clusters': 1}))
        report = evaluate_model(
            None, data, data, data, None, initializer=object(), device='cpu',
            n_walkers=1, return_rank_features=True, conditions=conditions,
        )
    assert report['walkers'][0]['rank_features'] == [1.0, 2.0, 3.0]
    assert sampler.call_args.kwargs['conditions'] is conditions


def test_unbound_metric_templates_receive_only_predicted_rigid_motion():
    def records(coords, chain):
        return ''.join(
            f'ATOM  {i:5d}  CA  ALA {chain}{i:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n'
            for i, (x, y, z) in enumerate(coords, 1))

    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([3., 4., 5.])
    template = np.vstack((coords + 10., [[12., 12., 12.]]))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, points, chain in (
                ('p1', coords, 'I'), ('p2', coords + 10., 'J'),
                ('m1', coords, 'A'), ('m2', template, 'B')):
            (root / f'{name}.pdb').write_text(records(points, chain))
        (root / 'lig.pdb').write_text(records(coords, 'A').replace('ATOM  ', 'HETATM'))
        native_path = root / 'native.pdb'
        native_path.write_text((root / 'm1.pdb').read_text() + (root / 'm2.pdb').read_text())

        def capture(predicted, native):
            lines = Path(predicted).read_text().splitlines()
            points = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])]
                               for line in lines if line.startswith('ATOM') and line[21] == 'B'])
            assert np.allclose(points, template @ rotation + translation, atol=1e-3)
            assert not any(line[21] in ('I', 'J') for line in lines)
            return .4, 1., 2., .5

        with patch.dict(sys.modules, {'DockQ.dockq_util': SimpleNamespace(cal_dockq=capture)}):
            result = compute_fullatom_dockq(
                coords, (coords + 10.) @ rotation + translation, coords,
                root / 'p1.pdb', root / 'p2.pdb', root / 'lig.pdb', native_path, root,
                metric_protein1_pdb=root / 'm1.pdb', metric_protein2_pdb=root / 'm2.pdb')
        assert result['dockq'] == .5


def test_sdf_metric_template_preserves_heavy_atom_order():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    expected = molecule.GetConformer().GetPositions()[:3]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'ligand.sdf'
        Chem.MolToMolFile(molecule, str(path))
        _, indices, coords = _ligand_records(path)
    assert len(indices) == 3
    assert np.allclose(coords, expected, atol=1e-3)


if __name__ == '__main__':
    test_walker_metrics_preserve_seeds_and_exact_failures()
    test_rank_features_are_saved_only_when_requested()
    test_unbound_metric_templates_receive_only_predicted_rigid_motion()
    test_sdf_metric_template_preserves_heavy_atom_order()
    print('Evaluation walker provenance test passed')
