"""DeepTernary model-top1 selection must be deterministic and contained."""

import csv
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from refine_deepternary_official import model_top1_prediction


def write_summary(path, case, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('seed', 'pred_p2_rmsd', 'complex_pred'))
        writer.writeheader(); writer.writerows(rows)


def test_model_top1_uses_surrogate_and_stable_seed_tie_break(tmp_path):
    case = 'fixture'
    case_dir = tmp_path / case; case_dir.mkdir(parents=True)
    predictions = []
    for seed in range(3):
        path = case_dir / f'complex_pred_{case}_{seed}.pdb'; path.write_text('MODEL\n')
        predictions.append(path)
    write_summary(case_dir / f'summary_{case}.csv', case, [
        {'seed': 2, 'pred_p2_rmsd': 1., 'complex_pred': predictions[2]},
        {'seed': 1, 'pred_p2_rmsd': 1., 'complex_pred': predictions[1]},
        {'seed': 0, 'pred_p2_rmsd': 3., 'complex_pred': predictions[0]},
    ])
    prediction, seed = model_top1_prediction(tmp_path, case, expected_candidates=3)
    assert seed == 1 and prediction == predictions[1].resolve()


def test_model_top1_rejects_path_escape(tmp_path):
    case = 'fixture'
    case_dir = tmp_path / case; case_dir.mkdir(parents=True)
    outside = tmp_path / 'outside.pdb'; outside.write_text('MODEL\n')
    write_summary(case_dir / f'summary_{case}.csv', case, [
        {'seed': 0, 'pred_p2_rmsd': 1., 'complex_pred': outside},
    ])
    try:
        model_top1_prediction(tmp_path, case)
    except ValueError as error:
        assert 'escapes' in str(error)
    else:
        raise AssertionError('official prediction path escaped its audited case directory')


def test_model_top1_checks_candidate_count(tmp_path):
    case = 'fixture'
    case_dir = tmp_path / case; case_dir.mkdir(parents=True)
    prediction = case_dir / f'complex_pred_{case}_0.pdb'; prediction.write_text('MODEL\n')
    write_summary(case_dir / f'summary_{case}.csv', case, [
        {'seed': 0, 'pred_p2_rmsd': 1., 'complex_pred': prediction},
    ])
    try:
        model_top1_prediction(tmp_path, case, expected_candidates=40)
    except ValueError as error:
        assert 'expected 40 candidates, found 1' in str(error)
    else:
        raise AssertionError('candidate-count protocol mismatch was accepted')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_model_top1_uses_surrogate_and_stable_seed_tie_break(root / 'tie')
        test_model_top1_rejects_path_escape(root / 'escape')
        test_model_top1_checks_candidate_count(root / 'count')
    print('DeepTernary model-top1 refinement entry test passed')
