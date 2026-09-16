"""Tests for future-value diversity accounting."""

from pathlib import Path
import importlib.util
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    'audit_future_selector_diversity',
    ROOT / 'scripts/audit_future_selector_diversity.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
from tricoopdiff.pose_ranking import StateQualityRanker


def test_future_diversity_audit_reports_rescued_basins():
    with tempfile.TemporaryDirectory() as directory:
        model = StateQualityRanker(14)
        checkpoint = Path(directory) / 'best.pt'
        torch.save({'architecture': 'future_geometry_selector_v1', 'target': 'final',
                    'pool_size': 256, 'selection_budget': 64,
                    'model_config': model.config, 'state_dict': model.state_dict(),
                    'mean': torch.zeros(14), 'scale': torch.ones(14)}, checkpoint)
        initial = torch.zeros(256); final = torch.zeros(256)
        final[17] = .4
        cases = [{'case': '1ABC_A_B_LIG',
                  'features': torch.randn(256, 14),
                  'initial': initial, 'final': final,
                  'union': torch.maximum(initial, final),
                  'sources': ['bitseed'] * 192 + ['sdpi'] * 64}]
        target, rows = audit.evaluate(checkpoint, cases, penalties=(0., .1))
    summary = audit.summarize(rows, penalties=(0., .1))
    assert target == 'final' and len(rows) == 2 and len(summary) == 2
    assert summary[0]['rescued_cases_available'] == 1


if __name__ == '__main__':
    test_future_diversity_audit_reports_rescued_basins()
