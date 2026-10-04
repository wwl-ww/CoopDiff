"""Tests for the learned-score diversity audit."""

from pathlib import Path
import importlib.util
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    'audit_initial_selector_diversity',
    ROOT / 'scripts/audit_initial_selector_diversity.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
from tricoopdiff.ranking.ranking_inference import StateQualityRanker


def test_diversity_audit_keeps_zero_penalty_as_exact_control(tmp_path=None):
    import tempfile
    context = tempfile.TemporaryDirectory() if tmp_path is None else None
    root = Path(context.name) if context is not None else tmp_path
    model = StateQualityRanker(14)
    checkpoint = root / 'best.pt'
    torch.save({'architecture': 'initial_geometry_selector_v1',
                'pool_size': 256, 'selection_budget': 64,
                'model_config': model.config, 'state_dict': model.state_dict(),
                'mean': torch.zeros(14), 'scale': torch.ones(14)}, checkpoint)
    features = torch.arange(256).float().unsqueeze(1).repeat(1, 14) / 255
    dockq = torch.linspace(0, 1, 256)
    cases = [{'case': '1ABC_A_B_LIG', 'features': features, 'dockq': dockq,
              'sources': ['bitseed'] * 192 + ['sdpi'] * 64}]
    rows = audit.evaluate_checkpoint(checkpoint, cases, (0., .1))
    assert len(rows) == 2 and rows[0]['penalty'] == 0
    with torch.no_grad():
        scores = model(features)['q_hat']
    expected = dockq[torch.argsort(scores, descending=True, stable=True)[:64]].max()
    assert abs(rows[0]['selected_best'] - float(expected)) < 1e-8
    if context is not None:
        context.cleanup()


if __name__ == '__main__':
    test_diversity_audit_keeps_zero_penalty_as_exact_control()
