"""Fixed-checkpoint audit must respect actual head supervision."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_fullpool_ranker import evaluation_options


def test_quality_only_audit_does_not_use_future_head():
    checkpoint = {'architecture': 'structural_quality_ranker', 'epoch': 20, 'optimizer_steps': 640,
                  'metadata': {'task': 'quality', 'arguments': {'quality_weight': 0., 'future_weight': 0.,
                                                               'future_target': 'union'}},
                  'head_supervision': {'rank': True, 'quality_aux': False, 'future': False}}
    assert evaluation_options(checkpoint) == {'future_enabled': False, 'future_target': 'union'}
    checkpoint['metadata']['arguments']['future_weight'] = 1.
    checkpoint['head_supervision']['future'] = True
    assert evaluation_options(checkpoint)['future_enabled']
