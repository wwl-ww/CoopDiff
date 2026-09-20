"""Native-free top-1 must follow selector order, never DockQ order."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'evaluate_selected_refinement', ROOT / 'scripts/evaluate_selected_refinement.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_native_free_top1_follows_selector_order():
    selected = [{'candidate_id': ['sdpi', 3]}, {'candidate_id': ['bitseed', 8]}]
    rows = {
        repr(['sdpi', 3]): {'candidate_id': ['sdpi', 3], 'pool_index': 3,
                            'source': 'sdpi', 'initial_dockq': .1,
                            'refined_dockq': .2, 'safe_dockq': .2,
                            'safe_stage': 'refined'},
        repr(['bitseed', 8]): {'candidate_id': ['bitseed', 8], 'pool_index': 8,
                              'source': 'bitseed', 'initial_dockq': .8,
                              'refined_dockq': .9, 'safe_dockq': .9,
                              'safe_stage': 'refined'},
    }
    result = module.native_free_top1(selected, rows)
    assert result['candidate_id'] == ['sdpi', 3]
    assert result['safe_dockq'] == .2
    assert result['selection_used_native'] is False


def test_native_free_top1_handles_missing_or_empty_selection():
    assert module.native_free_top1([], {}) is None
    assert module.native_free_top1([{'candidate_id': ['sdpi', 0]}], {}) is None
