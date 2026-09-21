"""Audit hypotheses must not silently become hard deployment constraints."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tricoopdiff.deployment_condition_guard import require_deployable_condition_manifest


def test_guard_rejects_audit_side_even_without_top_level_flag(tmp_path):
    path = tmp_path / 'manifest.json'
    for side in ({'audit_only': True}, {'mapping_method': 'conservative_charge_normalized_mcs'}):
        path.write_text(json.dumps({'sides': [side, None]}))
        try:
            require_deployable_condition_manifest(path)
        except ValueError as error:
            assert 'audit-only' in str(error)
        else:
            raise AssertionError('audit mapping reached constrained deployment')


def test_guard_preserves_trusted_single_and_dual_manifests(tmp_path):
    path = tmp_path / 'manifest.json'
    for sides in ([{'mapping_method': 'exact_stereo'}, None],
                  [{'mapping_method': 'exact_stereo'}, {'mapping_method': 'charge_normalized_exact_stereo'}]):
        path.write_text(json.dumps({'sides': sides}))
        assert require_deployable_condition_manifest(path)['sides'] == sides
