"""Keep audit-only correspondence hypotheses out of production sampling."""

import json
from pathlib import Path


def require_deployable_condition_manifest(path):
    """The regular loader validates chemistry; this gate validates usage scope."""
    payload = json.loads(Path(path).read_text())
    if payload.get('audit_only') or payload.get('deployment_eligible') is False:
        raise ValueError('audit-only condition asset is not a deployable manifest')
    sides = payload.get('sides')
    if not isinstance(sides, list) or len(sides) != 2:
        raise ValueError('deployment manifest requires exactly two side entries')
    for side in sides:
        if side is None:
            continue
        if not isinstance(side, dict):
            raise ValueError('invalid deployment condition side')
        if side.get('audit_only') or side.get('mapping_method') == 'conservative_charge_normalized_mcs':
            raise ValueError('audit-only chemical mapping cannot drive constrained sampling')
    return payload
