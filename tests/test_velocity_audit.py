"""CPU checks for validation velocity-direction diagnostic summaries."""

import importlib.util
from pathlib import Path
import sys

import torch


_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_velocity_directions.py'
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location('audit_velocity_directions', _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_vector_metrics_preserve_sign_and_scale():
    target = torch.tensor([2., 0., 0.])
    aligned = audit.vector_metrics(torch.tensor([1., 0., 0.]), target)
    opposed = audit.vector_metrics(torch.tensor([-1., 0., 0.]), target)
    assert aligned['cosine'] == 1 and aligned['signed_projection_ratio'] == .5
    assert opposed['cosine'] == -1 and opposed['signed_projection_ratio'] == -.5


def test_vector_metrics_encode_undefined_direction_without_nan():
    result = audit.vector_metrics(torch.zeros(3), torch.zeros(3))
    assert result['cosine'] is None
    assert result['signed_projection_ratio'] is None
    assert result['error_norm'] == 0


if __name__ == '__main__':
    test_vector_metrics_preserve_sign_and_scale()
    test_vector_metrics_encode_undefined_direction_without_nan()
    print('Velocity-audit tests passed')
