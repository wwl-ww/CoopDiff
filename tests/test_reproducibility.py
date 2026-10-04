"""Numerical policy must be explicit and resume fingerprints must honor it."""

import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from tricoopdiff.runtime import configure_inference_reproducibility


def test_rng_restore_replays_python_numpy_and_torch():
    import random
    import numpy as np
    from tricoopdiff.runtime import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    first = (random.random(), np.random.rand(4), torch.rand(5))
    restore_rng_state(state)
    second = (random.random(), np.random.rand(4), torch.rand(5))
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1]) and torch.equal(first[2], second[2])


def test_expert_policies_are_instance_local_and_sigma_configs_are_checked():
    from types import SimpleNamespace
    from tricoopdiff.models.expert import configure_expert_policy, validate_expert_sigma_args

    a, b = SimpleNamespace(model=SimpleNamespace()), SimpleNamespace(model=SimpleNamespace())
    with patch.dict(os.environ, {'NAN_CLAMP': '777', 'NAN_CLAMP_MODE': 'componentwise'}):
        configure_expert_policy(a, 100., 'irrep_norm')
        configure_expert_policy(b, 1., 'componentwise')
        assert a.model.activation_clamp == 100. and a.model.activation_clamp_mode == 'irrep_norm'
        assert b.model.activation_clamp == 1. and os.environ['NAN_CLAMP'] == '777'
    args = SimpleNamespace(tr_sigma_min=.1, tr_sigma_max=19., rot_sigma_min=.03,
                           rot_sigma_max=1.55, tor_sigma_min=.0314, tor_sigma_max=3.14)
    validate_expert_sigma_args(args)
    args.tr_sigma_max = 99.
    try:
        validate_expert_sigma_args(args)
    except ValueError as error:
        assert 'tr_sigma_max' in str(error)
    else:
        raise AssertionError('mismatched expert sigma was accepted')


def test_atomic_checkpoint_failure_preserves_existing_artifact(tmp_path):
    from tricoopdiff.runtime import atomic_torch_save

    destination = tmp_path / 'checkpoint.pt'
    atomic_torch_save({'step': 1}, destination)

    def partial_write_then_fail(payload, path):
        Path(path).write_bytes(b'incomplete')
        raise RuntimeError('interrupted checkpoint writer')

    with patch('tricoopdiff.runtime.torch.save', side_effect=partial_write_then_fail):
        try:
            atomic_torch_save({'step': 2}, destination)
        except RuntimeError:
            pass
        else:
            raise AssertionError('failed checkpoint write was hidden')
    assert torch.load(destination, weights_only=True)['step'] == 1
    assert list(tmp_path.iterdir()) == [destination]


def test_shared_asset_hash_streams_identical_bytes(tmp_path):
    import hashlib
    from tricoopdiff.runtime import sha256_file

    destination = tmp_path / 'asset.bin'
    value = b'cached-expert-asset' * 20
    destination.write_bytes(value)
    assert sha256_file(destination, chunk_size=7) == hashlib.sha256(value).hexdigest()


def test_atomic_report_rejects_nonfinite_without_overwriting(tmp_path):
    import json
    from tricoopdiff.runtime import atomic_json

    destination = tmp_path / 'report.json'
    atomic_json({'dockq': .5}, destination)
    try:
        atomic_json({'dockq': float('nan')}, destination)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid JSON metric was published')
    assert json.loads(destination.read_text())['dockq'] == .5
    assert list(tmp_path.iterdir()) == [destination]


def test_deterministic_policy_and_explicit_legacy_control():
    before = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn
    cuda = torch.backends.cuda
    backend_state = (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32,
                     cuda.matmul.allow_tf32)
    attention_state = (cuda.flash_sdp_enabled(), cuda.mem_efficient_sdp_enabled(),
                       cuda.cudnn_sdp_enabled(), cuda.math_sdp_enabled())
    try:
        with patch.dict(os.environ, {}, clear=True):
            config = configure_inference_reproducibility()
            assert config['cublas_workspace_config'] == ':4096:8'
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert config['math_sdp'] and not config['flash_sdp']
            assert not config['memory_efficient_sdp'] and not config['cudnn_sdp']
            assert not configure_inference_reproducibility(False)['deterministic_algorithms']
            assert not torch.are_deterministic_algorithms_enabled()
        with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': 'invalid'}):
            try:
                configure_inference_reproducibility()
            except ValueError:
                pass
            else:
                raise AssertionError('invalid deterministic configuration must fail')
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn_only)
        (cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32,
         cuda.matmul.allow_tf32) = backend_state
        for setter, value in zip(
                (cuda.enable_flash_sdp, cuda.enable_mem_efficient_sdp,
                 cuda.enable_cudnn_sdp, cuda.enable_math_sdp), attention_state):
            setter(value)


if __name__ == '__main__':
    test_deterministic_policy_and_explicit_legacy_control()
    print('Inference reproducibility policy tests passed')
