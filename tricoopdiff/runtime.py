"""Shared numerical policy, artifact identity and atomic checkpoint publishing."""

import hashlib
import json
import os
import random
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import tempfile

import torch
import numpy as np


def capture_rng_state():
    """Capture every active RNG without initializing CUDA for CPU callers."""
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': ([state.clone() for state in torch.cuda.get_rng_state_all()]
                           if torch.cuda.is_initialized() else [])}


def restore_rng_state(state):
    """Restore checkpoint streams; reject a different visible CUDA topology."""
    if state['torch_cuda'] and len(state['torch_cuda']) != torch.cuda.device_count():
        raise ValueError('checkpoint CUDA RNG topology differs from this runtime')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def preserve_rng(function):
    """Keep diagnostic sampling from changing the caller's random streams."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = capture_rng_state()
        try:
            return function(*args, **kwargs)
        finally:
            restore_rng_state(state)
    return wrapped


@contextmanager
def evaluation_context(model):
    """Read-only inference restores every module mode and active RNG stream."""
    state = capture_rng_state()
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        restore_rng_state(state)


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash cached assets without loading a complete checkpoint into RAM."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_publish(destination, write):
    """Publish a complete artifact; failures preserve the previous destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + '.tmp.', dir=destination.parent)
    os.close(descriptor)
    try:
        write(temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_torch_save(payload, destination):
    """Publish training state without shared temporary names or partial files."""
    _atomic_publish(destination, lambda path: torch.save(payload, path))


def atomic_json(payload, destination):
    """Publish strict UTF-8 JSON; non-finite numbers cannot become valid reports."""
    def write(path):
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                              encoding='utf-8')
    _atomic_publish(destination, write)


def configure_inference_reproducibility(deterministic=True):
    """Enable deterministic kernels; fail on unsupported operations, never warn-only.

    Call before constructing CUDA models. RNG seeds alone do not make CUDA
    reductions deterministic, and flow integration can amplify small differences.
    """
    if deterministic:
        workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if workspace not in (':4096:8', ':16:8'):
            raise ValueError('deterministic inference requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        # Fix the attention implementation rather than allowing backend/shape
        # heuristics to choose different fused reduction paths.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return {
        'deterministic_algorithms': bool(deterministic),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
        'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
        'math_sdp': torch.backends.cuda.math_sdp_enabled(),
    }
