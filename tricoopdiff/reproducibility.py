"""Explicit numerical policy for reproducible inference and diagnostic controls."""

import os
import random

import numpy as np
import torch


def capture_rng_state():
    """Serializable epoch-boundary state, compatible with weights-only loading."""
    numpy = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('resume CUDA device count differs from saved RNG state')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


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
