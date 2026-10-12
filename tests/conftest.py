"""Shared offline test bootstrap and bounded CPU parallelism."""

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Apply compatibility patches before any vendored DiffDock model import.
importlib.import_module("tricoopdiff.diffdock_compat")


def pytest_sessionstart(session):
    import torch

    torch.set_num_threads(2)
