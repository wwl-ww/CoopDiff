"""Compatibility command for the current production dual flow gates."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("test_dual_consensus.py")), run_name="__main__")
