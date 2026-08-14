"""Vendored DiffDock core (model + reverse sampling), self-contained.

This subpackage vendors the parts of DiffDock that TriCoopDiff needs to build
and run the frozen DiffDock-L expert, plus the reverse-sampling loop, so the
project no longer depends on the sibling ``../DiffDock`` checkout.

Upstream source: github.com/gcorso/DiffDock commit 47eb91a (+ uncommitted
``nan-stability-fixes`` working-tree changes). The only permitted divergences
from upstream are:
  1. relative imports (``.models`` / ``.utils``) + a lazy ``esm`` import +
     feature-dims moved to ``constants.py`` (drops RDKit / BioPython / ProDy /
     fair-esm from the model core);
  2. the NaN root-cause fix (``NAN_CLAMP`` clamp + ``NAN_TRACE`` diagnostics +
     ``tor_bonds[0]`` sigma indexing) in ``models/cg_model.py``, the
     ``NAN_TRACE`` tensor-product diagnostics in ``models/tensor_layers.py``,
     and the NaN fallback gate (Bug B) in ``sampling.py``.

Data featurization (``InferenceDataset`` / ``datasets/process_mols.py``) is NOT
vendored here and still lives in ``../DiffDock``.
"""
from . import constants  # noqa: F401
