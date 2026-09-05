"""Vendored DiffDock model core and data featurization.

This subpackage vendors everything TriCoopDiff needs from DiffDock: the model
core to build/run the frozen DiffDock-L expert and the data-featurization
modules — so the project no longer depends on the
sibling ``../DiffDock`` checkout at all.

Upstream source: github.com/gcorso/DiffDock commit 47eb91a (+ uncommitted
``nan-stability-fixes`` working-tree changes). The only permitted divergences
from upstream are:
  1. imports rewritten to ``tricoopdiff.diffdock.*`` + a lazy ``esm`` import +
     feature-dims moved to ``constants.py`` (keeps RDKit / BioPython / ProDy /
     fair-esm out of the *model core*; the vendored ``.datasets`` +
     ``.utils.inference_utils`` featurization modules do pull them, but only
     when featurization is imported);
  2. the NaN root-cause fix (``NAN_CLAMP`` clamp + ``NAN_TRACE`` diagnostics +
     ``tor_bonds[0]`` sigma indexing) in ``models/cg_model.py``, the
     ``NAN_TRACE`` tensor-product diagnostics in ``models/tensor_layers.py``.

Data featurization (``InferenceDataset`` / ``get_sequences`` /
``datasets/process_mols.py`` / ``utils/inference_utils.py``) is vendored under
``.datasets`` and ``.utils.inference_utils``; scripts import it via
``tricoopdiff.diffdock.datasets.*`` / ``tricoopdiff.diffdock.utils.inference_utils``
instead of the old ``from utils.inference_utils`` / ``from datasets.*`` imports
that required the ``../DiffDock`` checkout on ``sys.path``.
"""
from . import constants  # noqa: F401
