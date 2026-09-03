"""Vendored DiffDock data-featurization modules (``datasets/*``).

Ported verbatim from the upstream DiffDock checkout (``datasets/``) with only
the internal imports rewritten to ``tricoopdiff.diffdock.datasets.*`` /
``tricoopdiff.diffdock.utils.*``. These modules pull RDKit / BioPython / ProDy /
fair-esm, and are only imported by the featurization path — the model core
(``tricoopdiff.diffdock.models`` / ``.utils``) does not depend on them.
"""
