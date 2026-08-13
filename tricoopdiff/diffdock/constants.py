"""Feature-dimension constants, extracted from DiffDock's ``datasets/process_mols.py``.

These three tuples size the atom/residue embedding layers in ``CGModel``. They are
baked here as literals so the vendored model core does not need to import
``datasets.process_mols`` (which transitively pulls RDKit / BioPython / ProDy).

Values verified against ``../DiffDock`` on 2026-08-13 (commit 47eb91a) via
``from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims``.
"""

lig_feature_dims = ([119, 4, 12, 12, 8, 10, 6, 6, 2, 8, 2, 2, 2, 2, 2, 2], 0)

rec_residue_feature_dims = ([38], 0)

rec_atom_feature_dims = ([38, 119, 23, 38], 0)
