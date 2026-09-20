"""Native-free selector features for official DeepTernary PDB candidates."""

from pathlib import Path

import numpy as np
import torch

from .ranking_archive import GEOMETRY_NAMES, geometry_features


FEATURE_VERSION = 'deepternary_candidate_geometry_v1'
FEATURE_NAMES = ('pred_p2_rmsd', 'clash_ratio') + GEOMETRY_NAMES


def _coordinates(lines, predicate):
    values = [
        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        for line in lines if predicate(line)
    ]
    if not values:
        raise ValueError('DeepTernary candidate is missing a required coordinate group')
    result = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError('DeepTernary candidate contains non-finite coordinates')
    return result


def deepternary_candidate_features(path, pred_p2_rmsd, clash_ratio):
    """Return official scores plus the repository's 14 geometry features."""
    official = np.asarray([pred_p2_rmsd, clash_ratio], dtype=np.float32)
    if official.shape != (2,) or not np.isfinite(official).all() or (official < 0).any():
        raise ValueError('invalid DeepTernary surrogate or clash score')
    lines = Path(path).read_text().splitlines()
    e3 = _coordinates(lines, lambda line: line.startswith('ATOM')
                      and line[21] == 'A' and line[12:16].strip() == 'CA')
    poi = _coordinates(lines, lambda line: line.startswith('ATOM')
                       and line[21] == 'B' and line[12:16].strip() == 'CA')
    ligand = _coordinates(lines, lambda line: line.startswith('HETATM'))
    result = np.concatenate((official, geometry_features(e3, poi, ligand)))
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('invalid DeepTernary candidate feature vector')
    return result
