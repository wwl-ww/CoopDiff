#!/usr/bin/env python
"""Quantify SDPI/native-DSM endpoint mismatch without models or GPU work."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validate_initialization import read_case
from tricoopdiff.legacy.diffusion import InterfaceDiffusion
from tricoopdiff.initialization import INITIALIZATION_VERSION, TernaryInitializer, receptor_prior
from tricoopdiff.initialization.audit import (
    conditional_translation_kl, invariant_atom_pairs, invariant_geometry_gap,
)


def audit_case(directory, seeds, diffusion):
    a, b, data = read_case(directory)
    mol = Chem.Mol(data['mol'])
    native = np.asarray(mol.GetConformer().GetPositions())
    bonds = data['edge_index'].T[data['edge_mask']].tolist()
    pairs = invariant_atom_pairs(mol, bonds)
    sides = {}
    for name, receptor in (('e3', a), ('poi', b)):
        center, radius, sigma = receptor_prior(receptor)
        sides[name] = {
            'train_terminal_sigma_angstrom': diffusion.tr_sigma_max,
            'inference_sigma_angstrom': sigma.item(),
            'receptor_rms_radius_angstrom': radius.item(),
            'native_ligand_to_receptor_center_angstrom': float(np.linalg.norm(native.mean(0) - center.numpy())),
            'conditional_translation_kl_nats': conditional_translation_kl(
                native.mean(0), center.numpy(), diffusion.tr_sigma_max, sigma.item()),
        }
    samples = []
    for seed in seeds:
        print(f'[audit] {directory.name}, seed={seed}', flush=True)
        ligand, _, _, metadata = TernaryInitializer().initialize(a, b, data, seed)
        samples.append({'seed': seed, 'initialization': metadata,
                        'geometry': invariant_geometry_gap(native, ligand[0].numpy(), pairs)})
    return {
        'case': directory.name, 'n_atoms': len(native), 'n_torsions': len(bonds),
        'source_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ('protein1.pdb', 'protein2.pdb', 'ligand.sdf')},
        'translation': sides, 'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 7, 42])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diffusion = InterfaceDiffusion()
    report = {
        'audit_version': 1, 'initialization_version': INITIALIZATION_VERSION,
        'strict_endpoint_consistency': False,
        'interpretation': {
            'translation': 'Conditional KL for each native example, NOT the KL of the data marginal.',
            'torsion': 'Magnitude of the conditional first circular moment; uniform prior has zero.',
            'geometry': 'Necessary indexed-coordinate invariants only; no claim about DockQ or complete reachability.',
            'rotation': 'Not numerically audited here; finite IGSO3 noise is not exact Haar.',
        },
        'torsion_terminal': {
            'train_sigma_radians': diffusion.tor_sigma_max,
            'train_conditional_first_moment_magnitude': math.exp(-diffusion.tor_sigma_max ** 2 / 2),
            'uniform_prior_first_moment_magnitude': 0.0,
        },
        'cases': [audit_case(path, args.seeds, diffusion) for path in args.case_dirs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Audit completed (NOT a consistency pass): {args.out}')


if __name__ == '__main__':
    main()
