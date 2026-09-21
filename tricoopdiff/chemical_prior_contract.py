"""Checkpointed chemical prior configuration and immutable graph-only libraries."""

import hashlib
import json
from pathlib import Path

import torch
from rdkit import Chem

from .chemical_torsion_prior import (CHEMICAL_PRIOR_POLICY, independent_conformers,
                                     ordered_graph_sha256, sample_transferred_prior)


MIXTURE_POLICY = 'binary_anchored_chemical_uniform_torsion_mixture_v1'


def chemical_prior_config(*, seed=7315, count=16, uniform_probability=0., assembly=False):
    config = {'policy': CHEMICAL_PRIOR_POLICY, 'seed': seed, 'count': count,
              'binary_offset_prior': 'template_origin', 'rdkit_version': Chem.rdBase.rdkitVersion}
    if uniform_probability:
        config.update(policy=MIXTURE_POLICY, uniform_torsion_probability=uniform_probability)
    if assembly:
        config['binary_offset_prior'] = 'assembly_pose_mixture_v1'
    validate_prior_config(config)
    return config


def validate_prior_config(config):
    if config is None:  # Legacy checkpoint: independent uniform torsions + old offset prior.
        return
    if not isinstance(config, dict):
        raise ValueError('chemical prior contract must be a dictionary')
    keys = {'policy', 'seed', 'count', 'binary_offset_prior', 'rdkit_version'}
    if config.get('policy') == MIXTURE_POLICY:
        keys.add('uniform_torsion_probability')
        probability = config.get('uniform_torsion_probability')
        if not isinstance(probability, (float, int)) or not 0 < probability < 1:
            raise ValueError('mixed prior needs a probability strictly between zero and one')
    if (set(config) != keys
            or config['policy'] not in (CHEMICAL_PRIOR_POLICY, MIXTURE_POLICY)
            or type(config['seed']) is not int or config['seed'] < 0
            or type(config['count']) is not int or config['count'] < 1
            or config['binary_offset_prior'] not in ('template_origin', 'assembly_pose_mixture_v1')
            or config['rdkit_version'] != Chem.rdBase.rdkitVersion):
        raise ValueError('unsupported chemical prior contract or RDKit version')


def graph_seed(molecule, config):
    validate_prior_config(config)
    if config is None:
        raise ValueError('chemical conformers require a chemical prior contract')
    identity = ordered_graph_sha256(molecule)
    return int.from_bytes(hashlib.sha256(f'{config["seed"]}:{identity}'.encode()).digest()[:4], 'big') % (2**31)


class ChemicalPriorLibrary:
    """Fail explicitly on absent, rejected, changed, or mismatched libraries."""

    def __init__(self, manifest, config, *, index_sha256):
        validate_prior_config(config)
        if config is None:
            raise ValueError('conformer library requires a chemical prior contract')
        self.config = config
        manifest = Path(manifest)
        self.sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        payload = json.loads(manifest.read_text())
        if (payload['root_seed'] != config['seed'] or payload['count'] != config['count']
                or payload['index_sha256'] != index_sha256):
            raise ValueError('conformer library seed, size, or dataset mismatch')
        self.rows = {row['case']: row for row in payload['cases']}
        if len(self.rows) != len(payload['cases']):
            raise ValueError('duplicate conformer library case')
        self.cache = {}

    def get(self, case, molecule):
        molecule = Chem.Mol(molecule)  # Featurization caches may store RDKit binary serialization.
        row = self.rows.get(case)
        if row is None:
            raise RuntimeError(f'{case}: missing chemical prior record')
        if row['status'] != 'prepared':
            raise ValueError(f'chemical prior unavailable: {row.get("reason", "rejected")}')
        identity = ordered_graph_sha256(molecule)
        if case not in self.cache:
            path = Path(row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError(f'{case}: chemical conformer library hash mismatch')
            self.cache[case] = torch.load(path, weights_only=False, map_location='cpu')
        saved = self.cache[case]
        audit, coords = saved['audit'], saved['coordinates']
        if (saved['ordered_graph_sha256'] != identity
                or audit['policy'] != CHEMICAL_PRIOR_POLICY
                or audit['seed'] != graph_seed(molecule, self.config)
                or audit['requested'] != self.config['count']
                or audit['rdkit_version'] != self.config['rdkit_version']
                or audit['input_coordinates_used'] is not False
                or coords.shape != (audit['accepted'], molecule.GetNumAtoms(), 3)
                or not coords.shape[0] or not torch.isfinite(coords).all()):
            raise RuntimeError(f'{case}: chemical prior graph/provenance mismatch')
        return coords


def deployment_conformers(molecule, config):
    """Same graph-seeded algorithm as the immutable training libraries."""
    return independent_conformers(molecule, seed=graph_seed(molecule, config), count=config['count'])


def conditioned_chemical_prior(flow, edges, conformers, *, generator, config):
    validate_prior_config(config)
    if config is None:
        raise ValueError('chemical prior requires explicit checkpoint configuration')
    uniform = (config.get('uniform_torsion_probability', 0.) > 0
               and float(torch.rand((), device=generator.device, generator=generator))
               < config['uniform_torsion_probability'])
    if uniform:
        state = flow.sample_prior(generator=generator)
        audit = {'policy': MIXTURE_POLICY, 'component': 'uniform', 'conformer_index': None,
                 'conformer_count': len(conformers), 'transferred_torsions': 0}
    else:
        state, audit = sample_transferred_prior(flow, edges, conformers, generator=generator)
        audit = {**audit, 'component': 'chemical'}
    count = len(flow.plan.bonds)
    # Start exactly at each supplied binary pose. Bounds allow learned changes,
    # but noise must not deliberately throw away this known part of the input.
    from .single_anchor_flow import SingleAnchorTorsionFlow
    from .bounded_binary_flow import BoundedBinaryTorsionFlow
    offset_count = 6 if isinstance(flow, SingleAnchorTorsionFlow) else (
        12 if isinstance(flow, BoundedBinaryTorsionFlow) else 0)
    state[count:count + offset_count] = 0
    apply_pose_prior(flow, state, generator=generator, config=config)
    return state, {**audit, 'config': dict(config), 'binary_offset_prior': config['binary_offset_prior']}


def apply_pose_prior(flow, state, *, generator, config):
    if config['binary_offset_prior'] != 'assembly_pose_mixture_v1':
        return
    from .assembly_flow import AssemblyFlow
    if not isinstance(flow, AssemblyFlow):
        raise ValueError('assembly prior requires the unbounded assembly decoder')
    count = len(flow.plan.bonds)
    # Half the starts retain both binary poses; half teach recovery of a
    # misassembled pair. Exactly the same distribution is used at deployment.
    state[count:] = 0
    if float(torch.rand((), generator=generator, device=state.device)) < .5:
        noise = torch.randn((2, 2, 3), generator=generator, device=state.device, dtype=state.dtype)
        noise[:, 0] *= .5  # 5 Angstrom per translation component; no truncation.
        noise[:, 1] *= .5  # rotation-vector radians; no hard displacement limit.
        state[count:] = noise.flatten()


def apply_training_prior(sample, ligand, conformers, *, generator, config):
    if sample is None:
        return None
    state, audit = conditioned_chemical_prior(sample.flow, ligand['edge_index'], conformers,
                                              generator=generator, config=config)
    sample.prior_angles = state
    sample.audit['prior'] = audit
    return sample
