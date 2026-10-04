"""Native-free, fixed-clock probes and initial/final pose archive helpers."""

import hashlib
import json
import random

import numpy as np
from scipy.spatial import cKDTree
import torch


ARCHIVE_VERSION = 'initial_final_pose_archive_v1'
PROBE_TIME = 0.05
GEOMETRY_NAMES = tuple(
    f'{side}_{stat}' for side in ('e3_ligand', 'poi_ligand')
    for stat in ('min', 'q25', 'median', 'q75', 'contact6_fraction', 'clash2_fraction')
) + ('protein_ca_min', 'protein_com_distance')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(item) for item in value]
    return value


def state_hash(state):
    digest = hashlib.sha256()
    for key in ('ligand', 'poi', 'body_frame'):
        array = np.ascontiguousarray(torch.as_tensor(state[key]).cpu().numpy())
        digest.update(json.dumps([key, str(array.dtype), array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def geometry_features(e3, poi, ligand):
    """C-alpha distance proxies only; not full-atom physical-validity scores."""
    values = []
    for receptor in (e3, poi):
        nearest = torch.cdist(ligand, receptor).min(1).values
        values.extend(float(torch.quantile(nearest, q)) for q in (0., .25, .5, .75))
        values.extend(float((nearest < radius).float().mean()) for radius in (6., 2.))
    values.extend((float(torch.cdist(e3, poi).min()),
                   float(torch.linalg.vector_norm(e3.mean(0) - poi.mean(0)))))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('non-finite geometry features')
    return result


def select_archive_candidates(candidates, count, seed):
    """Uniform within actual source, approximately balanced; record inclusion odds."""
    if not 1 <= count <= len(candidates):
        raise ValueError('archive count must fit the candidate pool')
    groups = {source: [i for i, c in enumerate(candidates) if c['candidate_id'][0] == source]
              for source in ('bitseed', 'sdpi')}
    if sum(map(len, groups.values())) != len(candidates):
        raise ValueError('unknown candidate source')
    bit = min(count // 2, len(groups['bitseed']))
    sdpi = min(count - bit, len(groups['sdpi']))
    bit = count - sdpi
    rng = np.random.RandomState(seed)
    selected, probability = [], {}
    for source, quota in (('bitseed', bit), ('sdpi', sdpi)):
        ids = groups[source]
        if quota:
            selected.extend(int(i) for i in rng.choice(ids, quota, replace=False))
        probability[source] = quota / len(ids) if ids else 0.
    return sorted(selected), probability


@torch.no_grad()
def probe_state(model, diffusion, e3_data, poi_data, ligand_data, state, source):
    """Probe exact stored geometry at t=.05, dt=0; never advance the flow."""
    if model.training:
        raise ValueError('archive probe requires a frozen eval-mode generator')
    if source not in ('sdpi', 'bitseed'):
        raise ValueError('invalid actual source')
    device = e3_data['coords'].device
    ligand = torch.as_tensor(state['ligand'], device=device).unsqueeze(0)
    poi = torch.as_tensor(state['poi'], device=device).unsqueeze(0)
    frame = torch.as_tensor(state['body_frame'], device=device).unsqueeze(0)
    t = torch.tensor([PROBE_TIME], device=device)
    dt = torch.zeros_like(t)
    features = {'geometry': geometry_features(e3_data['coords'][0], poi[0], ligand[0]),
                'probe_valid': False, 'frozen': None}
    # Diagnostic forwards must not affect later candidate generation or sampling RNG.
    devices = [device.index or 0] if device.type == 'cuda' else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            sigma, _, _ = diffusion.t_to_sigma(t, t, t)
            cutoff = sigma * 3 + 20 if model.expert.model.dynamic_max_cross else None
            out = model(e3_data, {**poi_data, 'coords': poi},
                        {**ligand_data, 'coords': ligand}, t_e3l=t, t_pol=t,
                        r_e3_to_poi=frame, receptor_crop_cutoff=cutoff,
                        diffusion_state=diffusion.conditioning(t, dt, e3_data['coords'], poi_data['coords']),
                        source=int(source == 'bitseed'))
            vector = out['rank_features']
            if vector.ndim != 2 or vector.shape[0] != 1 or not torch.isfinite(vector).all():
                raise FloatingPointError('invalid frozen probe features')
            features.update(frozen=vector[0].detach().cpu().numpy().copy(), probe_valid=True)
    except FloatingPointError as error:
        features['probe_error'] = str(error)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return features


_LIGAND_STATS = ('min', 'q10', 'q25', 'median', 'q75',
                 'frac_lt2', 'frac_lt4', 'frac_lt6')
_PROTEIN_COVERAGE = ('coverage_lt2', 'coverage_lt4', 'coverage_lt6', 'coverage_lt8')
_INTERFACE_STATS = ('min', 'q01', 'q05', 'frac_lt2', 'frac_lt4', 'frac_lt6')

EXPANDED_GEOMETRY_NAMES = tuple(
    f'{side}_{name}'
    for side in ('e3_ligand', 'poi_ligand')
    for name in (*_LIGAND_STATS, *_PROTEIN_COVERAGE)
) + tuple(
    f'{direction}_{name}'
    for direction in ('e3_to_poi', 'poi_to_e3')
    for name in _INTERFACE_STATS
) + ('protein_heavy_centroid_distance',)


def apply_fitted_rigid(source_ca, target_ca, points):
    """Fit source C-alpha coordinates to a candidate and move matching heavy atoms."""
    source = _points(source_ca, 'source_ca')
    target = _points(target_ca, 'target_ca')
    moving = _points(points, 'points')
    if source.shape != target.shape or len(source) < 3:
        raise ValueError('source and target C-alpha arrays must match with N >= 3')
    source_center, target_center = source.mean(0), target.mean(0)
    x, y = source - source_center, target - target_center
    u, singular, vh = np.linalg.svd(x.T @ y)
    if singular[1] <= np.finfo(np.float64).eps * singular[0]:
        raise ValueError('C-alpha rigid fit is underdetermined')
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vh)
    rotation = u @ np.diag(sign) @ vh
    translation = target_center - source_center @ rotation
    fitted = source @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return moving @ rotation + translation, rmsd


def _points(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value):
        raise ValueError(f'{name} must have shape [N,3] with N > 0')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite coordinates')
    return value


def _fractions(distances, radii):
    return [float(np.mean(distances < radius)) for radius in radii]


def expanded_geometry_features(e3_heavy, poi_heavy, ligand_heavy):
    """Return invariant distance summaries without native or quality information."""
    e3 = _points(e3_heavy, 'e3_heavy')
    poi = _points(poi_heavy, 'poi_heavy')
    ligand = _points(ligand_heavy, 'ligand_heavy')
    e3_tree, poi_tree, ligand_tree = cKDTree(e3), cKDTree(poi), cKDTree(ligand)
    values = []
    for protein, tree in ((e3, e3_tree), (poi, poi_tree)):
        ligand_nearest = tree.query(ligand, k=1, workers=1)[0]
        values.extend(float(np.quantile(ligand_nearest, q))
                      for q in (0., .1, .25, .5, .75))
        values.extend(_fractions(ligand_nearest, (2., 4., 6.)))
        protein_nearest = ligand_tree.query(protein, k=1, workers=1)[0]
        values.extend(_fractions(protein_nearest, (2., 4., 6., 8.)))
    for source, target_tree in ((e3, poi_tree), (poi, e3_tree)):
        nearest = target_tree.query(source, k=1, workers=1)[0]
        values.extend(float(np.quantile(nearest, q)) for q in (0., .01, .05))
        values.extend(_fractions(nearest, (2., 4., 6.)))
    values.append(float(np.linalg.norm(e3.mean(0) - poi.mean(0))))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(EXPANDED_GEOMETRY_NAMES),) or not np.isfinite(result).all():
        raise ValueError('invalid expanded geometry feature vector')
    if (result < 0).any():
        raise ValueError('distance features must be nonnegative')
    return result
