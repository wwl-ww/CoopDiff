"""Native-free, fixed-clock probes and initial/final pose archive helpers."""

import hashlib
import json
import random

import numpy as np
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
