"""Label-free state features and deterministic post-ranking of saved poses."""

import math

import torch

from tricoopdiff.pose_ranking import StateQualityRanker


GENERATOR_KEYS = ('protocol', 'checkpoint_sha256', 'probe', 'runtime_sha256',
                  'pool', 'steps', 'expert_assets')
INFERENCE_FIELDS = {'architecture', 'model_config', 'state_dict', 'generator',
                    'geometry_only', 'mean', 'scale'}


def export_quality_checkpoint(checkpoint):
    """Exclude training labels, validation metrics, case IDs and training paths."""
    result = {key: checkpoint[key] for key in INFERENCE_FIELDS}
    result['generator'] = {key: checkpoint['generator'][key] for key in GENERATOR_KEYS}
    return result


def feature_vector(features, frozen_dim, geometry_only=False):
    """Allowlist inputs; metadata, source, stage and truth are not model features."""
    geometry = torch.as_tensor(features['geometry']).float()
    if geometry.ndim != 1 or not torch.isfinite(geometry).all() or (geometry < 0).any():
        raise ValueError('geometry must be a finite nonnegative vector')
    geometry = torch.log1p(geometry)
    if geometry_only:
        return geometry
    frozen = (torch.as_tensor(features['frozen']).float() if features['probe_valid']
              else torch.zeros(frozen_dim))
    if frozen.shape != (frozen_dim,) or not torch.isfinite(frozen).all():
        raise ValueError('invalid frozen feature vector')
    return torch.cat((frozen, geometry, torch.tensor([float(features['probe_valid'])])))


def archive_rows(archive):
    """Only sampled initial/final pairs, never unsampled full-pool initial states."""
    rows, seen = [], set()
    for pair in archive['pairs']:
        index, source = pair['pool_index'], pair['source']
        if index in seen or not isinstance(index, int) or not 0 <= index < len(archive['pool']):
            raise ValueError('invalid or duplicate pool index')
        seen.add(index)
        if source not in ('bitseed', 'sdpi') or archive['pool'][index]['source'] != source:
            raise ValueError('actual source mismatch')
        for stage in ('initial', 'final'):
            if stage == 'final' and not pair['valid']:
                continue
            features = pair[f'{stage}_features']
            if features is None:
                raise ValueError('missing features for an available state')
            rows.append({'pool_index': index, 'stage': stage, 'source': source, 'features': features})
    if not rows:
        raise ValueError('no archived states')
    return sorted(rows, key=lambda r: (r['pool_index'], r['stage'] != 'initial'))


def ordered_rows(rows):
    """Score descending; exact ties use pool ID then initial before final."""
    if any(not math.isfinite(r['score']) for r in rows):
        raise ValueError('non-finite ranking score')
    return sorted(rows, key=lambda r: (-r['score'], r['pool_index'], r['stage'] != 'initial'))


def load_quality_checkpoint(checkpoint, manifest):
    if checkpoint['architecture'] != 'state_quality_ranker_v1':
        raise ValueError('unsupported quality checkpoint')
    for key in GENERATOR_KEYS:
        if checkpoint['generator'][key] != manifest[key]:
            raise ValueError(f'ranker/generator mismatch: {key}')
    model = StateQualityRanker(**checkpoint['model_config']).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    mean, scale = checkpoint['mean'], checkpoint['scale']
    if (mean.shape != (model.config['input_dim'],) or scale.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('invalid training normalization')
    return model


@torch.no_grad()
def predict_archive(archive, model=None, checkpoint=None):
    rows = archive_rows(archive)
    if model is None:
        # Predeclared R0 diagnostic, NOT the historical Geometry-64 selector.
        # Balanced C-alpha contact fraction minus summed C-alpha clash fractions.
        scores = []
        for row in rows:
            geometry = torch.as_tensor(row['features']['geometry']).float()
            if geometry.shape != (14,) or not torch.isfinite(geometry).all():
                raise ValueError('R0 requires 14 finite C-alpha geometry features')
            scores.append(float(torch.minimum(geometry[4], geometry[10]) - geometry[5] - geometry[11]))
        predictions = [{'score': score} for score in scores]
    else:
        matrix = torch.stack([feature_vector(r['features'], archive['probe_feature_dim'],
                                            checkpoint['geometry_only']) for r in rows])
        out = model((matrix - checkpoint['mean']) / checkpoint['scale'])
        predictions = [{key: float(out[key][i]) for key in ('score', 'q_hat', 'p23', 'p49')}
                       for i in range(len(rows))]
    result = [{**{k: row[k] for k in ('pool_index', 'stage', 'source')}, **prediction}
              for row, prediction in zip(rows, predictions)]
    selected = {stage: [{k: r[k] for k in ('pool_index', 'stage', 'source')}
                        for r in ordered_rows([r for r in result if stage == 'union' or r['stage'] == stage])[:5]]
                for stage in ('initial', 'final', 'union')}
    return {'rows': result, 'selected_top5': selected}
