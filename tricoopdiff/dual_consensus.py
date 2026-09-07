"""Dual-hypothesis flow: E3 fixed, two ligand geometries, one POI body.

All vectors use the common E3 frame. B's ligand velocity is relative to POI;
POI motion transports B's hypothesis but never transports A's hypothesis.
Time runs from reachable clean (0) to independent local priors (1).
"""

from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .initialization import receptor_prior, validate_ligand_graph
from .sdpi_flow import (
    SDPIFlow as BinaryFlowPreconditioning, _align, _raw_torsions,
    _shape_and_gauge, _rotation_exp, _check_coords,
)


FLOW_VERSION = 'dual_consensus_flow_v1'
CHECKPOINT_VERSION = 18
ARCHITECTURE = 'dual_consensus_v1'
CONTRACT = {
    'conformer': 'ETKDGv3_seed0_random_coords',
    'target': 'torsion_kabsch_least_squares_v1',
    'poi_prior': 'haar_gaussian_sum_receptor_scales',
    'consensus': 'internal_then_endpoint_then_pose_v1',
    'collapse': 'a_manifold_no_coordinate_average',
}


def chemistry(data):
    from rdkit import Chem
    mol = data['mol']
    mol = Chem.Mol(mol) if isinstance(mol, bytes) else mol
    bonds, masks = validate_ligand_graph(
        mol, data['edge_index'], data['edge_mask'], data['mask_rotate'])
    return mol, bonds.numpy(), masks


@lru_cache(maxsize=256)
def _embed(mol_bytes):
    """One reproducible reachable manifold per indexed chemical graph."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.Mol(mol_bytes)
    mol.RemoveAllConformers()
    hydrogenated = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(hydrogenated, params) != 0:
        raise ValueError('dual consensus ETKDG failed; no native fallback')
    return hydrogenated.GetConformer().GetPositions()[:mol.GetNumAtoms()].copy()


def base_conformer(data):
    from rdkit import Chem
    mol, _, _ = chemistry(data)
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    return _embed(mol.ToBinary()).copy()


@lru_cache(maxsize=128)
def _project(base_bytes, target_bytes, n, bonds_tuple, masks_bytes, no_torsion):
    base = np.frombuffer(base_bytes, dtype=np.float64).reshape(n, 3)
    target = np.frombuffer(target_bytes, dtype=np.float64).reshape(n, 3)
    bonds = np.array(bonds_tuple, dtype=np.int64).reshape(-1, 2)
    masks = np.frombuffer(masks_bytes, dtype=bool).reshape(len(bonds), n)
    def residual(angles):
        return (_align(_raw_torsions(base, bonds, masks, angles), target) - target).ravel()
    angles = np.zeros(len(bonds))
    if len(bonds) and not no_torsion:
        fitted = least_squares(residual, angles, max_nfev=100, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        angles = fitted.x
    projected = _align(_raw_torsions(base, bonds, masks, angles), target)
    if not np.isfinite(projected).all():
        raise ValueError('nonfinite reachable projection')
    return projected, float(np.sqrt(np.mean(np.sum((projected - target)**2, axis=-1))))


def reachable_target(data, no_torsion=False):
    """Local least-squares projection, not a claim of global optimality.

    Native is used ONLY here for supervision. Inference calls base_conformer.
    Bond lengths, ring geometry and stereochemistry remain those of ETKDG.
    """
    _, bonds, masks = chemistry(data)
    base = np.ascontiguousarray(base_conformer(data), dtype=np.float64)
    target = np.ascontiguousarray(data['coords'][0].detach().cpu(), dtype=np.float64)
    projected, rmsd = _project(base.tobytes(), target.tobytes(), len(base),
                              tuple(map(tuple, bonds)), masks.tobytes(), no_torsion)
    return torch.as_tensor(projected.copy(), device=data['coords'].device,
                           dtype=data['coords'].dtype)[None], rmsd


def consensus_schedule(t):
    """Zero pose pressure at high noise; all schedules increase toward clean."""
    progress = (1 - t).clamp(0, 1)
    return {'internal': 0.1 * progress, 'torsion': 0.1 * progress,
            'endpoint': ((progress - 0.3) / 0.7).clamp(0, 1).square(),
            'pose': ((progress - 0.65) / 0.35).clamp(0, 1).square()}


def _rigid(coords, tr, rot, dt):
    center = coords.mean(1, keepdim=True)
    matrix = _rotation_exp(-dt[:, None] * rot)
    return (coords - center) @ matrix.transpose(-1, -2) + center - dt[:, None, None] * tr[:, None], matrix


def torsion_step(coords, angles, data):
    """Differentiable bond rotations with first-order horizontal gauge removal.

    Only rigid rotations and bond torsions are applied: no Cartesian averaging,
    no detached numpy reconstruction in the training loss.
    """
    if not angles.numel():
        return coords
    _, bonds, masks = chemistry(data)
    if angles.shape != (1, len(bonds)):
        raise ValueError('torsions must be [1, n_rot]')
    center = coords.mean(1, keepdim=True)
    x = coords[0] - center[0]
    tangent = torch.zeros_like(x)
    for (u, v), mask, angle in zip(bonds, masks, angles[0]):
        axis = torch.nn.functional.normalize(x[u] - x[v], dim=-1, eps=1e-12)
        tangent = tangent + torch.as_tensor(mask, device=x.device)[:, None] * angle * torch.cross(axis.expand_as(x), x - x[v], dim=-1)
    inertia = torch.eye(3, device=x.device, dtype=x.dtype) * x.square().sum() - x.T @ x
    spin = torch.linalg.pinv(inertia, hermitian=True) @ torch.cross(x, tangent, dim=-1).sum(0)
    moved = coords[0]
    for (u, v), mask, angle in zip(bonds, masks, angles[0]):
        axis = torch.nn.functional.normalize(moved[u] - moved[v], dim=-1, eps=1e-12)
        rotation = _rotation_exp(axis * angle)
        rotated = (moved - moved[v]) @ rotation.T + moved[v]
        moved = torch.where(torch.as_tensor(mask, device=x.device)[:, None], rotated, moved)
    moved = (moved - moved.mean(0)) @ _rotation_exp(-spin).T + center[0]
    return moved[None]


class DualConsensusFlow(BinaryFlowPreconditioning):
    version = FLOW_VERSION

    def sample_coupling(self, e3_coords, poi_coords, ligand_data, no_torsion=False):
        _, bonds, masks = chemistry(ligand_data)
        clean, error = reachable_target(ligand_data, no_torsion)
        scales = self.scales(e3_coords, poi_coords).detach().cpu().numpy()
        centers = np.stack([x[0].mean(0).detach().cpu().numpy() for x in (e3_coords, poi_coords)])
        return {'clean': clean, 'projection_rmsd': error,
                'bonds': bonds, 'masks': masks,
                'angles': np.zeros((2, len(bonds))) if no_torsion else np.random.uniform(-np.pi, np.pi, (2, len(bonds))),
                'omegas': Rotation.random(3).as_rotvec(),
                'centers': centers + np.random.randn(2, 3) * scales[:, None],
                'poi_center': centers[0] + np.random.randn(3) * scales.sum()}

    def path(self, ligand_data, poi_coords, coupling, t):
        if t.shape != (1,) or not torch.isfinite(t).all() or not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError('dual flow time must be one finite value in [0,1]')
        clean_tensor = coupling['clean'][0]
        clean = clean_tensor.detach().cpu().double().numpy()
        convert = lambda x: torch.as_tensor(x, dtype=clean_tensor.dtype, device=clean_tensor.device)
        value = float(t.item())
        rotations = convert(Rotation.from_rotvec(value * coupling['omegas']).as_matrix())
        c0 = convert(clean.mean(0))
        centers = (1 - value) * c0 + value * convert(coupling['centers'])
        p0 = poi_coords[0].mean(0)
        pc = (1 - value) * p0 + value * convert(coupling['poi_center'])
        rb = rotations[2]
        poi = (poi_coords[0] - p0) @ rb.T + pc
        hypotheses, targets = [], {}
        for i, name in enumerate(('e3_ligand', 'poi_ligand')):
            shape, spin = _shape_and_gauge(clean, coupling['bonds'], coupling['masks'], coupling['angles'][i], value)
            ligand = (convert(shape) - c0) @ rotations[i].T + centers[i]
            tr = convert(coupling['centers'][i]) - c0
            rot = convert(coupling['omegas'][i]) + rotations[i] @ convert(spin)
            if i == 1:
                ligand = (ligand - p0) @ rb.T + pc
                tr, rot = tr @ rb.T, rot @ rb.T
            hypotheses.append(ligand[None])
            targets[name] = {'tr': tr[None], 'rot': rot[None], 'tor': convert(coupling['angles'][i])[None]}
        targets['poi'] = {'tr': (convert(coupling['poi_center']) - p0)[None],
                          'rot': convert(coupling['omegas'][2])[None]}
        return {'a': hypotheses[0], 'b': hypotheses[1], 'poi': poi[None], 'body': rb[None]}, targets

    def initialize(self, e3_coords, poi_coords, ligand_data, seed=0, no_torsion=False):
        """Independent priors; no training/native ligand coordinates are read."""
        rng = np.random.RandomState(seed)
        base = base_conformer(ligand_data)
        _, bonds, masks = chemistry(ligand_data)
        scales = self.scales(e3_coords, poi_coords).detach().cpu().numpy()
        ca, cb = [x[0].mean(0).detach().cpu().numpy() for x in (e3_coords, poi_coords)]
        rotations = Rotation.random(3, random_state=rng).as_matrix()
        centers = np.stack((ca, cb)) + rng.randn(2, 3) * scales[:, None]
        pc = ca + rng.randn(3) * scales.sum()
        angles = np.zeros((2, len(bonds))) if no_torsion else rng.uniform(-np.pi, np.pi, (2, len(bonds)))
        ligand = []
        for i in range(2):
            shape = _raw_torsions(base, bonds, masks, angles[i])
            x = (shape - shape.mean(0)) @ rotations[i].T + centers[i]
            if i == 1:
                x = (x - cb) @ rotations[2].T + pc
            ligand.append(x)
        convert = lambda x: torch.as_tensor(x, device=e3_coords.device, dtype=e3_coords.dtype)
        poi = (poi_coords[0] - convert(cb)) @ convert(rotations[2]).T + convert(pc)
        return {'a': convert(ligand[0])[None], 'b': convert(ligand[1])[None],
                'poi': poi[None], 'body': convert(rotations[2])[None]}, {
                    'init_mode': 'sdpi', 'version': FLOW_VERSION, 'seed': seed,
                    'contract': CONTRACT.copy(), 'no_torsion': no_torsion,
                    'independent_hypotheses': True, 'native_coordinates_used': False}

    def normalized_predictions(self, output, t):
        from .diffusion import so3_score_norm
        from .diffdock.utils import torus
        tr, rot, tor = self.t_to_sigma(t, t, t)
        rot_norm = so3_score_norm(rot).reshape(-1, 1)
        tor_norm = torch.as_tensor(torus.score_norm(tor.detach().cpu().numpy()), device=t.device, dtype=t.dtype).sqrt().reshape(-1, 1)
        converted = {name: {'tr': -tr[:, None] * scores['tr'],
                             'rot': -scores['rot'] / rot_norm,
                             'tor': -scores['tor'] / tor_norm}
                     for name, scores in output['interface_scores'].items()}
        result = {}
        for name in ('e3_ligand', 'poi_ligand'):
            result[name] = {key: converted[name][key] + output['dual_corrections'][name][key]
                            for key in ('tr', 'rot', 'tor')}
        result['poi'] = output['dual_corrections']['poi']
        return result

    def velocities(self, normalized, e3_coords, poi_coords):
        scales = self.scales(e3_coords, poi_coords)
        return {name: {key: val * (scale if key == 'tr' else math.pi)
                       for key, val in normalized[name].items()}
                for name, scale in (('e3_ligand', scales[0]), ('poi_ligand', scales[1]), ('poi', scales.sum()))}

    @staticmethod
    def step(state, velocities, dt, ligand_data, freeze_poi=False):
        if dt.shape != (1,) or not torch.isfinite(dt).all() or not bool((dt >= 0).all()):
            raise ValueError('dual flow dt must be finite nonnegative [1]')
        for key in ('a', 'b', 'poi'):
            _check_coords(state[key])
        for name, components in velocities.items():
            for key, val in components.items():
                if not torch.isfinite(val).all() or (key != 'tor' and val.shape != (1, 3)):
                    raise ValueError(f'invalid {name}.{key} velocity')
        hypotheses = []
        for key, name in (('a', 'e3_ligand'), ('b', 'poi_ligand')):
            v = velocities[name]
            moved, _ = _rigid(state[key], v['tr'], v['rot'], dt)
            hypotheses.append(torsion_step(moved, -dt[:, None] * v['tor'], ligand_data))
        poi, body = state['poi'], state['body']
        if not freeze_poi:
            old_center = poi.mean(1, keepdim=True)
            poi, increment = _rigid(poi, velocities['poi']['tr'], velocities['poi']['rot'], dt)
            hypotheses[1] = (hypotheses[1] - old_center) @ increment.transpose(-1, -2) + poi.mean(1, keepdim=True)
            body = increment @ body
        result = dict(a=hypotheses[0], b=hypotheses[1], poi=poi, body=body)
        for key in ('a', 'b', 'poi'):
            _check_coords(result[key])
        return result


class ConsensusController(nn.Module):
    """Invariant coefficients multiplying equivariant geometric vector bases.

    Three independently controlled bodies: ligand A, ligand B (relative), POI.
    Zero initialization preserves the binary proposals. No confidence gating.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.rigid = nn.Sequential(nn.Linear(hidden_dim * 2 + 14, hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, 54))
        self.torsion = nn.Sequential(nn.Linear(hidden_dim * 2 + 2, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, 2))
        for head in (self.rigid, self.torsion):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, contexts, bond_contexts, a, b, e3, poi, scores, t, rigid_scores=None):
        ca, cb, ce, cp = [x.mean(1) for x in (a, b, e3, poi)]
        raw = torch.stack((ca - ce, cb - cp, cp - ce, cb - ca), dim=1)
        lengths = raw.norm(dim=-1)
        basis = raw / lengths.clamp_min(1.)[..., None]
        axial = torch.stack([torch.cross(basis[:, i], basis[:, j], dim=-1)
                             for i, j in ((0, 1), (0, 2), (0, 3), (1, 2))], dim=1)
        if rigid_scores is None:
            rigid_scores = tuple({'tr_score': ca.new_zeros(1, 3), 'rot_score': ca.new_zeros(1, 3)} for _ in range(2))
        proposals_tr = torch.stack([s['tr_score'].detach() for s in rigid_scores], dim=1)
        proposals_rot = torch.stack([s['rot_score'].detach() for s in rigid_scores], dim=1)
        evidence_norms = torch.cat((proposals_tr.norm(dim=-1), proposals_rot.norm(dim=-1)), dim=-1)
        basis = torch.cat((basis, torch.nn.functional.normalize(proposals_tr, dim=-1)), dim=1)
        axial = torch.cat((axial, torch.nn.functional.normalize(proposals_rot, dim=-1)), dim=1)
        disagreement = (a - b).square().sum(-1).mean(-1, keepdim=True).sqrt()
        schedule = consensus_schedule(t)
        invariants = torch.cat((torch.log1p(lengths), torch.log1p(disagreement), torch.log1p(evidence_norms),
                                t[:, None], torch.stack(tuple(schedule.values()), dim=-1)), dim=-1)
        coeff = self.rigid(torch.cat((*contexts, invariants), dim=-1)).reshape(1, 3, 3, 6)
        # Rigid heads may use both polar and axial bases: SO(3), not O(3).
        result = {}
        for i, name in enumerate(('e3_ligand', 'poi_ligand', 'poi')):
            result[name] = {'tr': (coeff[:, i, 0, :, None] * basis).sum(1),
                            'rot': (coeff[:, i, 1, :, None] * axial).sum(1)
                                   + (coeff[:, i, 2, :, None] * basis).sum(1)}
        n = scores[0].shape[-1]
        if n:
            evidence = torch.stack((torch.tanh(scores[0]), torch.tanh(scores[1])), dim=-1)
            context = torch.cat((*bond_contexts, evidence), dim=-1)
            tor = self.torsion(context)
        else:
            tor = a.new_zeros(1, 0, 2)
        for i, name in enumerate(('e3_ligand', 'poi_ligand')):
            result[name]['tor'] = tor[..., i]
            result[name] = {key: value * (1 - t[:, None]) for key, value in result[name].items()}
        return result


@torch.no_grad()
def dual_sample(model, diffusion, initializer, e3_data, poi_data, ligand_data,
                steps=None, device='cuda', seed=None, return_trajectory=False,
                return_trace=False, freeze_poi=False, init_mode='sdpi', t_anchor=None):
    """Integrate two hypotheses and emit exactly one manifold ligand (A).

    This v1 collapse is deliberately simple: preserve A's valid chemistry and
    the independently predicted POI, report remaining B discrepancy explicitly.
    Consensus is a learned objective, not proof of docking correctness.
    """
    if not isinstance(diffusion, DualConsensusFlow):
        raise ValueError('dual sampling requires DualConsensusFlow')
    if init_mode != 'sdpi' or t_anchor not in (None, 1.0):
        raise ValueError('dual sampling requires full sdpi prior at t=1')
    steps = diffusion.inference_steps if steps is None else steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    no_torsion = bool(model.expert.model.no_torsion)
    if initializer.no_torsion != no_torsion:
        raise ValueError('initializer/model no_torsion mismatch')
    if seed is None:
        import secrets
        seed = secrets.randbelow(2**32)
    model.eval()
    state, metadata = diffusion.initialize(e3_data['coords'], poi_data['coords'], ligand_data, seed, no_torsion)
    if freeze_poi:
        state['b'] = (state['b'] - state['poi'].mean(1, keepdim=True)) @ state['body'] + poi_data['coords'].mean(1, keepdim=True)
        state['poi'] = poi_data['coords'].clone()
        state['body'] = torch.eye(3, device=state['a'].device, dtype=state['a'].dtype)[None]
    metadata.update(freeze_poi=freeze_poi, generative_process=FLOW_VERSION,
                    expert=getattr(model.expert, 'provenance', {}))
    trace, trajectory, rejected = [], [], []
    def discrepancy(current):
        return float((current['a'] - current['b']).square().sum(-1).mean().sqrt())
    trace.append({'stage': 'init', 'initialization': metadata, 'consensus_rmsd': discrepancy(state)})
    for i in range(steps):
        t = state['a'].new_tensor([1 - i / steps])
        dt = torch.full_like(t, 1 / steps)
        tr_sigma, _, _ = diffusion.t_to_sigma(t, t, t)
        output = model(e3_data, {**poi_data, 'coords': state['poi']}, {**ligand_data, 'coords': state['a']},
                       t_e3l=t, t_pol=t, ligand_b_coords=state['b'], r_e3_to_poi=state['body'],
                       diffusion_state=diffusion.conditioning(t, dt, e3_data['coords'], poi_data['coords']),
                       receptor_crop_cutoff=tr_sigma * 3 + 20 if model.expert.model.dynamic_max_cross else None)
        velocity = diffusion.velocities(diffusion.normalized_predictions(output, t), e3_data['coords'], poi_data['coords'])
        bad = [f'{name}.{key}' for name, components in velocity.items() for key, v in components.items() if not torch.isfinite(v).all()]
        if bad:
            rejected.append({'step': i, 'components': bad})
            break
        state = diffusion.step(state, velocity, dt, ligand_data, freeze_poi)
        if return_trace:
            trace.append({'stage': 'step', 'step': i, 't': float(t), 'dt': float(dt),
                          'consensus_rmsd': discrepancy(state),
                          'consensus_weights': {k: float(v) for k, v in consensus_schedule(t).items()},
                          'prediction_units': 'angstrom/time, radians/time',
                          'finite': all(bool(torch.isfinite(x).all()) for x in state.values()),
                          'poi_rotation_det': float(torch.det(state['body'][0]))})
        if return_trajectory:
            trajectory.append((state['a'][0].cpu().numpy(), state['poi'][0].cpu().numpy()))
    result = {'ligand_coords': state['a'][0].cpu().numpy(), 'poi_coords': state['poi'][0].cpu().numpy(),
              'hypothesis_b_coords': state['b'][0].cpu().numpy(), 'consensus_rmsd': discrepancy(state),
              'initialization': metadata, 'valid': not rejected, 'nonfinite_score_steps': rejected,
              'termination_reason': 'nonfinite_score' if rejected else 'completed',
              'collapse': CONTRACT['collapse']}
    if return_trace:
        result['trace'] = trace
    if return_trajectory:
        result['trajectory'] = trajectory
    return result


def torsion_features(coords, data):
    """Periodic (cos,sin) features with canonical neighbors on each bond."""
    mol, bonds, _ = chemistry(data)
    features = []
    for u, v in bonds:
        left = min(a.GetIdx() for a in mol.GetAtomWithIdx(int(u)).GetNeighbors() if a.GetIdx() != v)
        right = min(a.GetIdx() for a in mol.GetAtomWithIdx(int(v)).GetNeighbors() if a.GetIdx() != u)
        axis = torch.nn.functional.normalize(coords[:, v] - coords[:, u], dim=-1)
        x, y = coords[:, left] - coords[:, u], coords[:, right] - coords[:, v]
        x = torch.nn.functional.normalize(x - (x * axis).sum(-1, keepdim=True) * axis, dim=-1)
        y = torch.nn.functional.normalize(y - (y * axis).sum(-1, keepdim=True) * axis, dim=-1)
        features.append(torch.stack(((x * y).sum(-1), (torch.cross(x, y, dim=-1) * axis).sum(-1)), dim=-1))
    return torch.stack(features, dim=1) if features else coords.new_zeros(1, 0, 2)


def dual_loss(trainer, e3_data, poi_data, ligand_data, case='unknown',
              t_override=None, t_pol_override=None, coupling_override=None, diagnostics=False):
    """Velocity matching plus annealed clean-estimate and global supervision."""
    flow, model = trainer.diffusion, trainer.model
    if e3_data['coords'].shape[0] != 1:
        raise ValueError('dual flow training requires B=1')
    t = torch.rand(1, device=trainer.device) if t_override is None else torch.as_tensor(
        t_override, device=trainer.device, dtype=e3_data['coords'].dtype).reshape(-1)
    if t_pol_override is not None and not torch.equal(t, torch.as_tensor(t_pol_override, device=t.device, dtype=t.dtype).reshape(-1)):
        raise ValueError('dual flow requires one shared clock')
    no_torsion = bool(getattr(model.expert.model, 'no_torsion', False))
    coupling = coupling_override if coupling_override is not None else flow.sample_coupling(
        e3_data['coords'], poi_data['coords'], ligand_data, no_torsion)
    state, target = flow.path(ligand_data, poi_data['coords'], coupling, t)
    dt = torch.full_like(t, 1 / flow.inference_steps)
    tr_sigma, _, _ = flow.t_to_sigma(t, t, t)
    output = model(e3_data, {**poi_data, 'coords': state['poi']}, {**ligand_data, 'coords': state['a']},
                   t_e3l=t, t_pol=t, ligand_b_coords=state['b'], r_e3_to_poi=state['body'],
                   diffusion_state=flow.conditioning(t, dt, e3_data['coords'], poi_data['coords']),
                   receptor_crop_cutoff=tr_sigma * 3 + 20 if model.expert.model.dynamic_max_cross else None)
    pred = flow.normalized_predictions(output, t)
    scales = flow.scales(e3_data['coords'], poi_data['coords'])
    losses, metrics = {}, {'projection_rmsd': coupling['projection_rmsd']}
    for name, prefix, scale in (('e3_ligand', 'e3l', scales[0]), ('poi_ligand', 'pol', scales[1]), ('poi', 'poi', scales.sum())):
        for key in pred[name]:
            expected = target[name][key] / (scale if key == 'tr' else math.pi)
            if key == 'tor' and no_torsion:
                expected = expected[:, :0]
            value = pred[name][key]
            if value.shape != expected.shape:
                raise ValueError(f'{name}.{key} shape mismatch')
            losses[f'loss_{prefix}_{key}'] = (value - expected).square().mean() if value.numel() else t.new_zeros(())
            if key == 'tor' and value.numel():
                raw = value - output['dual_corrections'][name]['tor']
                losses[f'loss_{prefix}_tor_aux'] = trainer.torsion_expert_aux_weight * (raw - expected).square().mean()
            if key == 'rot':
                cosine = torch.nn.functional.cosine_similarity(value, expected, dim=-1).mean()
                metrics[f'{prefix}_rot_cos'] = cosine.detach().item()
                metrics[f'{prefix}_rot_angle_deg'] = torch.acos(cosine.clamp(-1, 1)).detach().item() * 180 / math.pi
            if diagnostics:
                metrics[f'{prefix}_{key}_prediction_rms'] = value.detach().square().mean().sqrt().item() if value.numel() else 0.
                metrics[f'{prefix}_{key}_target_rms'] = expected.detach().square().mean().sqrt().item() if expected.numel() else 0.
    # A differentiable clean estimate, used as an auxiliary objective. This is
    # not assumed to be the exact endpoint of a time-varying learned field.
    velocity = flow.velocities(pred, e3_data['coords'], poi_data['coords'])
    clean_est = flow.step(state, velocity, t, ligand_data)
    a, b, p = clean_est['a'], clean_est['b'], clean_est['poi']
    relative_trace = clean_est['body'].diagonal(dim1=-2, dim2=-1).sum(-1)
    metrics['poi_pose_rotation_error_deg'] = float(torch.acos(((relative_trace - 1) / 2).clamp(-1, 1)).detach().mean() * 180 / math.pi)
    clean = coupling['clean']
    schedule = consensus_schedule(t)
    distance_a, distance_b = torch.cdist(a, a), torch.cdist(b, b)
    losses['loss_cons_internal'] = schedule['internal'].mean() * (distance_a - distance_b).square().mean() / 25
    tor_a, tor_b = torsion_features(a, ligand_data), torsion_features(b, ligand_data)
    losses['loss_cons_torsion'] = schedule['torsion'].mean() * ((tor_a - tor_b).square().mean() if tor_a.numel() else t.new_zeros(()))
    # Graph diameter endpoints work without native warhead annotations, including
    # MGD. They are topology diagnostics, not asserted chemical warhead labels.
    from rdkit import Chem
    mol, _, _ = chemistry(ligand_data)
    end = np.unravel_index(np.argmax(Chem.GetDistanceMatrix(mol)), (mol.GetNumAtoms(),) * 2)
    endpoint_a = a[:, end[0]] - a[:, end[1]]
    endpoint_b = b[:, end[0]] - b[:, end[1]]
    losses['loss_cons_endpoint'] = schedule['endpoint'].mean() * (endpoint_a - endpoint_b).square().mean() / 25
    losses['loss_cons_pose'] = schedule['pose'].mean() * (a - b).square().mean() / 25
    # Global losses have their own native/reachable targets: agreement alone
    # cannot minimize these terms. POI velocities above are explicitly supervised.
    weight = (1 - t).square().mean()
    losses['loss_global_pose'] = weight * ((p - poi_data['coords']).square().mean() / scales.sum().square()
                                           + 0.5 * ((a - clean).square().mean() + (b - clean).square().mean()) / 25)
    # Uniform deterministic receptor subsampling caps pair memory, not labels.
    def residues(x):
        index = torch.linspace(0, x.shape[1] - 1, min(x.shape[1], 64), device=x.device).long()
        return x[:, index]
    e, pr, pn = residues(e3_data['coords']), residues(p), residues(poi_data['coords'])
    def contact(x, y):
        return torch.sigmoid((8 - torch.cdist(x, y)) / 2)
    losses['loss_global_contacts'] = weight * ((contact(e, pr) - contact(e, pn)).square().mean()
        + (contact(pr, a) - contact(pn, clean)).square().mean()
        + (contact(e, a) - contact(e, clean)).square().mean())
    metrics['consensus_rmsd'] = (a - b).square().sum(-1).mean().sqrt().detach().item()
    stage = model.dual_stage
    if stage in ('independent', 'exchange'):
        losses = {k: v for k, v in losses.items() if k.startswith(('loss_e3l_', 'loss_pol_'))}
    elif stage == 'soft':
        losses = {k: v for k, v in losses.items() if not k.startswith(('loss_global_', 'loss_poi_')) and k != 'loss_cons_pose'}
    total = sum(losses.values())
    if not torch.isfinite(total):
        from .train import NonFiniteTrainingError
        raise NonFiniteTrainingError(f'{case}: nonfinite dual consensus loss')
    if diagnostics:
        return {'loss_total': total, 'components': losses, 'rms': metrics}
    return {'loss_total': total, 'loss_score': total.detach().item(),
            **{k: v.detach().item() for k, v in losses.items()}, **metrics}
