"""Conditional flow matching on dual poses and shared ligand torsions.

Time is clean=0, SDPI=1. Native crystal coordinates remain the training target.
Endpoint equivalence is on movable DOFs conditional on rigid-fragment geometry,
NOT an assertion that ETKDG and crystal internal geometry distributions agree.
DiffDock sigmas below are only pretrained feature/output preconditioners.
"""

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .diffusion import InterfaceDiffusion, so3_score_norm
from .initialization import receptor_prior, sample_binary_pose, validate_ligand_graph
from .rigid import assemble_interface_pair


FLOW_VERSION = 'sdpi_flow_v1'
CHECKPOINT_VERSION = 17


def _check_coords(coords):
    if (coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 3
            or coords.shape[1] == 0 or not torch.isfinite(coords).all()):
        raise ValueError('SDPI flow coordinates must be finite [1, N, 3] with N > 0')


def _rotation_exp(vector):
    """SO(3) exponential without small-angle division or dtype conversion."""
    matrix = vector.new_zeros((*vector.shape[:-1], 3, 3))
    x, y, z = vector.unbind(-1)
    matrix[..., 0, 1], matrix[..., 0, 2] = -z, y
    matrix[..., 1, 0], matrix[..., 1, 2] = z, -x
    matrix[..., 2, 0], matrix[..., 2, 1] = -y, x
    return torch.linalg.matrix_exp(matrix)


def _align(moving, target):
    """Proper Kabsch fit in float64, row-vector convention."""
    cm, ct = moving.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((moving - cm).T @ (target - ct))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = (u @ correction @ vt).T
    return (moving - cm) @ rotation.T + ct


def _raw_torsions(coords, bonds, masks, angles):
    from .diffdock.utils.torsion import modify_conformer_torsion_angles

    if not len(bonds):
        return coords.copy()
    return modify_conformer_torsion_angles(
        coords.copy(), torch.as_tensor(bonds), masks, angles, as_numpy=True,
    )


def _torsion_shape(coords, bonds, masks, angles):
    return _align(_raw_torsions(coords, bonds, masks, angles), coords)


def _horizontal_torsion_velocity(coords, bonds, masks, angular_velocity):
    """Horizontal torsion velocity and the removed rigid angular velocity."""
    velocity = np.zeros_like(coords)
    for (u, v), mask, speed in zip(bonds, masks, angular_velocity):
        # Vendored convention: axis points v -> u.
        axis = coords[u] - coords[v]
        length = np.linalg.norm(axis)
        if length < 1e-10:
            raise ValueError('degenerate torsion axis')
        velocity[mask] += speed * np.cross(axis / length, coords[mask] - coords[v])
    velocity -= velocity.mean(0)
    x = coords - coords.mean(0)
    inertia = np.eye(3) * np.sum(x * x) - x.T @ x
    spin = np.linalg.pinv(inertia, rcond=1e-12) @ np.cross(x, velocity).sum(0)
    return velocity - np.cross(spin, x), spin


def _shape_and_gauge(coords, bonds, masks, angles, time):
    """Smooth raw torsion path, recentered, with analytic solver-gauge spin.

    Do not fit every path state back to the crystal: a global Kabsch chart can
    become singular for heavily folded conformers. Raw bond rotations are
    smooth. The solver removes their instantaneous rigid spin, so include
    that spin in BOTH interface angular targets. No finite differences enter
    training targets; full Cartesian finite differences are independent tests.
    """
    shape = _raw_torsions(coords, bonds, masks, time * angles)
    shape += coords.mean(0) - shape.mean(0)
    _, spin = _horizontal_torsion_velocity(shape, bonds, masks, angles)
    return shape, spin


class SDPIFlow:
    """The production transport contract; there is no temperature/SDE branch."""

    version = FLOW_VERSION

    def __init__(self, inference_steps=20):
        if not isinstance(inference_steps, int) or isinstance(inference_steps, bool) or inference_steps < 1:
            raise ValueError('inference_steps must be a positive integer')
        self.inference_steps = inference_steps
        self._expert_conditioning = InterfaceDiffusion(inference_steps=inference_steps)

    def t_to_sigma(self, t_tr, t_rot, t_tor):
        """Native expert preconditioning; these are NOT flow noise variances."""
        return self._expert_conditioning.t_to_sigma(t_tr, t_rot, t_tor)

    def get_inference_schedule(self, steps, t_start=1.0):
        if t_start != 1:
            raise ValueError('SDPI flow must start at its prior endpoint t=1')
        return self._expert_conditioning.get_inference_schedule(steps, t_start)

    @staticmethod
    def scales(e3_coords, poi_coords):
        _check_coords(e3_coords)
        _check_coords(poi_coords)
        return torch.stack((receptor_prior(e3_coords[0])[2],
                            receptor_prior(poi_coords[0])[2]))

    def conditioning(self, t, dt, e3_coords, poi_coords):
        _, rot, tor = self.t_to_sigma(t, t, t)
        scales = self.scales(e3_coords, poi_coords)
        return torch.cat((torch.log(torch.stack((
            scales[0].expand_as(t), rot, tor, scales[1].expand_as(t), rot,
        ), dim=-1)), dt.reshape(-1, 1).expand(-1, 3), t.unsqueeze(-1)), dim=-1)

    def normalized_predictions(self, output, t):
        """Convert pretrained score units to dimensionless velocity outputs.

        This is a trainable function parameterization, not a score-to-flow
        identity. LoRA/TriFuse must be retrained under the velocity loss.
        """
        if 'normalized_velocity' in output:
            return output['normalized_velocity']
        from .diffdock.utils import torus

        tr, rot, tor = self.t_to_sigma(t, t, t)
        rot_norm = so3_score_norm(rot).reshape(-1, 1)
        tor_norm = torch.as_tensor(
            torus.score_norm(tor.detach().cpu().numpy()), device=t.device, dtype=t.dtype,
        ).sqrt().reshape(-1, 1)
        result = {}
        for branch, scores in output['interface_scores'].items():
            result[branch] = {'tr': -tr[:, None] * scores['tr'],
                              'rot': -scores['rot'] / rot_norm,
                              'tor': -scores['tor'] / tor_norm}
        result['torsion'] = -output['torsion_score'] / tor_norm
        return result

    def velocities(self, normalized, e3_coords, poi_coords):
        scales = self.scales(e3_coords, poi_coords)
        result = {}
        for i, name in enumerate(('e3_ligand', 'poi_ligand')):
            result[name] = {'tr': normalized[name]['tr'] * scales[i],
                            'rot': normalized[name]['rot'] * math.pi}
        result['torsion'] = normalized['torsion'] * math.pi
        return result

    @staticmethod
    def sample_coupling(e3_coords, poi_coords, ligand_data, no_torsion=False):
        """Draw the same independent Haar/Gaussian/uniform kernels as SDPI."""
        from rdkit import Chem

        for coords in (e3_coords, poi_coords, ligand_data['coords']):
            _check_coords(coords)
        mol = ligand_data['mol']
        if isinstance(mol, bytes):
            mol = Chem.Mol(mol)
        bonds, masks = validate_ligand_graph(
            mol, ligand_data['edge_index'], ligand_data['edge_mask'], ligand_data['mask_rotate'],
        )
        native = ligand_data['coords'][0]
        if native.shape != (mol.GetNumAtoms(), 3) or not torch.isfinite(native).all():
            raise ValueError('invalid indexed native ligand coordinates')
        # Native geometry is only the TRAINING target/reference. Inference
        # still independently embeds ETKDG and never reads this conformer.
        angles = np.zeros(len(bonds)) if no_torsion else np.random.uniform(-np.pi, np.pi, len(bonds))
        omegas, centers = [], []
        for receptor in (e3_coords[0], poi_coords[0]):
            rotation, _, metadata = sample_binary_pose(native, receptor)
            omegas.append(Rotation.from_matrix(rotation.detach().cpu().double().numpy()).as_rotvec())
            centers.append(np.array(metadata['receptor_center']) + np.array(metadata['delta']))
        return {'angles': angles, 'omegas': np.array(omegas), 'centers': np.array(centers),
                'bonds': bonds.numpy(), 'masks': masks}

    def path(self, ligand_data, poi_coords, coupling, t):
        """One B=1 training state and its forward (clean -> prior) velocity."""
        _check_coords(ligand_data['coords'])
        _check_coords(poi_coords)
        if t.shape != (1,) or not torch.isfinite(t).all() or not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError('flow time must be one finite value in [0, 1]')
        native = ligand_data['coords'][0]
        dtype, device = native.dtype, native.device
        value = float(t.item())
        clean = native.detach().cpu().double().numpy()
        shape, gauge = _shape_and_gauge(clean, coupling['bonds'], coupling['masks'], coupling['angles'], value)
        convert = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        rotations = convert(Rotation.from_rotvec(value * coupling['omegas']).as_matrix())
        center0 = clean.mean(0)
        centers = convert((1 - value) * center0 + value * coupling['centers'])
        offsets = centers - (rotations @ convert(center0)[:, None]).squeeze(-1)
        ligand = convert(shape) @ rotations[0].T + offsets[0]
        poi, body = assemble_interface_pair(poi_coords[0], rotations[0], offsets[0], rotations[1], offsets[1])
        tr = convert(coupling['centers'] - center0)
        rot = convert(coupling['omegas']) + (rotations @ convert(gauge)[:, None]).squeeze(-1)
        # Expert B emits body-local vectors mapped into the E3/global frame.
        target = {'e3_ligand': {'tr': tr[0:1], 'rot': rot[0:1]},
                  'poi_ligand': {'tr': tr[1:2] @ body.T, 'rot': rot[1:2] @ body.T},
                  'torsion': convert(coupling['angles'])[None]}
        return ligand[None], poi[None], body[None], target

    @staticmethod
    def step(ligand, poi, body, velocities, dt, ligand_data, freeze_poi=False):
        """Reverse Lie-Euler step; dt>0 means integrate from t to t-dt."""
        _check_coords(ligand)
        _check_coords(poi)
        if dt.shape != (1,) or not torch.isfinite(dt).all() or not bool((dt > 0).all()):
            raise ValueError('flow step must be one finite positive dt')
        old_center = ligand.mean(1)
        transforms = []
        for branch in ('e3_ligand', 'poi_ligand'):
            velocity = velocities[branch]
            for component in ('tr', 'rot'):
                if velocity[component].shape != (1, 3) or not torch.isfinite(velocity[component]).all():
                    raise ValueError('rigid velocities must be finite [1, 3]')
            rotation = _rotation_exp(-dt[:, None] * velocity['rot'])
            center = old_center - dt[:, None] * velocity['tr']
            offset = center - (rotation @ old_center[..., None]).squeeze(-1)
            transforms.append((rotation, offset))
        (ra, ta), (rb, tb) = transforms
        rigid = ligand @ ra.transpose(-1, -2) + ta[:, None]
        if not freeze_poi:
            poi, increment = assemble_interface_pair(poi, ra, ta, rb, tb)
            body = increment @ body
        torsion = velocities['torsion']
        n_bonds = int(ligand_data['edge_mask'].sum())
        if (torsion.ndim != 2 or torsion.shape[0] != 1 or torsion.shape[1] not in (0, n_bonds)
                or not torch.isfinite(torsion).all()):
            raise ValueError('torsion velocities must be finite [1, n_rot] (or empty for no_torsion)')
        angles = (-dt[:, None] * torsion).detach().cpu().double().numpy()[0]
        if len(angles):
            bonds = ligand_data['edge_index'].T[ligand_data['edge_mask']].detach().cpu().numpy()
            masks = ligand_data['mask_rotate']
            if isinstance(masks, (list, tuple)):
                masks = masks[0]
            if torch.is_tensor(masks):
                masks = masks.detach().cpu().numpy()
            moved = _torsion_shape(rigid[0].detach().cpu().double().numpy(), bonds, masks, angles)
            rigid = torch.as_tensor(moved, dtype=ligand.dtype, device=ligand.device)[None]
        _check_coords(rigid)
        _check_coords(poi)
        if not torch.isfinite(body).all():
            raise FloatingPointError('non-finite flow body frame')
        return rigid, poi, body
