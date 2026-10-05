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

from tricoopdiff.models.expert import expert_t_to_sigma, so3_score_norm
from tricoopdiff.initialization import receptor_prior, sample_binary_pose, validate_ligand_graph
from tricoopdiff.geometry.rigid import (apply_rigid, assemble_interface_pair,
                                      is_proper_rotation, proper_rigid_fit, rotation_exp)


FLOW_VERSION = 'sdpi_flow_v1'
CHECKPOINT_VERSION = 17
INDEPENDENT_RIGID_MODE = 'independent_centroids_v1'
LEGACY_RIGID_MODE = 'dual_interfaces_v1'
CONTINUOUS_TORSION_MODE = 'continuous_horizontal_v1'
SYMMETRIC_TORSION_MODE = 'symmetric_horizontal_v1'


def _check_coords(coords):
    if (coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[2] != 3
            or coords.shape[1] == 0 or not torch.isfinite(coords).all()):
        raise ValueError('SDPI flow coordinates must be finite [1, N, 3] with N > 0')


def _raw_torsions(coords, bonds, masks, angles):
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    if not len(bonds):
        return coords.copy()
    return modify_conformer_torsion_angles(
        coords.copy(), torch.as_tensor(bonds), masks, angles, as_numpy=True,
    )


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

    def t_to_sigma(self, t_tr, t_rot, t_tor):
        """Native expert preconditioning; these are NOT flow noise variances."""
        return expert_t_to_sigma(t_tr, t_rot, t_tor)

    def get_inference_schedule(self, steps, t_start=1.0):
        if t_start != 1:
            raise ValueError('SDPI flow must start at its prior endpoint t=1')
        return self.get_refinement_schedule(steps, t_start)

    def get_refinement_schedule(self, steps, t_start):
        """Return a truncated clock for an externally generated fixed proposal.

        This deliberately does not relax ``get_inference_schedule``: ordinary
        SDPI samples must still start at their prior endpoint. The caller is
        responsible for restricting this schedule to fixed native-free seeds.
        """
        if not isinstance(t_start, (int, float)) or not 0.0 < float(t_start) <= 1.0:
            raise ValueError('refinement t_start must be in (0, 1]')
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('refinement steps must be a positive integer')
        clock = np.linspace(float(t_start), 0., steps + 1)
        return {name + '_schedule': clock.copy() for name in ('tr', 'rot', 'tor')}

    def reverse_times(self, reference, steps=None, t_start=1.0):
        """One shared deterministic clock for endpoint training and sampling."""
        steps = self.inference_steps if steps is None else steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('reverse steps must be a positive integer')
        if not math.isfinite(float(t_start)) or not 0 < float(t_start) <= 1:
            raise ValueError('reverse start time must be in (0, 1]')
        clock = torch.linspace(float(t_start), 0., steps + 1,
                               device=reference.device, dtype=reference.dtype)
        for index in range(steps):
            yield clock[index:index + 1], (clock[index] - clock[index + 1]).reshape(1)

    def forward_state(self, model, e3_data, poi_data, ligand_data,
                      ligand, poi, body, t, dt, source=0, conditions=None):
        return self.forward_states(model, [(e3_data, poi_data, ligand_data, ligand,
                                           poi, body, t, dt, source, conditions)])[0]

    def forward_states(self, model, states):
        """One shared clock/geometry contract for single and ragged batch execution."""
        requests = [self._forward_inputs(model, *state) for state in states]
        outputs = (model.forward_many(requests) if hasattr(model, 'forward_many')
                   else [model(request['e3_data'], request['poi_data'], request['ligand_data'],
                               **{key: value for key, value in request.items()
                                  if key not in ('e3_data', 'poi_data', 'ligand_data')})
                         for request in requests])
        for output in outputs:
            self._validate_output(model, output)
        return outputs

    def _forward_inputs(self, model, e3_data, poi_data, ligand_data,
                        ligand, poi, body, t, dt, source=0, conditions=None):
        """Read the same current geometry and expert clock in both execution paths.

        Template coordinates supply fixed conditioning scales; moving coordinates
        supply graph geometry. Input dictionaries are never mutated.
        """
        conditioning = self.conditioning(t, dt, e3_data['coords'], poi_data['coords'])
        crop = None
        if bool(getattr(model.expert.model, 'dynamic_max_cross', False)):
            tr_sigma, _, _ = self.t_to_sigma(t, t, t)
            crop = tr_sigma * 3 + 20
        condition_kwargs = {'conditions': conditions} if conditions is not None else {}
        return dict(e3_data=e3_data, poi_data={**poi_data, 'coords': poi},
                    ligand_data={**ligand_data, 'coords': ligand}, t_e3l=t, t_pol=t,
                    r_e3_to_poi=body, diffusion_state=conditioning,
                    receptor_crop_cutoff=crop, source=source, **condition_kwargs)

    @staticmethod
    def _validate_output(model, output):
        if (getattr(getattr(model, 'trifuse', None), 'direct_ternary_motion', False)
                and output.get('normalized_velocity', {}).get('rigid_output_mode')
                != INDEPENDENT_RIGID_MODE):
            raise ValueError('direct model must execute independent centroid motion')
        symmetric = getattr(getattr(model, 'trifuse', None), 'symmetric_torsion', False)
        if ((symmetric or getattr(getattr(model, 'trifuse', None), 'continuous_torsion', False))
                and output.get('normalized_velocity', {}).get('torsion_update_mode')
                != (SYMMETRIC_TORSION_MODE if symmetric else CONTINUOUS_TORSION_MODE)):
            raise ValueError('continuous torsion model must execute its horizontal update')

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

    @staticmethod
    def apply_output(ligand, poi, body, velocities, dt, ligand_data, output, freeze_poi=False):
        """Execute the shared rigid/torsion update used by training and sampling."""
        return SDPIFlow.differentiable_step(
            ligand, poi, body, velocities, dt, ligand_data, freeze_poi)

    def normalized_predictions(self, output, t):
        """Convert pretrained score units to dimensionless velocity outputs.

        This is a trainable function parameterization, not a score-to-flow
        identity. LoRA/TriFuse must be retrained under the velocity loss.
        """
        if 'normalized_velocity' in output:
            return output['normalized_velocity']
        from tricoopdiff.diffdock.utils import torus

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
        if 'rigid_output_mode' in normalized:
            result['rigid_output_mode'] = normalized['rigid_output_mode']
        if 'torsion_update_mode' in normalized:
            result['torsion_update_mode'] = normalized['torsion_update_mode']
        return result

    @staticmethod
    def independent_targets(targets, ligand, poi):
        """Express the same forward path in ligand/POI centroid coordinates."""
        a, b = targets['e3_ligand'], targets['poi_ligand']
        angular = a['rot'] - b['rot']
        lever = poi.mean(1) - ligand.mean(1)
        return {'e3_ligand': dict(a),
                'poi_ligand': {'tr': a['tr'] - b['tr'] + torch.cross(angular, lever, dim=-1),
                               'rot': angular},
                'torsion': targets['torsion'],
                'rigid_output_mode': INDEPENDENT_RIGID_MODE}

    @staticmethod
    def rigid_step(ligand, poi, body, velocities, dt, freeze_poi=False):
        """Shared differentiable rigid decoder for production and training."""
        mode = velocities.get('rigid_output_mode', LEGACY_RIGID_MODE)
        if mode not in (LEGACY_RIGID_MODE, INDEPENDENT_RIGID_MODE):
            raise ValueError(f'unknown rigid output mode: {mode!r}')
        transforms = []
        for index, branch in enumerate(('e3_ligand', 'poi_ligand')):
            velocity = velocities[branch]
            for component in ('tr', 'rot'):
                if velocity[component].shape != (1, 3) or not torch.isfinite(velocity[component]).all():
                    raise ValueError('rigid velocities must be finite [1, 3]')
            pivot = (poi if mode == INDEPENDENT_RIGID_MODE and index == 1 else ligand).mean(1)
            rotation = rotation_exp(-dt[:, None] * velocity['rot'])
            offset = pivot - dt[:, None] * velocity['tr'] - (rotation @ pivot[..., None]).squeeze(-1)
            transforms.append((rotation, offset))
        (ra, ta), (rb, tb) = transforms
        rigid = ligand @ ra.transpose(-1, -2) + ta[:, None]
        if not freeze_poi:
            if mode == INDEPENDENT_RIGID_MODE:
                poi = poi @ rb.transpose(-1, -2) + tb[:, None]
                body = rb @ body
            else:
                poi, increment = assemble_interface_pair(poi, ra, ta, rb, tb)
                body = increment @ body
        return rigid, poi, body

    @staticmethod
    def differentiable_step(ligand, poi, body, velocities, dt, ligand_data,
                            freeze_poi=False):
        """Execute the production rigid/torsion contract without detaching inputs."""
        _check_coords(ligand)
        _check_coords(poi)
        if (body.shape != (1, 3, 3) or not is_proper_rotation(body)
                or dt.shape != (1,) or not torch.isfinite(dt).all()
                or not bool((dt > 0).all())):
            raise ValueError('body must be a proper [1,3,3] rotation and dt finite positive [1]')
        rigid, poi, body = SDPIFlow.rigid_step(ligand, poi, body, velocities, dt, freeze_poi)
        _check_coords(rigid)
        _check_coords(poi)
        if not torch.isfinite(body).all():
            raise FloatingPointError('non-finite flow body frame')
        torsion = velocities['torsion']
        torsion_mode = velocities.get('torsion_update_mode')
        if torsion_mode not in (None, CONTINUOUS_TORSION_MODE, SYMMETRIC_TORSION_MODE):
            raise ValueError(f'unknown torsion update mode: {torsion_mode!r}')
        # A no_torsion expert can disable torsions even when the chemical graph
        # contains rotatable bonds. Match the production empty-output contract.
        if torsion.shape == (1, 0):
            return rigid, poi, body
        bonds = ligand_data['edge_index'].T[ligand_data['edge_mask']].to(
            device=ligand.device, dtype=torch.long)
        masks = ligand_data['mask_rotate']
        if isinstance(masks, (list, tuple)):
            masks = masks[0]
        if masks is None and not len(bonds):
            masks = torch.empty(0, ligand.shape[1], dtype=torch.bool, device=ligand.device)
        else:
            masks = torch.as_tensor(masks, dtype=torch.bool, device=ligand.device)
        if (torsion.shape != (1, len(bonds)) or not torch.isfinite(torsion).all()
                or masks.shape != (len(bonds), ligand.shape[1])):
            raise ValueError('finite torsion velocities and ligand graph sizes must agree')
        raw = rigid
        symmetric = torsion_mode == SYMMETRIC_TORSION_MODE
        displacement = torch.zeros_like(rigid) if torsion_mode in (CONTINUOUS_TORSION_MODE, SYMMETRIC_TORSION_MODE) else None
        gauge_prior = torch.zeros_like(rigid[:, 0]) if displacement is not None else None
        for index, (u, v) in enumerate(bonds):
            axis = raw[:, u] - raw[:, v]
            norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
            if bool((norm <= 1e-10).any()):
                raise ValueError('degenerate torsion axis')
            if displacement is not None:
                initial_axis = rigid[:, u] - rigid[:, v]
                initial_axis = initial_axis / torch.linalg.vector_norm(initial_axis, dim=-1, keepdim=True)
                angle = initial_axis * (-dt * torsion[:, index])[:, None]
                if not symmetric:
                    gauge_prior = gauge_prior + .5 * angle
                pivot = (rigid[:, u:u + 1] + rigid[:, v:v + 1]) * .5 if symmetric else rigid[:, v:v + 1]
                side = masks[index].to(rigid.dtype) - (.5 if symmetric else 0.)
                displacement = displacement + torch.linalg.cross(
                    angle[:, None].expand_as(rigid), rigid - pivot, dim=-1,
                ) * side[None, :, None]
            rotation = rotation_exp(axis / norm * (-dt * torsion[:, index])[:, None]
                                    * (.5 if symmetric else 1.))
            pivot = (raw[:, u:u + 1] + raw[:, v:v + 1]) * .5 if symmetric else raw[:, v:v + 1]
            moved = (raw - pivot) @ rotation.transpose(-1, -2) + pivot
            # Reversing the axis and complementing the mask swaps the two
            # half-rotations exactly, not merely their infinitesimal tangent.
            stationary = (raw - pivot) @ rotation + pivot if symmetric else raw
            raw = torch.where(masks[index][None, :, None], moved, stationary)
        if len(bonds):
            if displacement is None:
                rotation, translation, _ = proper_rigid_fit(raw, rigid)
                rigid = apply_rigid(raw, rotation, translation)
            else:
                # An exact finite Kabsch gauge is discontinuous at legitimate
                # folded conformers with rank-deficient cross-covariance.
                # Instead retract the local horizontal torsion tangent: remove
                # its mean motion and rigid spin, without changing shape DOFs.
                center = rigid.mean(1, keepdim=True)
                centered = (rigid - center).double()
                identity = torch.eye(3, dtype=torch.float64, device=rigid.device)[None]
                inertia = centered.square().sum((1, 2))[:, None, None] * identity
                inertia = inertia - centered.transpose(1, 2) @ centered
                torque = torch.linalg.cross(centered, displacement.double(), dim=-1).sum(1)
                # The symmetric finite chart already removes half-side spin;
                # its zero prior is the same tangent as the historical chart.
                # Symmetric half-side gauge keeps the legacy ridge prior covariant
                # under bond reversal plus complementary rotation fragments.
                # A zero-spin prior would leave an orientation-dependent
                # first-order residual, even though the ideal projection does not.
                ridge = inertia.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(torch.finfo(torch.float64).tiny) * 1e-6
                spin = torch.linalg.solve(inertia + ridge[:, None, None] * identity,
                                          (torque + ridge[:, None] * gauge_prior.double())[:, :, None]).squeeze(-1)
                rotation = rotation_exp(-spin).to(rigid.dtype)
                rigid = (raw - raw.mean(1, keepdim=True)) @ rotation.transpose(-1, -2) + center
        _check_coords(rigid)
        _check_coords(poi)
        return rigid, poi, body

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
        """Inference and training execute the same reverse Lie-Euler decoder."""
        return SDPIFlow.differentiable_step(
            ligand, poi, body, velocities, dt, ligand_data, freeze_poi)
