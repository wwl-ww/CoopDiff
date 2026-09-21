"""One known binary pose; unobserved POI retains six free rigid-body DOFs.

Geometry only: this module is not yet connected to the two-sided trainer.
No second warhead or protein-relative pose is fabricated.
"""

import math

import torch

from .anchored_flow import wrap_angles
from .anchored_torsion import AnchoredTorsionPlan, chemical_rotatable_pairs
from .bounded_binary_flow import ball_map, inverse_ball, rotation_exp
from .pose_anchors import proper_rigid_fit


def compile_single_anchor_torsions(molecule, edge_index, edge_mask, anchor):
    """Keep chemical bridge torsions that do not split the one known core."""
    from rdkit import Chem

    mol = Chem.Mol(molecule)
    count, core = mol.GetNumAtoms(), set(anchor)
    edges, mask = torch.as_tensor(edge_index).cpu(), torch.as_tensor(edge_mask).cpu()
    if (len(core) < 3 or len(core) != len(anchor)
            or any(type(i) is not int or not 0 <= i < count for i in core)
            or edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
            or mask.shape != (edges.shape[1],) or mask.dtype != torch.bool
            or (edges < 0).any() or (edges >= count).any()):
        raise ValueError('invalid single-anchor chemical graph or mapped core')
    expected = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    if {frozenset(pair) for pair in edges.T.tolist()} != expected or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('single-anchor graph must match the connected chemical molecule')
    adjacency = [set() for _ in range(count)]
    for a, b in edges.T.tolist():
        adjacency[a].add(b)
        adjacency[b].add(a)
    allowed = chemical_rotatable_pairs(mol)
    indices, bonds, moving, rejected, seen = [], [], [], [], set()
    for index, (u, v) in enumerate(edges[:, mask].T.tolist()):
        cut = frozenset((u, v))
        if cut in seen:
            raise ValueError('duplicate single-anchor torsion')
        seen.add(cut)
        if cut not in allowed:
            rejected.append((index, 'nonrotatable_chemical_bond'))
            continue
        component, pending = {v}, [v]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if frozenset((current, neighbor)) != cut and neighbor not in component:
                    component.add(neighbor)
                    pending.append(neighbor)
        if u in component:
            rejected.append((index, 'ring_closure'))
            continue
        if core & component and core - component:
            rejected.append((index, 'splits_protected_end'))
            continue
        if core <= component:
            u, v = v, u
            component = set(range(count)) - component
        indices.append(index)
        bonds.append((u, v))
        moving.append(tuple(sorted(component)))
    return AnchoredTorsionPlan(count, tuple(indices), tuple(bonds), tuple(moving),
                              (False,) * len(bonds), tuple(rejected))


class SingleAnchorTorsionFlow:
    """K torsions, six bounded known-side latents, six free POI parameters.

    POI center is relative to E3, divided by a declared length scale. Its global
    rotation vector is divided by pi. Paths in that chart are not SO(3) geodesics.
    """

    version = 'single_anchor_free_poi_chart_v1'

    def __init__(self, strict, anchor, e3_center, poi_scale, *, translation_bound=0., rotation_bound=0.):
        if any(strict.plan.moves_poi):
            raise ValueError('POI must not follow an invented second warhead')
        self.strict, self.plan, self.anchor = strict, strict.plan, tuple(anchor)
        self.center = torch.as_tensor(e3_center, dtype=strict.ligand.dtype, device=strict.ligand.device)
        self.scale = float(poi_scale)
        self.translation_bound, self.rotation_bound = float(translation_bound), float(rotation_bound)
        if (self.center.shape != (3,) or not torch.isfinite(self.center).all()
                or not math.isfinite(self.scale) or self.scale <= 0
                or not math.isfinite(self.translation_bound) or self.translation_bound < 0
                or not math.isfinite(self.rotation_bound) or not 0 <= self.rotation_bound < math.pi
                or len(set(self.anchor)) != len(self.anchor) or len(self.anchor) < 3
                or any(type(i) is not int or not 0 <= i < len(strict.ligand) for i in self.anchor)
                or any(set(self.anchor) & set(moving) for moving in strict.plan.moving_atoms)):
            raise ValueError('invalid single-anchor reference/bounds')
        proper_rigid_fit(strict.ligand[list(self.anchor)], strict.ligand[list(self.anchor)])
        self.torsion_count, self.size = len(self.plan.bonds), len(self.plan.bonds) + 12

    def _state(self, state):
        if (state.shape != (self.size,) or state.dtype != self.strict.ligand.dtype
                or state.device != self.strict.ligand.device or not torch.isfinite(state).all()):
            raise ValueError('single-anchor state shape/dtype/device mismatch')

    def consensus_velocity(self, normalized_torsion):
        return self.strict.consensus_velocity(normalized_torsion)

    def decode(self, state):
        self._state(state)
        k = self.torsion_count
        ligand = self.strict.decode(state[:k])[0].double()
        translation = ball_map(state[k:k + 3].double(), self.translation_bound)
        rotation = rotation_exp(ball_map(state[k + 3:k + 6].double(), self.rotation_bound))
        center = self.strict.ligand[list(self.anchor)].double().mean(0)
        ligand = (ligand - center) @ rotation.T + center + translation
        poi_rotation = rotation_exp(state[k + 9:].double() * math.pi)
        poi = self.strict.poi.double()
        poi = ((poi - poi.mean(0)) @ poi_rotation.T + self.center.double()
               + self.scale * state[k + 6:k + 9].double())
        return ligand.to(state), poi.to(state), (poi_rotation @ self.strict.body.double()).to(state)

    def sample_prior(self, *, generator):
        from scipy.spatial.transform import Rotation

        state = self.strict.ligand.new_zeros(self.size)
        k = self.torsion_count
        state[:k] = self.strict.sample_prior(generator=generator)
        for start, bound in ((k, self.translation_bound), (k + 3, self.rotation_bound)):
            if bound:
                state[start:start + 3] = torch.randn(3, device=state.device, dtype=state.dtype, generator=generator)
        state[k + 6:k + 9] = torch.randn(3, device=state.device, dtype=state.dtype, generator=generator)
        quaternion = torch.randn(4, device=state.device, dtype=state.dtype, generator=generator)
        state[k + 9:] = torch.as_tensor(Rotation.from_quat(quaternion.cpu().numpy()).as_rotvec(),
                                       dtype=state.dtype, device=state.device) / math.pi
        return state

    def path(self, target, prior, time):
        self._state(target)
        self._state(prior)
        if not math.isfinite(float(time)) or not 0 <= float(time) <= 1:
            raise ValueError('invalid single-anchor time')
        velocity = prior - target
        velocity = torch.cat((wrap_angles(velocity[:self.torsion_count]), velocity[self.torsion_count:]))
        state = target + float(time) * velocity
        state = torch.cat((wrap_angles(state[:self.torsion_count]), state[self.torsion_count:]))
        return state, self.decode(state), velocity

    def step(self, state, velocity, dt):
        self._state(state)
        self._state(velocity)
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('invalid single-anchor step')
        updated = state - float(dt) * velocity
        updated = torch.cat((wrap_angles(updated[:self.torsion_count]), updated[self.torsion_count:]))
        return updated, self.decode(updated)

    def cartesian_velocity(self, state, velocity, *, create_graph=False):
        self._state(state)
        self._state(velocity)
        return torch.autograd.functional.jvp(lambda q: self.decode(q)[:2], state, velocity,
                                              create_graph=create_graph)[1]

    @torch.no_grad()
    def encode_target(self, ligand, poi, edges, *, tolerance=1e-3):
        from scipy.spatial.transform import Rotation

        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('invalid single-anchor target tolerance')
        for actual, reference in ((ligand, self.strict.ligand), (poi, self.strict.poi)):
            if (actual.shape != reference.shape or actual.dtype != reference.dtype
                    or actual.device != reference.device or not torch.isfinite(actual).all()):
                raise ValueError('single-anchor target shape/dtype/device mismatch')
        reference = self.strict.ligand.double()
        re, te, _ = proper_rigid_fit(reference[list(self.anchor)], ligand[list(self.anchor)].double())
        unshifted = ((ligand.double() - te) @ re).to(ligand)
        angles, _ = self.strict.encode_target(unshifted, self.strict.poi, edges, tolerance=tolerance)
        center = reference[list(self.anchor)].mean(0)
        rotvec = lambda r: torch.as_tensor(Rotation.from_matrix(r.cpu().numpy()).as_rotvec(),
                                           device=ligand.device, dtype=torch.float64)
        rp, _, _ = proper_rigid_fit(self.strict.poi.double(), poi.double())
        state = torch.cat((angles, inverse_ball(re @ center + te - center, self.translation_bound).to(angles),
                           inverse_ball(rotvec(re), self.rotation_bound).to(angles),
                           ((poi.mean(0) - self.center) / self.scale), (rotvec(rp) / math.pi).to(angles)))
        decoded = self.decode(state)
        errors = {'ligand_max_error': float((decoded[0] - ligand).norm(dim=-1).max()),
                  'poi_max_error': float((decoded[1] - poi).norm(dim=-1).max())}
        if max(errors.values()) > tolerance:
            raise ValueError(f'target outside single-anchor chart: {errors}')
        return state, errors
