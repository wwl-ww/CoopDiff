"""Experimental conditional torsion chart, shared by path and integration.

The reference must already satisfy the supplied binary geometry. Training may
use a native reference; deployment must use an independently assembled one.
No endpoint fitting or production checkpoint compatibility is implied here.
"""

import math

import torch

from .anchored_torsion import anchored_torsion_step, compile_anchored_torsions


FLOW_VERSION = 'anchored_torsion_chart_v2_fp64_geometry'


def audit_bond_geometry(molecule, ligand, *, tolerance=.15):
    """Reject distorted covalent bonds before freezing their lengths in a chart.

    RDKit's topological distance-geometry bounds, enlarged by the explicit
    Angstrom tolerance, are a sanity gate, not an energy/stereochemistry check.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    mol = Chem.Mol(molecule)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('bond tolerance must be finite and nonnegative')
    if ligand.shape != (mol.GetNumAtoms(), 3) or not torch.isfinite(ligand).all():
        raise ValueError('bond audit coordinates must match the chemical graph')
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    violations = []
    maximum = 0.
    for bond in mol.GetBonds():
        a, b = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        length = float((ligand[a] - ligand[b]).norm().detach())
        lower, upper = float(bounds[b, a]), float(bounds[a, b])
        excess = max(lower - length, length - upper, 0.)
        maximum = max(maximum, excess)
        if excess > tolerance:
            violations.append({'atoms': [a, b], 'length': length,
                               'lower': lower, 'upper': upper})
    if violations:
        raise ValueError(f'binary assembly has distorted covalent bonds: {violations}')
    return {'bond_bounds': 'RDKit distance geometry', 'tolerance_angstrom': tolerance,
            'max_excess_angstrom': maximum, 'bonds_checked': mol.GetNumBonds()}


def wrap_angles(value):
    return torch.remainder(value + math.pi, 2 * math.pi) - math.pi


class AnchoredTorsionFlow:
    """Time runs from data=0 to prior=1; angles are relative to one reference."""

    version = FLOW_VERSION

    def __init__(self, ligand, poi, body, plan):
        self.ligand = ligand.detach().clone()
        self.poi = poi.detach().clone()
        self.body = body.detach().clone()
        self.plan = plan
        # Validate the geometry even for a plan with no movable bonds.
        self.decode(self.ligand.new_zeros(len(plan.bonds)))

    @classmethod
    def from_binary_initializer(cls, initializer, e3, poi, ligand_data, *, seed,
                                max_fit_rmsd, bond_tolerance=.15):
        """Build a deployment chart from the existing native-free binary assembly.

        The caller declares acceptable fit error. Preserving an approximate
        assembly cannot strengthen it into an exact experimental constraint.
        The selected legal map stays fixed for this chart's entire trajectory.
        """
        from .binary_pose_constraints import BinaryPoseInitializer

        if not isinstance(initializer, BinaryPoseInitializer) or initializer.no_torsion:
            raise ValueError('anchored chart requires the torsion-enabled binary initializer')
        if not math.isfinite(max_fit_rmsd) or max_fit_rmsd <= 0:
            raise ValueError('declare a finite positive binary-fit tolerance')
        def validate_candidate(candidate, maps):
            if candidate['e3_frame_rmsd'] > 1e-3:
                raise ValueError('binary side1 protein coordinates differ from current E3 input')
            fits = [candidate['side1_fit_rmsd'], candidate['side2_fit_rmsd']]
            if max(fits) > max_fit_rmsd:
                raise ValueError(f'binary assembly exceeds declared fit tolerance: {fits}')
            bonds = audit_bond_geometry(ligand_data['mol'], candidate['ligand_coords'],
                                       tolerance=bond_tolerance)
            compile_anchored_torsions(candidate['ligand_coords'].shape[0],
                                     ligand_data['edge_index'], ligand_data['edge_mask'],
                                     *maps, molecule=ligand_data['mol'])
            return {'policy': 'anchored_validity_before_clash_selection_v1',
                    'bond_geometry': bonds, 'max_fit_rmsd': max_fit_rmsd}

        ligand, receptor, body, audit = initializer.initialize(
            e3, poi, ligand_data, seed=seed, candidate_validator=validate_candidate)
        if audit['e3_frame_rmsd'] > 1e-3:
            raise ValueError('binary side1 protein coordinates differ from current E3 input')
        if max(audit['side_fit_rmsd']) > max_fit_rmsd:
            raise ValueError(f'binary assembly exceeds declared fit tolerance: {audit["side_fit_rmsd"]}')
        bond_audit = audit_bond_geometry(ligand_data['mol'], ligand[0], tolerance=bond_tolerance)
        left, right = audit['selected_atom_maps']
        plan = compile_anchored_torsions(ligand.shape[1], ligand_data['edge_index'],
                                        ligand_data['edge_mask'], left, right,
                                        molecule=ligand_data['mol'])
        chart = cls(ligand[0], receptor[0], body[0], plan)
        audit = {**audit, 'generative_process': chart.version,
                 'bond_geometry': bond_audit,
                 'max_fit_rmsd': max_fit_rmsd,
                 'allowed_torsion_indices': list(plan.original_indices),
                 'rejected_torsions': list(plan.rejected)}
        return chart, audit

    def _angles(self, value):
        if (value.shape != (len(self.plan.bonds),) or not torch.isfinite(value).all()
                or value.dtype != self.ligand.dtype or value.device != self.ligand.device):
            raise ValueError('angles must match the anchored chart shape/device/dtype')

    def decode(self, angles):
        self._angles(angles)
        return anchored_torsion_step(self.ligand, self.poi, self.body, angles, self.plan)

    def sample_prior(self, *, generator):
        """Uniform allowed torsions; an explicit generator owns randomness."""
        return (torch.rand(len(self.plan.bonds), dtype=self.ligand.dtype,
                           device=self.ligand.device, generator=generator) * 2 - 1) * math.pi

    def encode_target(self, ligand, poi, edge_index, *, tolerance=1e-3):
        """Recover torsions and reject a target outside this fixed-geometry chart.

        Targets must already use the anchored frame. No best-fit global alignment
        is allowed to hide an inconsistent anchor. This is a training/data audit,
        not an inference operation requiring native coordinates.
        """
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('target tolerance must be finite and positive')
        for value, reference in ((ligand, self.ligand), (poi, self.poi)):
            if (value.shape != reference.shape or value.device != reference.device
                    or value.dtype != reference.dtype or not torch.isfinite(value).all()):
                raise ValueError('target must match the reference shape/device/dtype')
        edges = torch.as_tensor(edge_index).detach().cpu()
        if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
                or (edges < 0).any() or (edges >= self.plan.atom_count).any()):
            raise ValueError('invalid target chemical edges')
        neighbors = [set() for _ in range(self.plan.atom_count)]
        for u, v in edges.T.tolist():
            neighbors[u].add(v)
            neighbors[v].add(u)

        def dihedral(coords, a, u, v, b):
            coords = coords.double()
            axis = coords[u] - coords[v]
            if axis.norm() < 1e-8:
                raise ValueError('degenerate target bond axis')
            axis = axis / axis.norm()
            left, right = coords[a] - coords[u], coords[b] - coords[v]
            left = left - left.dot(axis) * axis
            right = right - right.dot(axis) * axis
            if min(left.norm(), right.norm()) < 1e-8:
                raise ValueError('collinear torsion reference atoms')
            return torch.atan2(torch.cross(left, right, dim=-1).dot(axis), left.dot(right))

        angles = []
        for (u, v), moving in zip(self.plan.bonds, self.plan.moving_atoms):
            left = sorted(neighbors[u] - {v} - set(moving))
            right = sorted((neighbors[v] - {u}) & set(moving))
            if not left or not right:
                raise ValueError('torsion requires a bonded reference atom on each side')
            a, b = left[0], right[0]
            angles.append(wrap_angles(dihedral(ligand, a, u, v, b)
                                      - dihedral(self.ligand, a, u, v, b)))
        angles = torch.stack(angles).to(self.ligand) if angles else self.ligand.new_zeros(0)
        decoded_ligand, decoded_poi, _ = self.decode(angles)
        errors = {
            'ligand_max_error': float((decoded_ligand - ligand).norm(dim=-1).max().detach()),
            'poi_max_error': float((decoded_poi - poi).norm(dim=-1).max().detach()),
        }
        if max(errors.values()) > tolerance:
            raise ValueError(f'target outside anchored chart: {errors}; tolerance={tolerance}')
        return angles, errors

    def path(self, data_angles, prior_angles, time):
        self._angles(data_angles)
        self._angles(prior_angles)
        t = torch.as_tensor(time, dtype=self.ligand.dtype, device=self.ligand.device)
        if t.numel() != 1 or not torch.isfinite(t).all() or not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError('flow time must be a finite scalar in [0,1]')
        target = wrap_angles(prior_angles - data_angles)
        angles = wrap_angles(data_angles + t.reshape(()) * target)
        return angles, self.decode(angles), target

    def step(self, angles, velocity, dt):
        self._angles(angles)
        self._angles(velocity)
        step = torch.as_tensor(dt, dtype=angles.dtype, device=angles.device)
        if step.numel() != 1 or not torch.isfinite(step).all() or not bool((step > 0).all()):
            raise ValueError('reverse step must be finite and positive')
        updated = wrap_angles(angles - step.reshape(()) * velocity)
        return updated, self.decode(updated)

    def cartesian_velocity(self, angles, velocity, *, create_graph=False):
        """Return J_D(q) v for ligand and POI, including all coupled motion."""
        self._angles(angles)
        self._angles(velocity)
        if not angles.numel():
            return torch.zeros_like(self.ligand), torch.zeros_like(self.poi)

        def coordinates(value):
            ligand, poi, _ = self.decode(value)
            return ligand, poi

        _, tangent = torch.autograd.functional.jvp(
            coordinates, angles, velocity, create_graph=create_graph)
        return tangent

    def consensus_velocity(self, normalized_torsion):
        """Select existing per-bond outputs; preserve SDPI's pi normalization.

        This adapter does not make an old checkpoint trained for the new flow.
        Rigid heads are not integrated in the complete two-end chart.
        """
        expected = len(self.plan.original_indices) + len(self.plan.rejected)
        if (normalized_torsion.shape != (1, expected)
                or not torch.isfinite(normalized_torsion).all()):
            raise ValueError('consensus torsion output does not match original bond list')
        return normalized_torsion[0, list(self.plan.original_indices)] * math.pi
