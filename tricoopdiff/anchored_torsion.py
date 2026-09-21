"""Experimental anchored torsions; not yet connected to the production flow.

Compile bridge cuts from the existing torsion list. Each protected end stays
intact, and the second receptor follows its end's rigid motion. This preserves
an initially supplied binary geometry; it does not repair an incompatible pose.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnchoredTorsionPlan:
    atom_count: int
    original_indices: tuple
    bonds: tuple
    moving_atoms: tuple
    moves_poi: tuple
    rejected: tuple


def chemical_rotatable_pairs(molecule):
    """Conservative single-bond DOFs; exclude linear and acyl-heteroatom bonds.

    The vendored torsion mask is a graph bridge mask, not a chemical bond-order
    filter. Keep its output indexing but never rotate double/triple bonds here.
    This does not constitute an energy or ring-conformer model.
    """
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    mol = Chem.Mol(molecule)
    pairs = {frozenset(match) for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)}
    resonance = Chem.MolFromSmarts('[C,S](=[O,S,N])-[N,O,S]')
    pairs -= {frozenset((match[0], match[2])) for match in mol.GetSubstructMatches(resonance)}
    return pairs


def compile_anchored_torsions(atom_count, edge_index, edge_mask, end1, end2, *, molecule=None):
    """Root bridge rotations away from end1; retain original torsion ordering.

    Ends are explicit mapped atom sets, not inferred linker/contact labels.
    Reversing a bond and its moving component preserves its internal-angle
    convention while removing the rigid motion of the anchored component.
    """
    edges = torch.as_tensor(edge_index).detach().cpu()
    selected = torch.as_tensor(edge_mask).detach().cpu()
    if (atom_count < 1 or edges.ndim != 2 or edges.shape[0] != 2
            or edges.dtype != torch.long or selected.dtype != torch.bool
            or selected.shape != (edges.shape[1],)
            or (edges < 0).any() or (edges >= atom_count).any()):
        raise ValueError('invalid ligand graph or torsion mask')
    ends = [set(values) for values in (end1, end2)]
    if (any(not values for values in ends) or ends[0] & ends[1]
            or any(type(i) is not int or not 0 <= i < atom_count
                   for values in ends for i in values)):
        raise ValueError('two nonempty disjoint mapped ends are required')
    adjacency = [set() for _ in range(atom_count)]
    for a, b in edges.T.tolist():
        if a == b:
            raise ValueError('self edges are not chemical bonds')
        adjacency[a].add(b)
        adjacency[b].add(a)

    def component(start, cut=None):
        seen, pending = {start}, [start]
        while pending:
            a = pending.pop()
            for b in adjacency[a]:
                if cut is not None and frozenset((a, b)) == cut:
                    continue
                if b not in seen:
                    seen.add(b)
                    pending.append(b)
        return seen

    if len(component(0)) != atom_count:
        raise ValueError('ligand must be connected')
    allowed = None
    if molecule is not None:
        from rdkit import Chem

        mol = Chem.Mol(molecule)
        expected_edges = {frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                          for bond in mol.GetBonds()}
        observed_edges = {frozenset(edge) for edge in edges.T.tolist()}
        if mol.GetNumAtoms() != atom_count or observed_edges != expected_edges:
            raise ValueError('torsion graph differs from the supplied chemical molecule')
        allowed = chemical_rotatable_pairs(mol)
    indices, bonds, masks, follows, rejected = [], [], [], [], []
    used = set()
    for index, (u, v) in enumerate(edges[:, selected].T.tolist()):
        cut = frozenset((u, v))
        if cut in used:
            raise ValueError('duplicate directed torsion for one chemical bond')
        used.add(cut)
        if allowed is not None and cut not in allowed:
            rejected.append((index, 'nonrotatable_chemical_bond'))
            continue
        moving = component(v, cut)
        if u in moving:
            rejected.append((index, 'ring_closure'))
            continue
        if any(values & moving and values - moving for values in ends):
            rejected.append((index, 'splits_protected_end'))
            continue
        if ends[0] <= moving:
            u, v = v, u
            moving = set(range(atom_count)) - moving
        indices.append(index)
        bonds.append((u, v))
        masks.append(tuple(sorted(moving)))
        follows.append(ends[1] <= moving)
    return AnchoredTorsionPlan(atom_count, tuple(indices), tuple(bonds), tuple(masks),
                              tuple(follows), tuple(rejected))


def anchored_torsion_step(ligand, poi, body, increments, plan):
    """Apply signed angle increments; caller handles forward/reverse time.

    Uses the vendored v->u axis convention. Coordinates and angles remain in
    torch for differentiation. Body maps the original POI frame to global.
    """
    if (ligand.shape != (plan.atom_count, 3) or poi.ndim != 2
            or poi.shape[1] != 3 or not len(poi) or body.shape != (3, 3)
            or increments.shape != (len(plan.bonds),)):
        raise ValueError('anchored state/angle shape mismatch')
    if any(not torch.isfinite(x).all() for x in (ligand, poi, body, increments)):
        raise ValueError('anchored state must be finite')
    if any(x.device != ligand.device or x.dtype != ligand.dtype
           for x in (poi, body, increments)):
        raise ValueError('anchored tensors must share dtype and device')
    output_dtype = ligand.dtype
    if output_dtype not in (torch.float32, torch.float64):
        raise ValueError('anchored geometry requires float32 or float64 inputs')
    # Long lever arms magnify float32 drift over many successive rotations.
    # Keep this small geometry kernel in float64, not the neural network.
    ligand, poi, body, increments = (value.double() for value in (ligand, poi, body, increments))
    for (u, v), moving, follows, angle in zip(
            plan.bonds, plan.moving_atoms, plan.moves_poi, increments):
        origin = ligand[v]
        axis = ligand[u] - origin
        length = axis.norm()
        if length < 1e-8:
            raise ValueError('degenerate torsion axis')
        x, y, z = axis / length
        zero = x * 0
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        rotation = torch.linalg.matrix_exp(angle * skew)
        mask = torch.zeros(plan.atom_count, dtype=torch.bool, device=ligand.device)
        mask[list(moving)] = True
        ligand = torch.where(mask[:, None], (ligand - origin) @ rotation.T + origin, ligand)
        if follows:
            poi = (poi - origin) @ rotation.T + origin
            body = rotation @ body
    return tuple(value.to(output_dtype) for value in (ligand, poi, body))
