"""Two-warhead anchor geometry for PROTAC-like ternary assemblies.

The training splitter uses a deposited ternary pose only to create a
*synthetic input condition*. Validation/inference must instead use external
binary complexes; the target ternary pose must never enter that path.
"""

import hashlib
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS
from scipy.spatial import cKDTree

from tricoopdiff.geometry.rigid import apply_rigid, proper_rigid_fit


def atom_order_graph_signature(molecule):
    """Hash ordered heavy-atom chemistry, including the ligand index contract.

    Canonical SMILES is insufficient here: it identifies an isomeric graph but
    intentionally forgets the atom ordering used by the coordinate and rotor
    tensors. Binary-anchor atom indices must refer to exactly the same order.
    """
    if isinstance(molecule, bytes):
        molecule = Chem.Mol(molecule)
    if molecule is None:
        raise ValueError('a ligand molecule is required for anchor graph identity')
    heavy = Chem.RemoveHs(Chem.Mol(molecule))
    heavy.RemoveAllConformers()
    atoms = [(atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
              bool(atom.GetIsAromatic())) for atom in heavy.GetAtoms()]
    bonds = sorted((min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    str(bond.GetBondType()), bool(bond.GetIsAromatic()))
                   for bond in heavy.GetBonds())
    payload = json.dumps({'atoms': atoms, 'bonds': bonds}, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_anchor_graph_identity(condition, molecule):
    """Reject anchor indices generated for a different ligand atom ordering."""
    expected = condition.get('atom_order_graph_sha256') if isinstance(condition, dict) else None
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('binary-anchor condition is missing its ordered ligand graph identity')
    actual = atom_order_graph_signature(molecule)
    if actual != expected:
        raise ValueError('binary-anchor atom indices do not match ligand coordinate/rotor ordering')
    return actual


def _fragment_after_cut(molecule, left, right, start):
    """Return the connected component containing ``start`` after one bond cut."""
    seen = {start}
    pending = [start]
    while pending:
        atom = pending.pop()
        for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors():
            other = neighbor.GetIdx()
            if {atom, other} == {left, right} or other in seen:
                continue
            seen.add(other)
            pending.append(other)
    return frozenset(seen)


def _noncollinear(coordinates):
    if len(coordinates) < 3:
        return False
    centered = coordinates - coordinates.mean(0)
    return np.linalg.svd(centered, compute_uv=False)[1] > .5


def split_training_warheads(molecule, ligand_xyz, protein1_xyz, protein2_xyz,
                            *, contact_cutoff=4.5, opposite_clearance=6.0,
                            min_anchor_atoms=8, max_linker_fraction=.5):
    """Identify two disjoint bound cores separated by a flexible linker.

    The returned cores are training-only pseudo-binary conditions. Selection
    uses the native ligand/protein contact geometry, so callers must never run
    this function during validation or inference. No result is returned for
    molecular glues or ambiguous ligands without two separable contact cores.
    """
    if (isinstance(min_anchor_atoms, bool) or not isinstance(min_anchor_atoms, int)
            or min_anchor_atoms < 5 or not np.isfinite(max_linker_fraction)
            or not 0 < max_linker_fraction < 1):
        raise ValueError('invalid PROTAC anchor/linker size constraints')
    ligand_xyz = np.asarray(ligand_xyz, dtype=float)
    protein1_xyz = np.asarray(protein1_xyz, dtype=float)
    protein2_xyz = np.asarray(protein2_xyz, dtype=float)
    if (molecule is None or len(Chem.GetMolFrags(molecule)) != 1
            or ligand_xyz.shape != (molecule.GetNumAtoms(), 3)
            or any(value.ndim != 2 or value.shape[1] != 3 or not len(value)
                   for value in (protein1_xyz, protein2_xyz, ligand_xyz))
            or not all(np.isfinite(value).all()
                       for value in (ligand_xyz, protein1_xyz, protein2_xyz))):
        raise ValueError('anchor splitter requires one connected, finite complex')
    distance1 = cKDTree(protein1_xyz).query(ligand_xyz)[0]
    distance2 = cKDTree(protein2_xyz).query(ligand_xyz)[0]
    exclusive1 = frozenset(np.where((distance1 <= contact_cutoff)
                                    & (distance2 >= opposite_clearance))[0].tolist())
    exclusive2 = frozenset(np.where((distance2 <= contact_cutoff)
                                    & (distance1 >= opposite_clearance))[0].tolist())
    if len(exclusive1) < 3 or len(exclusive2) < 3:
        return None

    cuts = []
    for bond in molecule.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        side = _fragment_after_cut(molecule, left, right, left)
        if len(side) == molecule.GetNumAtoms():
            continue
        other = frozenset(range(molecule.GetNumAtoms()))-side
        cuts.extend(((side, (left, right)), (other, (right, left))))

    options1 = [(core, edge) for core, edge in cuts
                if len(core) >= min_anchor_atoms and len(core & exclusive1) >= 3
                and len(core & exclusive2) == 0
                and _noncollinear(ligand_xyz[list(core)])]
    options2 = [(core, edge) for core, edge in cuts
                if len(core) >= min_anchor_atoms and len(core & exclusive2) >= 3
                and len(core & exclusive1) == 0
                and _noncollinear(ligand_xyz[list(core)])]
    ring_atoms = set().union(*map(set, molecule.GetRingInfo().AtomRings())) \
        if molecule.GetRingInfo().NumRings() else set()
    candidates = []
    for core1, edge1 in options1:
        for core2, edge2 in options2:
            if core1 & core2:
                continue
            linker = frozenset(range(molecule.GetNumAtoms()))-core1-core2
            if len(linker) < 2:
                continue
            bridge = _bridge_attachment(molecule, core1, core2)
            if (bridge is None or bridge[2] != linker
                    or bridge[0] != edge1[0] or bridge[1] != edge2[0]):
                continue
            if len(linker) / molecule.GetNumAtoms() > max_linker_fraction:
                continue
            coverage = len(core1 & exclusive1)+len(core2 & exclusive2)
            ring_coverage = len((core1 | core2) & ring_atoms)
            internal_rotors = sum(
                bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
                and bond.GetBeginAtomIdx() in core and bond.GetEndAtomIdx() in core
                for bond in molecule.GetBonds() for core in (core1, core2))
            # First cover the two exclusive interfaces, then retain whole
            # cyclic warhead chemistry while avoiding flexible linker capture.
            score = (coverage, ring_coverage, -internal_rotors, -len(linker))
            candidates.append((score, core1, core2, linker, edge1, edge2))
    if not candidates:
        return None
    _, core1, core2, linker, edge1, edge2 = max(candidates,
                                                 key=lambda row: row[0])
    return {'anchor1_atoms': tuple(sorted(core1)),
            'anchor2_atoms': tuple(sorted(core2)),
            'linker_atoms': tuple(sorted(linker)),
            'anchor1_exit_bond': edge1, 'anchor2_exit_bond': edge2,
            'exclusive_contact_counts': (len(exclusive1), len(exclusive2)),
            'contract': 'synthetic_binary_anchor_split_v1',
            'atom_order_graph_sha256': atom_order_graph_signature(molecule),
            'native_coordinates_used_to_construct_training_condition': True}


def make_training_binary_condition(split, native_ligand, native_protein2, *,
                                   generator=None):
    """Make independent binary frames from a training-only ternary example.

    The second protein and its bound warhead receive the *same* random rigid
    transform. Their local binding geometry survives, while the relative
    protein pose in the deposited ternary structure is erased. This function
    never supplies a full-ligand native conformer to the initializer.
    """
    import torch

    ligand = torch.as_tensor(native_ligand)
    protein2 = torch.as_tensor(native_protein2)
    if (ligand.ndim != 2 or ligand.shape[1] != 3 or
            protein2.ndim != 2 or protein2.shape[1] != 3 or
            not ligand.is_floating_point() or not protein2.is_floating_point() or
            not torch.isfinite(ligand).all() or not torch.isfinite(protein2).all()):
        raise ValueError('training binary condition needs finite [N,3] coordinates')
    first, second = split['anchor1_atoms'], split['anchor2_atoms']
    if (split.get('contract') != 'synthetic_binary_anchor_split_v1' or
            set(first) & set(second) or min(len(first), len(second)) < 3 or
            max((*first, *second)) >= len(ligand)):
        raise ValueError('training binary split is invalid')
    random = torch.randn(4, device=ligand.device, dtype=ligand.dtype,
                         generator=generator)
    quaternion = random / random.norm().clamp_min(1e-12)
    w, x, y, z = quaternion.unbind()
    rotation = torch.stack((
        torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
        torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
        torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))),
    ))
    translation = torch.randn(3, device=ligand.device, dtype=ligand.dtype,
                              generator=generator) * 10
    center = protein2.mean(0)
    local_protein2 = (protein2-center) @ rotation.T + translation
    local_anchor2 = (ligand[list(second)]-center) @ rotation.T + translation
    return {'anchor1_atoms': first, 'anchor2_atoms': second,
            'anchor1_xyz': ligand[list(first)].detach().clone(),
            'anchor2_xyz': local_anchor2.detach(),
            'protein2_reference': local_protein2.detach(),
            'contract': 'synthetic_independent_binary_frames_v1',
            'atom_order_graph_sha256': split['atom_order_graph_sha256'],
            'native_full_ligand_conformer_is_input': False}


def assemble_from_binary_anchors(ligand_xyz, protein2_xyz, anchor1_xyz,
                                 anchor2_xyz, anchor1_atoms, anchor2_atoms):
    """Place the full ligand and second protein by their two bound warheads.

    ``anchor1_xyz`` is in the fixed protein-1 frame; ``anchor2_xyz`` and
    ``protein2_xyz`` share a protein-2 local frame. The full ligand conformer
    supplies the relative warhead orientation through its linker. This
    differentiable kinematic map replaces an unconstrained protein-2 pose.
    Atom maps are chemical input conditions and must not be fitted to native
    ternary coordinates during inference.
    """
    import torch

    if not all(torch.is_tensor(value) and value.is_floating_point()
               for value in (ligand_xyz, protein2_xyz, anchor1_xyz,
                             anchor2_xyz)):
        raise ValueError('anchor assembly requires floating tensors')
    if (ligand_xyz.ndim != 2 or ligand_xyz.shape[1] != 3
            or protein2_xyz.ndim != 2 or protein2_xyz.shape[1] != 3
            or anchor1_xyz.ndim != 2 or anchor1_xyz.shape[1] != 3
            or anchor2_xyz.ndim != 2 or anchor2_xyz.shape[1] != 3):
        raise ValueError('anchor assembly coordinates must be [N,3]')
    first = torch.as_tensor(anchor1_atoms, dtype=torch.long,
                            device=ligand_xyz.device)
    second = torch.as_tensor(anchor2_atoms, dtype=torch.long,
                             device=ligand_xyz.device)
    if (len(first) != len(anchor1_xyz) or len(second) != len(anchor2_xyz)
            or min(len(first), len(second)) < 3
            or first.min() < 0 or second.min() < 0
            or first.max() >= len(ligand_xyz) or second.max() >= len(ligand_xyz)
            or len(set(first.tolist())) != len(first)
            or len(set(second.tolist())) != len(second)
            or set(first.tolist()) & set(second.tolist())):
        raise ValueError('two disjoint mapped warheads with at least three atoms are required')
    if (not all(torch.isfinite(value).all()
                for value in (ligand_xyz, protein2_xyz, anchor1_xyz, anchor2_xyz))
            or torch.linalg.svdvals(anchor1_xyz-anchor1_xyz.mean(0))[1] <= .5
            or torch.linalg.svdvals(anchor2_xyz-anchor2_xyz.mean(0))[1] <= .5):
        raise ValueError('warhead frames must be finite and noncollinear')
    rotation1, shift1, error1 = proper_rigid_fit(ligand_xyz[first], anchor1_xyz)
    ligand_pose = apply_rigid(ligand_xyz, rotation1, shift1)
    rotation2, shift2, error2 = proper_rigid_fit(anchor2_xyz,
                                                ligand_pose[second])
    protein2_pose = apply_rigid(protein2_xyz, rotation2, shift2)
    return {'ligand': ligand_pose, 'protein2': protein2_pose,
            'protein2_rotation': rotation2, 'protein2_translation': shift2,
            'anchor_rmsd': torch.stack((error1, error2))}


def assemble_binary_mapping_hypotheses(ligand_xyz, protein2_xyz, mapping):
    """Assemble graph-symmetric mappings without oracle selection.

    Results are ordered by rigid-fit residual, measuring whether this full
    ligand conformer can satisfy the observed binary warheads while preserving
    their internal geometry. This is a conformer feasibility score, not DockQ.
    """
    import torch

    hypotheses = mapping.get('mapping_hypotheses')
    if not hypotheses:
        raise ValueError('binary mapping has no explicit mapping hypotheses')
    outputs = []
    for index, hypothesis in enumerate(hypotheses):
        result = assemble_from_binary_anchors(
            ligand_xyz, protein2_xyz,
            torch.as_tensor(hypothesis['anchor1_xyz'], device=ligand_xyz.device,
                            dtype=ligand_xyz.dtype),
            torch.as_tensor(hypothesis['anchor2_xyz'], device=ligand_xyz.device,
                            dtype=ligand_xyz.dtype),
            hypothesis['anchor1_atoms'], hypothesis['anchor2_atoms'])
        result['mapping_index'] = index
        result['anchor_fit_rmsd'] = result['anchor_rmsd'].square().mean().sqrt()
        outputs.append(result)
    return sorted(outputs, key=lambda result: float(result['anchor_fit_rmsd'].detach()))


def close_linker_to_binary_anchors(molecule, anchor1_atoms, anchor1_xyz,
                                   anchor2_atoms, anchor2_xyz, *, seed=73,
                                   starts=8, max_iterations=1000):
    """Fit PROTAC linker conformers to two anchor frames in one protein pose.

    The protein-pair pose is an input hypothesis (for example an HDOCK pose).
    Each binary warhead is first expressed in that common frame. The solver
    then fixes both observed warheads in place and minimizes the linker under
    the ligand force field. Thus a candidate only survives if its *actual
    chemical linker* can connect the two binary poses; random linker
    conformations no longer move a protein to manufacture a plausible-looking
    protein pose.

    Returns candidates ordered by force-field strain above the unrestrained
    conformer energy. This is a physical feasibility score, not DockQ.
    """
    if (isinstance(starts, bool) or not isinstance(starts, int) or starts < 1
            or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError('invalid linker closure optimization settings')
    ligand = Chem.RemoveHs(Chem.Mol(molecule))
    ligand.RemoveAllConformers()
    if len(Chem.GetMolFrags(ligand)) != 1:
        raise ValueError('linker closure requires one connected ligand graph')
    first = np.asarray(anchor1_atoms, dtype=np.int64)
    second = np.asarray(anchor2_atoms, dtype=np.int64)
    target1 = np.asarray(anchor1_xyz, dtype=np.float64)
    target2 = np.asarray(anchor2_xyz, dtype=np.float64)
    atom_count = ligand.GetNumAtoms()
    if (target1.shape != (len(first), 3) or target2.shape != (len(second), 3)
            or min(len(first), len(second)) < 3
            or len(set(first.tolist())) != len(first)
            or len(set(second.tolist())) != len(second)
            or set(first.tolist()) & set(second.tolist())
            or min(first.min(), second.min()) < 0
            or max(first.max(), second.max()) >= atom_count
            or not np.isfinite(target1).all() or not np.isfinite(target2).all()):
        raise ValueError('linker closure requires finite, disjoint mapped anchor coordinates')
    ligand = Chem.AddHs(ligand)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed) % (2**31-1) or 1
    parameters.numThreads = 1
    conformer_ids = list(AllChem.EmbedMultipleConfs(ligand, numConfs=starts,
                                                    params=parameters))
    embedding_mode = 'ETKDGv3'
    if not conformer_ids:
        ligand.RemoveAllConformers()
        parameters.useRandomCoords = True
        parameters.maxIterations = 1000
        conformer_ids = list(AllChem.EmbedMultipleConfs(
            ligand, numConfs=starts, params=parameters))
        embedding_mode = 'ETKDGv3_random_coordinates_fallback'
    if not conformer_ids:
        raise ValueError(f'ETKDG produced no linker starts (requested {starts})')
    properties = AllChem.MMFFGetMoleculeProperties(ligand)
    if properties is None:
        raise ValueError('MMFF parameters unavailable for PROTAC graph')

    def rigid_fit(source, target):
        source_center, target_center = source.mean(0), target.mean(0)
        u, _, vt = np.linalg.svd((source-source_center).T @ (target-target_center))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = u @ vt
        translation = target_center-source_center @ rotation
        return rotation, translation

    outputs = []
    for conformer_id in conformer_ids:
        conformer = ligand.GetConformer(conformer_id)
        coordinates = np.asarray(conformer.GetPositions(), dtype=np.float64)
        free_ff = AllChem.MMFFGetMoleculeForceField(ligand, properties,
                                                    confId=conformer_id)
        if free_ff is None:
            continue
        free_ff.Minimize(maxIts=max_iterations)
        free_energy = float(free_ff.CalcEnergy())
        coordinates = np.asarray(ligand.GetConformer(conformer_id).GetPositions())
        rotation, translation = rigid_fit(coordinates[first], target1)
        coordinates = coordinates @ rotation+translation
        for atom, point in enumerate(coordinates):
            conformer.SetAtomPosition(atom, point.tolist())
        for atoms, target in ((first, target1), (second, target2)):
            for atom, point in zip(atoms, target):
                conformer.SetAtomPosition(int(atom), point.tolist())
        ff = AllChem.MMFFGetMoleculeForceField(ligand, properties,
                                              confId=conformer_id)
        if ff is None:
            continue
        for atom in np.concatenate((first, second)):
            ff.AddFixedPoint(int(atom))
        status = ff.Minimize(maxIts=max_iterations)
        minimized_energy = float(ff.CalcEnergy())
        result_xyz = np.asarray(ligand.GetConformer(conformer_id).GetPositions())
        first_error = float(np.sqrt(np.mean(np.sum(
            (result_xyz[first]-target1)**2, axis=1))))
        second_error = float(np.sqrt(np.mean(np.sum(
            (result_xyz[second]-target2)**2, axis=1))))
        # A candidate must carry only its own conformer.  Downstream geometry
        # refinement reads GetConformer() without a conformer ID; retaining
        # every embedded start silently feeds it conformer 0 instead.
        selected = Chem.Mol(ligand)
        for other in list(selected.GetConformers()):
            if other.GetId() != conformer_id:
                selected.RemoveConformer(other.GetId())
        outputs.append({'ligand': Chem.RemoveHs(selected),
            'coordinates': result_xyz[:atom_count].copy(),
            'anchor1_rmsd': first_error, 'anchor2_rmsd': second_error,
            'unrestrained_energy': free_energy, 'closed_energy': minimized_energy,
            'strain_energy': minimized_energy-free_energy,
            'optimizer_status': int(status), 'conformer_id': int(conformer_id),
            'embedding_mode': embedding_mode})
    if not outputs:
        raise ValueError('no valid MMFF linker-closure optimization was produced')
    return sorted(outputs, key=lambda row: row['strain_energy'])


def select_binary_mapping_hypothesis(mapping, index):
    """Select one explicit symmetry mapping for a reproducible walker."""
    hypotheses = mapping.get('mapping_hypotheses')
    if not hypotheses or isinstance(index, bool) or not isinstance(index, int):
        raise ValueError('an explicit integer binary-mapping hypothesis is required')
    if not 0 <= index < len(hypotheses):
        raise ValueError('binary-mapping hypothesis index is out of range')
    return {**mapping, **hypotheses[index],
            'mapping_hypothesis_index': index,
            'selected_mapping_policy': 'explicit_walker_hypothesis_v1'}


def _bridge_attachment(molecule, core1, core2):
    """Find the unique non-core graph component that connects both warheads."""
    first, second = set(core1), set(core2)
    remaining = set(range(molecule.GetNumAtoms()))-first-second
    bridges = []
    while remaining:
        start = remaining.pop()
        component = {start}
        pending = [start]
        while pending:
            for neighbor in molecule.GetAtomWithIdx(pending.pop()).GetNeighbors():
                index = neighbor.GetIdx()
                if index in remaining:
                    remaining.remove(index)
                    component.add(index)
                    pending.append(index)
        touching1 = {neighbor.GetIdx() for index in component
                     for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
                     if neighbor.GetIdx() in first}
        touching2 = {neighbor.GetIdx() for index in component
                     for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
                     if neighbor.GetIdx() in second}
        if touching1 and touching2:
            bridges.append((touching1, touching2, component))
    # Binary reference ligands can omit peripheral warhead substituents.  Those
    # disconnected non-core components are part of the full molecule, but not
    # part of the unique component joining the two observed binding cores.
    if (len(bridges) != 1 or len(bridges[0][0]) != 1
            or len(bridges[0][1]) != 1):
        return None
    return (next(iter(bridges[0][0])), next(iter(bridges[0][1])),
            frozenset(bridges[0][2]))


def match_external_binary_anchors(full_molecule, fragment1, fragment2, *,
                                  min_coverage=.75, timeout=15,
                                  hypothesis_limit=None, geometry_limit=None,
                                  geometry_fragment_atoms=None):
    """Map external binary ligands onto a PROTAC graph without ternary poses.

    This is a conservative exact-bond common-core contract, not a claim that
    every binary reference compound is the same warhead. Ambiguous attachment
    sites, insufficient coverage, timed-out chemistry, or overlapping cores
    are rejected instead of silently using a native-coordinate tie-breaker.
    """
    if (not 0 < min_coverage <= 1 or timeout < 1
            or (hypothesis_limit is not None and hypothesis_limit < 1)
            or (geometry_limit is not None and geometry_limit < 1)
            or (hypothesis_limit is not None and geometry_limit is not None)
            or (geometry_fragment_atoms is not None
                and geometry_limit is None)):
        raise ValueError('invalid binary matching threshold or timeout')
    full = Chem.RemoveHs(Chem.Mol(full_molecule))
    full.RemoveAllConformers()  # A deposited PROTAC conformer is never an input.
    fragments = [Chem.RemoveHs(Chem.Mol(fragment))
                 for fragment in (fragment1, fragment2)]
    if any(fragment.GetNumConformers() != 1 for fragment in fragments):
        raise ValueError('each external binary ligand needs one observed pose')
    if geometry_fragment_atoms is not None:
        if (len(geometry_fragment_atoms) != 2 or any(
                len(set(atoms)) < 3 or min(atoms) < 0
                or max(atoms) >= fragment.GetNumAtoms()
                for atoms, fragment in zip(geometry_fragment_atoms, fragments))):
            raise ValueError('invalid fragment atoms for geometric deduplication')
        geometry_fragment_atoms = tuple(map(frozenset,
                                            geometry_fragment_atoms))
    if len(Chem.GetMolFrags(full)) != 1:
        raise ValueError('the indexed PROTAC must be one connected molecule')

    options, coverages = [], []
    for fragment in fragments:
        result = rdFMCS.FindMCS([full, fragment], timeout=timeout,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True, completeRingsOnly=True)
        coverage = result.numAtoms / fragment.GetNumAtoms()
        if result.canceled or coverage < min_coverage or not result.smartsString:
            return None
        query = Chem.MolFromSmarts(result.smartsString)
        full_matches = full.GetSubstructMatches(query, uniquify=False,
                                                maxMatches=1000)
        fragment_matches = fragment.GetSubstructMatches(query, uniquify=False,
                                                        maxMatches=1000)
        if not full_matches or not fragment_matches:
            return None
        # Keep the two match axes separate. Their Cartesian product can have
        # millions of permutations for a large symmetric warhead.
        options.append((full_matches, fragment_matches))
        coverages.append(coverage)

    # Check every possible attachment even when only one representative is
    # needed; a truncated scan must never hide an ambiguous linker bridge.
    joint_count = 0
    attachments = set()
    selected = None
    selected_key = None
    hypotheses = []
    seen_geometries = set()
    xyz1 = fragments[0].GetConformer().GetPositions()
    xyz2 = fragments[1].GetConformer().GetPositions()
    bridge_cache = {}
    right_options = [(atoms, set(atoms), frozenset(atoms))
                     for atoms in options[1][0]]
    for left_full in options[0][0]:
        left_atoms = set(left_full)
        left_key = frozenset(left_atoms)
        for left_fragment in options[0][1]:
            left = (left_full, left_fragment)
            for right_full, right_atoms, right_key in right_options:
                if not left_atoms.isdisjoint(right_atoms):
                    continue
                bridge_key = (left_key, right_key)
                if bridge_key not in bridge_cache:
                    bridge_cache[bridge_key] = _bridge_attachment(
                        full, left_full, right_full)
                bridge = bridge_cache[bridge_key]
                if bridge is None:
                    continue
                attachments.add((bridge[0], bridge[1]))
                if len(attachments) > 1:
                    return None
                for right_fragment in options[1][1]:
                    right = (right_full, right_fragment)
                    joint_count += 1
                    if (hypothesis_limit is None and geometry_limit is None
                            and joint_count > 200000):
                        raise ValueError(
                            'too many binary anchor symmetries; request a '
                            'bounded geometry_limit before linker search')
                    key = (left_full, left_fragment, right_full, right_fragment)
                    if selected_key is None or key < selected_key:
                        selected_key = key
                        selected = (left, right, bridge)
                    retain = (hypothesis_limit is None
                              or len(hypotheses) < hypothesis_limit)
                    if geometry_limit is not None:
                        retain = len(hypotheses) < geometry_limit
                        if retain:
                            signature = (
                                tuple(sorted((full_atom, fragment_atom)
                                             for full_atom, fragment_atom in
                                             zip(left_full, left_fragment)
                                             if geometry_fragment_atoms is None
                                             or fragment_atom in
                                             geometry_fragment_atoms[0])),
                                tuple(sorted((full_atom, fragment_atom)
                                             for full_atom, fragment_atom in
                                             zip(right_full, right_fragment)
                                             if geometry_fragment_atoms is None
                                             or fragment_atom in
                                             geometry_fragment_atoms[1])))
                            retain = signature not in seen_geometries
                            if retain:
                                seen_geometries.add(signature)
                    if retain:
                        hypotheses.append({
                            'anchor1_atoms': left_full,
                            'anchor2_atoms': right_full,
                            'anchor1_fragment_atoms': left_fragment,
                            'anchor2_fragment_atoms': right_fragment,
                            'anchor1_xyz': xyz1[list(left_fragment)],
                            'anchor2_xyz': xyz2[list(right_fragment)],
                        })
    if not joint_count:
        return None
    # Keep the full set by default; a caller may request only its first map
    # for a cheap physical gate, then retry with the full set on failure.
    hypotheses = tuple(hypotheses)
    return {'anchor1_atoms': selected[0][0], 'anchor2_atoms': selected[1][0],
            'anchor1_xyz': xyz1[list(selected[0][1])],
            'anchor2_xyz': xyz2[list(selected[1][1])],
            'attachment_atoms': next(iter(attachments)),
            'bridge_atoms': tuple(sorted(selected[2][2])),
            'binary_core_coverage': tuple(coverages),
            'symmetry_mapping_count': joint_count,
            'mapping_hypotheses': hypotheses,
            'selected_mapping_policy': (
                'first_hypothesis_after_full_attachment_audit'
                if hypothesis_limit is not None
                else 'bounded_distinct_geometry_after_full_attachment_audit'
                if geometry_limit is not None
                else 'inspection_only_enumerate_for_inference'),
            'geometry_fragment_atoms': (
                tuple(tuple(sorted(atoms)) for atoms in geometry_fragment_atoms)
                if geometry_fragment_atoms is not None else None),
            'atom_order_graph_sha256': atom_order_graph_signature(full),
            'contract': 'external_binary_exact_bond_anchor_v1',
            'native_ternary_coordinates_used': False}


def linker_torsion_mask(ligand_data, anchor1_atoms, anchor2_atoms):
    """Allow only torsions that separate the two complete rigid warheads."""
    import torch

    masks = ligand_data['mask_rotate']
    if isinstance(masks, (tuple, list)):
        if len(masks) != 1:
            raise ValueError('expected one ligand torsion mask')
        masks = masks[0]
    masks = torch.as_tensor(masks, dtype=torch.bool,
                            device=ligand_data['coords'].device)
    first = torch.as_tensor(anchor1_atoms, dtype=torch.long, device=masks.device)
    second = torch.as_tensor(anchor2_atoms, dtype=torch.long, device=masks.device)
    if (masks.ndim != 2 or len(first) < 3 or len(second) < 3
            or set(first.tolist()) & set(second.tolist())
            or first.min() < 0 or second.min() < 0
            or max(int(first.max()), int(second.max())) >= masks.shape[1]):
        raise ValueError('invalid disjoint warhead/torsion correspondence')
    first_inside = masks[:, first].all(-1)
    first_outside = (~masks[:, first]).all(-1)
    second_inside = masks[:, second].all(-1)
    second_outside = (~masks[:, second]).all(-1)
    return (first_inside & second_outside) | (first_outside & second_inside)


def anchored_flow_step(ligand, protein2, body, velocities, dt, ligand_data,
                       protein2_reference, condition):
    """Execute a linker-only update and reassemble the two bound binary units.

    Protein rigid velocities and whole-ligand rigid drift are deliberately not
    applied as independent actions: they would violate the observed anchors.
    ``condition`` contains external binary poses at inference, or explicitly
    marked pseudo-binary poses during training.
    """
    import torch

    from tricoopdiff.sdpi_flow import SDPIFlow

    allowed = linker_torsion_mask(ligand_data, condition['anchor1_atoms'],
                                  condition['anchor2_atoms'])
    if not bool(allowed.any()):
        raise ValueError('no torsion links the two bound warheads')
    if velocities['torsion'].shape != (1, len(allowed)):
        raise ValueError('torsion velocity does not match the ligand graph')
    constrained = {**velocities,
                   'torsion': velocities['torsion'] * allowed[None]}
    changed, _, _ = SDPIFlow.differentiable_step(
        ligand, protein2, body, constrained, dt, ligand_data, freeze_poi=True)
    placed = assemble_from_binary_anchors(
        changed[0], protein2_reference[0],
        torch.as_tensor(condition['anchor1_xyz'], device=ligand.device,
                        dtype=ligand.dtype),
        torch.as_tensor(condition['anchor2_xyz'], device=ligand.device,
                        dtype=ligand.dtype),
        condition['anchor1_atoms'], condition['anchor2_atoms'])
    return (placed['ligand'][None], placed['protein2'][None],
            placed['protein2_rotation'][None], placed['anchor_rmsd'])
