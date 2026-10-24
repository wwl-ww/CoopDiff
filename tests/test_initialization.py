"""Initialization regression checks. Run with pytest or execute this file directly."""

if __name__ == "__main__":
    from helpers import run_tests

    run_tests(__file__)

import hashlib
import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial.transform import Rotation


def test_hdock_cache_hit_rebuilds_createpl_inputs():
    import assembly_pipeline as pipeline

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        entry = {'case': 'cached', 'protein1': 'p1', 'protein2': 'p2'}
        opt = SimpleNamespace(seed=73, hdock_dir=directory, fftw_lib_dir=directory)
        identity = {'inputs': {'protein1': 'one', 'protein2': 'two'},
                    'seed': 73, 'hdock_sha256': 'binary'}
        (directory / 'docking_complete.json').write_text(json.dumps(identity))
        (directory / 'Hdock.out').touch()
        with patch.object(pipeline, 'input_identity', return_value=identity['inputs']), \
             patch.object(pipeline, 'sha256_file', return_value='binary'), \
             patch.object(pipeline, 'write_hdock_input') as write_input:
            assert pipeline._docking_job_locked(opt, entry, directory) is entry
        assert write_input.call_count == 2, 'cached Hdock.out still needs r.pdb and l.pdb for createpl'


def test_pure_ternary_checkpoint_rejects_old_reference_contract():
    import assembly_pipeline as pipeline
    from tricoopdiff.models.assembly import (
        DRIFTING_MOTION_CONTRACT, PURE_CANDIDATE_REFERENCE_CONTRACT,
        PURE_TERNARY_MODEL_PROFILE, production_model_config,
    )

    config = production_model_config(
        16, 16, True, model_profile=PURE_TERNARY_MODEL_PROFILE,
        candidate_driven=True, motion_contract=DRIFTING_MOTION_CONTRACT,
    )
    assert config['candidate_reference_contract'] == PURE_CANDIDATE_REFERENCE_CONTRACT
    old_config = {**config,
                  'candidate_reference_contract': 'immutable_pose_displacement_transport_v1'}
    payload = {'model_config': old_config, 'format': pipeline.CANDIDATE_FRAMEWORK_VERSION}
    try:
        pipeline.model_for(SimpleNamespace(objective='drifting', motion_contract=None,
                                           device='cpu', weights_from=None), 16, payload)
    except ValueError as error:
        assert 'compatible candidate-driven model' in str(error)
    else:
        raise AssertionError('old reference contract entered the new pure ternary model')


def test_old_weights_cannot_silently_skip_contact_guidance():
    import assembly_pipeline as pipeline
    from tricoopdiff.models.assembly import (
        DRIFTING_MOTION_CONTRACT, PURE_TERNARY_MODEL_PROFILE, production_model_config,
    )

    config = production_model_config(
        16, 16, True, model_profile=PURE_TERNARY_MODEL_PROFILE,
        candidate_driven=True, motion_contract=DRIFTING_MOTION_CONTRACT,
        contact_guidance_contract=None,
    )
    payload = {'model_config': config, 'format': pipeline.CANDIDATE_FRAMEWORK_VERSION}
    opt = SimpleNamespace(objective='drifting', motion_contract=None, device='cpu',
                          weights_from=Path('old_checkpoint.pt'))
    try:
        pipeline.model_for(opt, 16, payload)
    except ValueError as error:
        assert 'cannot silently start the contact-guided' in str(error)
    else:
        raise AssertionError('old weights silently disabled the new contact-guided model')


def test_online_cache_reuses_pending_slots_and_never_generates_after_full():
    import assembly_pipeline as pipeline
    from tricoopdiff.runtime import atomic_torch_save
    with tempfile.TemporaryDirectory() as temporary:
        opt = SimpleNamespace(bank=Path(temporary), online_cache_size=4, seed=73)
        entry = {'case': 'case'}
        directory = opt.bank/'case'
        base = {'format': 'test', 'generator_sha256': 'testdigest'}
        case_seed = (73+int(hashlib.sha256(b'case').hexdigest()[:8], 16)) % (2**31-1)
        for slot in range(4):
            protocol = {**base, 'cache_contract': 'fill_new_then_uniform_fixed_slots_v1',
                'capacity': 4, 'slot': slot, 'slot_seed': (case_seed+1009*(slot+1)) % (2**31-1)}
            atomic_torch_save({'protocol': protocol, 'candidate': {'slot': slot}},
                              directory/f'online_slot_{slot}.pt')
        with patch.object(pipeline, 'generation_protocol', return_value=base), \
             patch.object(pipeline, 'docking_job', side_effect=AssertionError('unexpected generation')):
            # Even if all files exist after an interrupted update, replay the
            # pending visit's new slot rather than randomly drawing another.
            for visit in range(4):
                candidate, _, _ = pipeline.online_candidate(opt, entry, None, visit)
                assert candidate['slot'] == visit
            choices = [pipeline.online_candidate(opt, entry, None, visit)[0]['slot']
                       for visit in range(4, 104)]
            assert set(choices) == {0, 1, 2, 3}
            again = [pipeline.online_candidate(opt, entry, None, visit)[0]['slot']
                     for visit in range(4, 104)]
            assert choices == again, 'resume changed online cache choices'
            assert len(list(directory.glob('online_slot_*.pt'))) == 4
            from tricoopdiff.runtime import atomic_json
            atomic_json({'protocol': base, 'reason': 'joint ligand ETKDG generation failed'},
                        directory/'initialization_failed.testdigest.json')
            try:
                pipeline.online_candidate(opt, entry, None, 104)
            except pipeline.ConformerGenerationFailure:
                pass
            else:
                raise AssertionError('failed ETKDG case was retried instead of skipped')


def test_cached_only_pipeline_never_rebuilds_missing_features():
    import assembly_pipeline as pipeline
    with tempfile.TemporaryDirectory() as temporary:
        opt = SimpleNamespace(cache_dir=temporary, feature_cache_source=None,
                              cached_only=True, device='cpu')
        entry = {'case': 'missing', 'protein1': 'p1', 'protein2': 'p2', 'ligand': 'lig'}
        error = FileNotFoundError('missing: cache-only mode requires a compatible cache directory')
        with patch.object(pipeline, 'featurize_complex', side_effect=error) as read:
            try:
                pipeline.load_case(opt, entry, None)
            except pipeline.FeatureCacheUnavailable:
                pass
            else:
                raise AssertionError('missing cache was reconstructed')
            assert read.call_args.kwargs['cache_only'] is True
        # Structural cache corruption is not reclassified as missing data.
        with patch.object(pipeline, 'featurize_complex', side_effect=ValueError('corrupt tensor')):
            try:
                pipeline.load_case(opt, entry, None)
            except ValueError as exc:
                assert str(exc) == 'corrupt tensor'
            else:
                raise AssertionError('cache corruption was silently ignored')


def test_cached_only_admission_is_pinned_without_reordering_cases():
    import assembly_pipeline as pipeline
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        index = root/'index.json'
        index.touch()
        cache = root/'features'
        cache.mkdir()
        (cache/'b.pt').touch()
        opt = SimpleNamespace(output=root, cache_dir=cache, feature_cache_source=None, index=index)
        entries = [{'case': 'a'}, {'case': 'b'}, {'case': 'c'}]
        manifest = pipeline.cached_admission(opt, entries)
        assert manifest['ordered_cases'] == ['a', 'b', 'c']
        assert manifest['available_cases'] == ['b']
        assert manifest['missing_cache_cases'] == ['a', 'c']
        (cache/'a.pt').touch()
        assert pipeline.cached_admission(opt, entries) == manifest, 'resume expanded its dataset'


def test_retained_pose_generator_identity_is_explicit_and_not_rewritten():
    import assembly_pipeline as pipeline
    opt = SimpleNamespace(reuse_generator_sha256=['old'])
    expected = {'generator_sha256': 'new', 'seed': 73, 'expert': 'fixed'}
    recorded = {**expected, 'generator_sha256': 'old'}
    assert pipeline.same_generation_protocol(recorded, expected, opt)
    assert recorded['generator_sha256'] == 'old'
    assert not pipeline.same_generation_protocol({**recorded, 'seed': 74}, expected, opt)
    assert not pipeline.same_generation_protocol({**recorded, 'generator_sha256': 'unknown'}, expected, opt)


def test_shared_conformer_policy_embeds_once_with_random_coordinates():
    from rdkit.Chem import AllChem
    from tricoopdiff.initialization import embed_chemical_graph
    original = AllChem.EmbedMolecule
    calls = []

    def checked(mol, params):
        calls.append(params.useRandomCoords)
        return original(mol, params)

    with patch.object(AllChem, 'EmbedMolecule', side_effect=checked):
        result, attempts, random_coords = embed_chemical_graph(Chem.MolFromSmiles('CCCC'), 73)
    assert calls == [True], 'shared generation duplicated embedding or changed SDPI policy'
    assert attempts == 1 and random_coords and result.GetNumConformers() == 1


def test_cluster_epoch_plan_reuses_original_sampling_and_preserves_rng():
    import assembly_pipeline as pipeline
    entries = [{'case': name} for name in 'abcdefgh']
    clusters = [{'items': list(pair), 'representative': [pair[0]]}
                for pair in ('ab', 'cd', 'ef', 'gh')]
    available = set('abdefgh')
    np.random.seed(321)
    expected_next = np.random.random()
    np.random.seed(321)
    selected, shards, owners = pipeline.cluster_epoch_plan(entries, clusters, available, 73, 2, 4)
    assert np.random.random() == expected_next, 'epoch selection consumed rollout RNG'
    eligible = [e for e in entries if e['case'] in available]
    loader = SimpleNamespace(entries=eligible, entries_by_case={e['case']: e for e in eligible},
                             clusters=clusters, shuffle=True)
    np.random.seed(75)
    original = pipeline.TernaryDataLoader._epoch_entries(loader)
    assert selected == original, 'cluster 80/20 choices or shuffle differs from original loader'
    assert len(selected) == 4 and [len(s) for s in shards] == [1]*4
    assert sum(len(s) for s in shards) == len(selected)
    assert 'c' not in [e['case'] for e in selected]
    for rank, shard in enumerate(shards):
        assert all(owners[entries[i]['case']] == rank for i in shard)


def test_assembly_candidate_frame_and_rigidity():
    from tricoopdiff.initialization.candidates import assembly_candidate_from_pose
    from tricoopdiff.geometry.rigid import rotation_exp

    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
    poi = e3 * 1.2
    ligand = e3[:3] + 1.
    body = rotation_exp(torch.tensor([.1, -.2, .3]))
    placed = apply_rigid(poi, body, torch.tensor([8., 2., 1.]))
    frame = rotation_exp(torch.tensor([-.3, .2, .1]))
    offset = torch.tensor([13., -8., 4.])
    pose_e3 = apply_rigid(e3, frame, offset)
    pose_poi = apply_rigid(placed, frame, offset)
    pose_ligand = apply_rigid(ligand, frame, offset)
    result = assembly_candidate_from_pose(e3, poi, pose_e3, pose_poi, pose_ligand,
                                         candidate_id='generated-rank1', rank=1)
    assert torch.allclose(result['ligand'][0], ligand, atol=1e-5), 'ligand lost shared E3 frame'
    assert torch.allclose(result['poi'][0], placed, atol=1e-5), 'protein relative pose changed'
    assert torch.allclose(result['rotation'][0], body, atol=1e-5), 'body rotation convention changed'
    assert result['provenance']['candidate_source'] == 'hdock_etkdg'
    assert not result['provenance']['native_information_used']
    distorted = pose_poi.clone()
    distorted[0, 0] += .5
    try:
        assembly_candidate_from_pose(e3, poi, pose_e3, distorted, pose_ligand,
                                     candidate_id='distorted', rank=1)
    except ValueError as error:
        assert 'rigid image' in str(error)
    else:
        raise AssertionError('nonrigid protein candidate was silently projected')


def test_short_chain_candidate_keeps_allatom_supplied_rotation():
    from tricoopdiff.initialization.candidates import assembly_candidate_from_pose
    from tricoopdiff.geometry.rigid import rotation_exp

    e3 = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
    poi = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    rotation = rotation_exp(torch.tensor([.5, 0., 0.]))
    placed = apply_rigid(poi, rotation, torch.tensor([4., 2., 1.]))
    result = assembly_candidate_from_pose(e3, poi, e3, placed, e3[:2]+1.,
        candidate_id='short-chain', rank=1, poi_rotation=rotation)
    assert torch.equal(result['rotation'][0], rotation), 'CA-only fit discarded rotation around the two-point axis'
    try:
        assembly_candidate_from_pose(e3, poi, e3, placed, e3[:2]+1., candidate_id='short-chain', rank=1)
    except ValueError:
        pass
    else:
        raise AssertionError('a two-point CA reference must not invent a full body orientation')

from helpers import flow_example, ligand_input
from tricoopdiff.geometry.rigid import (
    _ProperPolarRotation,
    apply_rigid,
    assemble_interface_pair,
    invert_rigid,
    joint_surface_anchors,
    proper_rigid_fit,
    surface_frame,
)
from tricoopdiff.initialization import (
    INITIAL_NOISE_STD_PROPORTION,
    TernaryInitializer,
    receptor_prior,
    validate_ligand_graph,
)
from tricoopdiff.initialization.audit import (
    conditional_translation_kl,
    invariant_atom_pairs,
    invariant_geometry_gap,
)
from tricoopdiff.initialization.candidates import (
    FixedCandidateInitializer,
    generate_hybrid_candidates,
    generate_sdpi_candidates,
    select_balanced_pairs,
    select_hybrid_walkers,
    select_partial_pairs,
    select_pose_consensus,
)
from tricoopdiff.initialization.endpoints import (
    CachedEndpointStore,
    DUAL_INIT_CACHE_VERSION,
    LazyTrainingEndpointStore,
    RESAMPLED_TRAINING_ENDPOINT_PROTOCOL,
    _fit_numpy_rigid,
    endpoint_input_signature,
    endpoint_internal_geometry,
    project_endpoint_coupling,
    validate_endpoint_cache,
)
from tricoopdiff.sdpi_flow import SDPIFlow, _raw_torsions


# Initialization

def test_initializer_preserves_all_caller_random_streams():
    import random
    from tricoopdiff.runtime import capture_rng_state, restore_rng_state

    data = ligand_input()
    e3, poi = torch.randn(8, 3), torch.randn(9, 3)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state)
    first = TernaryInitializer().initialize(e3, poi, data, seed=73)
    second = TernaryInitializer().initialize(e3, poi, data, seed=73)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    assert all(torch.equal(a, b) for a, b in zip(first[:3], second[:3]))


def test_exact_inverse_and_pair_distances():
    torch.manual_seed(7)
    for seed in range(8):
        ra, rb = torch.tensor(Rotation.random(2, random_state=seed).as_matrix())
        ta, tb = torch.randn(2, 3, dtype=torch.float64) * 30
        ligand = torch.randn(14, 3, dtype=torch.float64)
        a, b = torch.randn(2, 23, 3, dtype=torch.float64) * 20
        la, lb = apply_rigid(ligand, ra, ta), apply_rigid(ligand, rb, tb)
        placed_b, rotation = assemble_interface_pair(b, ra, ta, rb, tb)
        inv, offset = invert_rigid(rb, tb)
        assert torch.allclose(apply_rigid(lb, inv, offset), ligand, atol=1e-10)
        assert torch.allclose(rotation.T @ rotation, torch.eye(3).double(), atol=1e-10)
        assert torch.det(rotation) > 0
        assert (torch.cdist(placed_b, la) - torch.cdist(b, lb)).abs().max() < 1e-5
        # The A gauge stays unchanged; all A-side distances are preserved too.
        central_a = apply_rigid(a, *invert_rigid(ra, ta))
        assert (torch.cdist(central_a, ligand) - torch.cdist(a, la)).abs().max() < 1e-5


def test_receptor_scale_and_rejected_legacy_parameters():
    receptor = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.]])
    center, radius, sigma = receptor_prior(receptor)
    expected = ((receptor - center).square().sum(-1).mean()).sqrt()
    assert torch.allclose(sigma, expected * INITIAL_NOISE_STD_PROPORTION / 1.73)
    assert torch.allclose(receptor_prior(receptor * 3 + 25)[2], sigma * 3)
    for kwargs in ({"initial_noise_std_proportion": -1},
                   {"initial_noise_std_proportion": 1},
                   {"tr_sigma_max": 19}, {"poi_shell_min": 30},
                   {"coupled_radius": 5}):
        try:
            TernaryInitializer(**kwargs)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"legacy distribution accepted: {kwargs}")


def test_joint_initialization_is_shared_and_reproducible():
    data = ligand_input()
    torch.manual_seed(4)
    a = torch.randn(37, 3) * 8 + 12
    b = torch.randn(19, 3) * 3 - 8
    init = TernaryInitializer()
    l, p, rotation, meta = init.initialize(a, b, data, seed=12)
    again = init.initialize(a, b, data, seed=12)
    assert torch.equal(l, again[0]) and torch.equal(p, again[1])
    assert meta == again[3]
    assert meta['init_mode'] == 'sdpi'
    assert meta['version'] == 'sdpi_v1'
    # Native coordinates must not enter inference internal geometry.
    changed = {**data, "coords": torch.randn_like(data["coords"]) * 1000}
    assert torch.equal(l, init.initialize(a, b, changed, seed=12)[0])
    assert torch.allclose((p - p.mean(1, keepdim=True)) @ rotation,
                          (b - b.mean(0)).unsqueeze(0), atol=1e-5)
    sides = meta['sides']
    ra, rb = (torch.tensor(sides[s]['rotation']) for s in ('e3', 'poi'))
    ta, tb = (torch.tensor(sides[s]['translation']) for s in ('e3', 'poi'))
    shared = apply_rigid(l[0], *invert_rigid(ra, ta))
    lb = apply_rigid(shared, rb, tb)
    assert (torch.cdist(p[0], l[0]) - torch.cdist(b, lb)).abs().max() < 1e-5
    for name, receptor in (("e3", a), ("poi", b)):
        assert abs(sides[name]['sigma_init'] - receptor_prior(receptor)[2].item()) < 1e-6
    assert len(meta['torsion_updates']) == int(data['edge_mask'].sum())


def test_diffdock_conformer_policy_is_explicit_and_reproducible():
    data = ligand_input()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    init = TernaryInitializer(conformer_policy="diffdock_v1_1")
    first = init.initialize(receptor, receptor * 2, data, seed=17)
    second = init.initialize(receptor, receptor * 2, data, seed=17)
    assert torch.equal(first[0], second[0])
    assert first[3]["conformer"]["policy"] == "diffdock_v1_1"
    assert first[3]["conformer"]["method"] == "ETKDGv2"
    try:
        TernaryInitializer(conformer_policy="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown conformer policy accepted")


def test_bad_atom_and_torsion_maps_fail():
    data = ligand_input()
    mol = Chem.Mol(data['mol'])
    bonds = data['edge_index'].clone()
    bonds[0, 0] = 999
    for edges, masks in ((bonds, data['mask_rotate']),
                         (data['edge_index'], data['mask_rotate'].astype(float)),
                         (data['edge_index'].flip(0), data['mask_rotate']),
                         (data['edge_index'], ~data['mask_rotate'])):
        try:
            validate_ligand_graph(mol, edges, data['edge_mask'], masks)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid atom/directed torsion mapping accepted")


def test_torsion_is_applied_once_and_no_torsion_is_explicit():
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    data = ligand_input()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]])
    target = 'tricoopdiff.diffdock.utils.torsion.modify_conformer_torsion_angles'
    for disabled in (False, True):
        with patch(target, wraps=modify_conformer_torsion_angles) as torsion:
            *_, metadata = TernaryInitializer(no_torsion=disabled).initialize(
                receptor, receptor * 2, data, seed=5,
            )
        assert torsion.call_count == (0 if disabled else 1)
        assert metadata['no_torsion'] == disabled
        if disabled:
            assert not any(metadata['torsion_updates'])


def test_float64_initial_torsions_preserve_bond_geometry_precision():
    data = ligand_input()
    data['coords'] = data['coords'].double()
    receptor = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]],
                            dtype=torch.float64)
    rigid = TernaryInitializer(no_torsion=True).initialize(
        receptor, receptor * 2, data, seed=73)[0][0]
    flexible = TernaryInitializer().initialize(
        receptor, receptor * 2, data, seed=73)[0][0]
    u, v = data['edge_index']
    expected = torch.linalg.vector_norm(rigid[u] - rigid[v], dim=-1)
    actual = torch.linalg.vector_norm(flexible[u] - flexible[v], dim=-1)
    assert flexible.dtype == torch.float64
    assert torch.allclose(actual, expected, atol=1e-10, rtol=0.), (
        'float64 torsion initialization silently rounded the conformer to float32')


def test_sampler_rejects_removed_modes_and_mismatched_clock():
    from tricoopdiff.legacy.diffusion import InterfaceDiffusion
    from tricoopdiff.sampler import sample_single

    model = torch.nn.Linear(1, 1)
    data = {'coords': torch.ones(1, 4, 3)}
    for kwargs in ({'init_mode': 'dual_native'}, {'init_mode': 'native'}, {'init_mode': 'coupled5a'},
                   {'init_mode': 'anchor_local'}, {'t_anchor': 0.7},
                   {'steps': 0}, {'steps': -1}, {'steps': 1.5}):
        try:
            sample_single(model, InterfaceDiffusion(), TernaryInitializer(),
                          data, data, data, device='cpu', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"removed or mismatched prior accepted: {kwargs}")


def test_sampler_rejects_empty_ensemble():
    from tricoopdiff.sampler import sample_ensemble

    for kwargs in ({'n_walkers': 0}, {'seeds': []}):
        try:
            sample_ensemble(None, None, None, {}, {}, {}, **kwargs)
        except ValueError as error:
            assert 'walker' in str(error)
        else:
            raise AssertionError('empty ensemble must not be reported as completed')


def test_coupled_anchor_perturbation_preserves_local_geometry():
    from tricoopdiff.initialization.candidates import coupled_anchor_perturbation

    data = ligand_input('CCCCCCCC')
    ligand = data['coords']
    poi = ligand + torch.tensor([0., 0., 2.])
    center = {'ligand': ligand, 'poi': poi, 'rotation': torch.eye(3)[None]}
    e3 = ligand[0] - torch.tensor([0., 0., 2.])
    before = ligand.clone()
    result, meta = coupled_anchor_perturbation(
        center, e3, data, seed=73, translation_std=0., rotation_std=0.)
    assert meta['mode'] == 'two_anchor_coupled_torsion', 'separable anchors must use coupled motion'
    a, b = meta['e3_anchor_atoms'], meta['poi_anchor_atoms']
    assert torch.allclose(result['ligand'][0, a], ligand[0, a], atol=1e-4)
    assert torch.allclose(torch.cdist(result['ligand'][0, b], result['poi'][0]),
                          torch.cdist(ligand[0, b], poi[0]), atol=1e-4)
    edges = data['edge_index'].T
    def lengths(coords):
        return (coords[0, edges[:, 0]] - coords[0, edges[:, 1]]).norm(dim=-1)
    assert torch.allclose(lengths(result['ligand']), lengths(ligand), atol=1e-4)
    again, _ = coupled_anchor_perturbation(
        center, e3, data, seed=73, translation_std=0., rotation_std=0.)
    assert torch.equal(result['ligand'], again['ligand'])
    assert torch.equal(ligand, before), 'initialization must not mutate its center'
    rigid_data = ligand_input('c1ccccc1')
    rigid_center = {'ligand': rigid_data['coords'], 'poi': poi, 'rotation': torch.eye(3)[None]}
    _, rigid_meta = coupled_anchor_perturbation(rigid_center, e3, rigid_data, seed=73)
    assert rigid_meta['mode'] == 'rigid_only_no_separable_anchors'


# Dual init

def dual_init_payload(source='bitseed', fallback=None):
    record = {
        'case': 'demo', 'actual_source': source,
        'candidate_id': ['bitseed', 0, 1, 2], 'conformer_seed': 17,
        'binary_pose_ids': [1, 2], 'fallback': fallback,
        'native_used_in_candidate_generation': False,
        'native_used_in_candidate_selection': False,
        'native_used_for_training_projection': True,
        'endpoint': {
            'ligand': np.zeros((4, 3)), 'poi': np.zeros((5, 3)),
            'body_frame': np.eye(3),
        },
        'training_endpoint': {
            'ligand': np.zeros((4, 3)), 'poi': np.zeros((5, 3)),
            'body_frame': np.eye(3),
        },
        'coupling': {
            'angles': np.zeros(1), 'omegas': np.zeros((2, 3)),
            'centers': np.zeros((2, 3)), 'bonds': np.array([[1, 2]]),
            'masks': np.array([[False, False, True, True]]),
        },
        'projection': {'contract': 'torsion_rigid_endpoint_projection_v1'},
    }
    return {
        'cache_version': DUAL_INIT_CACHE_VERSION,
        'generation_protocol': {'frozen_expert_checkpoint_sha256': 'abc123'},
        'entries': [record],
    }


def test_cache_roundtrip_preserves_deterministic_coupling():
    value = dual_init_payload()
    validate_endpoint_cache(value)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'endpoints.pt'
        torch.save(value, path)
        store = CachedEndpointStore(path)
        first, record = store.coupling('demo', occurrence=4)
        first['angles'][0] = 99
        second, _ = store.coupling('demo', occurrence=4)
        assert second['angles'][0] == 0
        assert record['actual_source'] == 'bitseed'
        record['coupling']['angles'][0] = 88
        record['actual_source'] = 'sdpi'
        third, fresh_record = store.coupling('demo', occurrence=4)
        assert third['angles'][0] == 0 and fresh_record['actual_source'] == 'bitseed', (
            'returned metadata must not expose mutable cached training state')


def test_fallback_cannot_be_relabelled_as_bitseed():
    value = dual_init_payload(fallback='sdpi_v1')
    try:
        validate_endpoint_cache(value)
    except ValueError as error:
        assert 'fallback' in str(error)
    else:
        raise AssertionError('fallback endpoint must retain its actual SDPI source')


def test_ood_expert_failure_is_a_valid_sdpi_fallback_record():
    value = dual_init_payload(source='sdpi', fallback='sdpi_v1')
    validate_endpoint_cache(value)


def test_lazy_training_protocol_is_exactly_one_pose_per_expert():
    expert = type('Expert', (), {'provenance': {
        'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc',
        'no_torsion': False,
    }})()
    with tempfile.TemporaryDirectory() as directory:
        store = LazyTrainingEndpointStore(directory, expert)
        assert store.protocol == {
            'version': RESAMPLED_TRAINING_ENDPOINT_PROTOCOL,
            'conformers': 1, 'poses_per_side': 1, 'binary_steps': 20,
            'binary_retry_limit': 1, 'selection': 'single_pair_no_ranking',
            'fallback': 'sdpi_v1_on_any_invalid_expert_side',
            'base_seed': 41000,
            'expert_runtime': store.protocol['expert_runtime'],
            'generation_version': 'raw_target_projection_v4',
            'frozen_expert': {
                'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc',
                'no_torsion': False,
            },
        }


def dual_init_input_batch():
    return {
        'e3_data': {'coords': torch.zeros(1, 3, 3), 'features': torch.ones(1, 3, 2)},
        'poi_data': {'coords': torch.zeros(1, 5, 3)},
        'ligand_data': {
            'coords': torch.arange(12).float().reshape(1, 4, 3),
            'mol': b'ordered chemistry', 'edge_index': torch.tensor([[1], [2]]),
            'mask_rotate': np.array([[False, False, True, True]]),
        },
    }


def dual_init_lazy_store(directory):
    expert = type('Expert', (), {'provenance': {
        'diffdock_commit': 'test', 'score_checkpoint_sha256': 'abc', 'no_torsion': False,
    }})()
    return LazyTrainingEndpointStore(directory, expert)


def test_resampled_endpoints_use_occurrence_and_keep_validation_fixed():
    from unittest.mock import patch
    from tricoopdiff.initialization.endpoints import RESAMPLED_TRAINING_ENDPOINT_PROTOCOL

    with tempfile.TemporaryDirectory() as directory:
        expert = dual_init_lazy_store(directory).expert
        store = LazyTrainingEndpointStore(directory, expert)
        assert store.protocol['version'] == RESAMPLED_TRAINING_ENDPOINT_PROTOCOL

        def generate(case, batch, signature, occurrence):
            record = deepcopy(dual_init_payload()['entries'][0])
            record['endpoint']['ligand'] += occurrence
            record['target_ligand'] = np.zeros((4, 3))
            return record

        with patch.object(store, '_generate', side_effect=generate) as generation:
            a, _ = store.coupling('demo', occurrence=0, batch=dual_init_input_batch())
            repeated, _ = store.coupling('demo', occurrence=0, batch=dual_init_input_batch())
            b, _ = store.coupling('demo', occurrence=1, batch=dual_init_input_batch())
            validation, _ = store.coupling('demo', occurrence=0, batch=dual_init_input_batch())
        assert generation.call_count == 3
        assert np.array_equal(a['initial_state']['ligand'], repeated['initial_state']['ligand'])
        assert not np.array_equal(a['initial_state']['ligand'], b['initial_state']['ligand'])
        assert np.array_equal(a['initial_state']['ligand'], validation['initial_state']['ligand'])
        assert store._path('demo', 0) != store._path('demo', 1)
        assert len(store._records) == 1, 'resampling must not accumulate every occurrence in RAM'


def test_cache_contract_includes_and_rechecks_actual_expert_policy():
    from types import SimpleNamespace
    from tricoopdiff.models.expert import configure_expert_policy

    with tempfile.TemporaryDirectory() as directory:
        expert = dual_init_lazy_store(directory).expert
        expert.model = SimpleNamespace()
        configure_expert_policy(expert, 100., 'irrep_norm')
        a = LazyTrainingEndpointStore(directory, expert)
        configure_expert_policy(expert, 1., 'componentwise')
        b = LazyTrainingEndpointStore(directory, expert)
        assert a.protocol != b.protocol
        try:
            a.coupling('demo', batch=dual_init_input_batch())
        except ValueError as error:
            assert 'runtime changed' in str(error)
        else:
            raise AssertionError('a cache must not outlive a change in expert policy')


def test_endpoint_signature_tracks_geometry_chemistry_and_atom_order():
    batch = dual_init_input_batch()
    signature = endpoint_input_signature(batch)
    assert endpoint_input_signature(deepcopy(batch)) == signature
    for component, key, replacement in (
        ('ligand_data', 'coords', torch.zeros(1, 6, 3)),
        ('ligand_data', 'coords', batch['ligand_data']['coords'].flip(1)),
        ('ligand_data', 'mol', b'different chemistry'),
        ('ligand_data', 'edge_index', torch.tensor([[2], [1]])),
        ('ligand_data', 'mask_rotate', np.ones((1, 4), dtype=bool)),
        ('e3_data', 'coords', torch.ones(1, 3, 3)),
        ('e3_data', 'features', torch.zeros(1, 3, 2)),
        ('poi_data', 'coords', torch.ones(1, 5, 3)),
    ):
        changed = deepcopy(batch)
        changed[component][key] = replacement
        assert endpoint_input_signature(changed) != signature, (component, key)


def test_lazy_cache_checks_disk_and_memory_inputs_without_overwriting():
    with tempfile.TemporaryDirectory() as directory:
        store = dual_init_lazy_store(directory)
        batch = dual_init_input_batch()
        path = store._path('demo')
        record = deepcopy(dual_init_payload()['entries'][0])
        record['occurrence'] = 0
        record['target_ligand'] = np.zeros((4, 3))
        torch.save({'case': 'demo', 'occurrence': 0, 'protocol': store.protocol,
                    'record': record, 'input_signature': endpoint_input_signature(batch)}, path)
        original = path.read_bytes()
        first, record = store.coupling('demo', batch=batch)
        first['initial_state']['ligand'][0, 0] = 99
        record['endpoint']['ligand'][0, 0] = 88
        record['target_ligand'][0, 0] = 77
        record['actual_source'] = 'sdpi'
        again, fresh_record = store.coupling('demo', batch=deepcopy(batch))
        assert again['initial_state']['ligand'][0, 0] == 0
        assert again['target_ligand'][0, 0] == 0 and fresh_record['actual_source'] == 'bitseed', (
            'returned metadata changed deterministic cached inputs, labels or source')
        changed = deepcopy(batch)
        changed['ligand_data']['coords'] = torch.ones(1, 6, 3)
        for reader in (store, dual_init_lazy_store(directory)):
            try:
                reader.coupling('demo', batch=changed)
            except ValueError as error:
                assert 'input signature' in str(error)
            else:
                raise AssertionError('changed input must not reuse the cached coupling')
        assert path.read_bytes() == original, 'rejection must leave old cache untouched'


def test_read_only_endpoint_follower_never_generates_missing_case():
    with tempfile.TemporaryDirectory() as directory:
        store = LazyTrainingEndpointStore(
            directory, dual_init_lazy_store(directory).expert, read_only=True, wait_seconds=0.)
        try:
            store.coupling('missing', batch=dual_init_input_batch())
        except TimeoutError as error:
            assert 'read-only endpoint follower' in str(error)
        else:
            raise AssertionError('read-only follower generated a missing endpoint')


def test_lazy_legacy_cache_and_missing_batch_are_rejected():
    with tempfile.TemporaryDirectory() as directory:
        store = dual_init_lazy_store(directory)
        path = store._path('demo')
        torch.save({'case': 'demo', 'protocol': store.protocol,
                    'record': dual_init_payload()['entries'][0]}, path)
        for batch in (None, dual_init_input_batch()):
            try:
                store.coupling('demo', batch=batch)
            except ValueError as error:
                assert 'batch is required' in str(error) or 'signature missing' in str(error)
            else:
                raise AssertionError('unverified legacy cache must be rejected')


# Endpoint kinematics

ENDPOINT_EDGE_INDEX = np.array([
    [0, 1, 1, 2, 2, 3],
    [1, 0, 2, 1, 3, 2],
])


def test_rigid_rotation_gradient_preserves_covariance_scale_law():
    covariance = torch.tensor([[3., .1, .2], [.3, 2., .1], [.1, .2, 1.]],
                              dtype=torch.float64)
    probe = torch.tensor([[1., 2., 3.], [4., 2., 5.], [6., 7., 3.]],
                         dtype=torch.float64)
    reference_rotation = reference_gradient = None
    for scale in (1., 1e-6, 1e-10):
        value = (covariance * scale).requires_grad_()
        rotation = _ProperPolarRotation.apply(value)
        gradient, = torch.autograd.grad((rotation * probe).sum(), value)
        if reference_rotation is None:
            reference_rotation, reference_gradient = rotation, gradient
        assert torch.allclose(rotation, reference_rotation, atol=1e-12, rtol=0.)
        assert torch.allclose(gradient * scale, reference_gradient, atol=1e-12, rtol=0.), (
            'a well-conditioned small covariance changed the rigid-fit gradient')
        assert torch.autograd.gradcheck(_ProperPolarRotation.apply, (value,),
                                        eps=scale * 1e-5, atol=1e-4, rtol=1e-4)


def test_linear_rigid_fit_preserves_correspondence_under_roundoff():
    source = np.array([[-1., 0., 0.], [1., 0., 0.]])
    for perturbation in (-1e-12, 0., 1e-12):
        target = source + [10., 2., 3.]
        target[0, 0] += perturbation
        rotation, translation, rmsd, underdetermined = _fit_numpy_rigid(source, target)
        assert underdetermined and rmsd < 1e-10, 'roundoff reversed atom correspondence'
        assert np.allclose(source @ rotation.T + translation, target, atol=1e-10)
        assert np.isclose(np.linalg.det(rotation), 1.), 'rigid fit introduced a reflection'
    rotation, translation, rmsd, underdetermined = _fit_numpy_rigid(source[:1], target[:1])
    assert underdetermined and rmsd < 1e-10
    assert np.allclose(source[:1] @ rotation.T + translation, target[:1])


def test_rigid_endpoint_preserves_sdpi_internal_invariants():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    endpoint = reference @ rotation.T + np.array([4.0, -3.0, 8.0])
    audit = endpoint_internal_geometry(reference, endpoint, ENDPOINT_EDGE_INDEX)
    assert audit["necessary_condition_passed"]
    assert audit["bond_lengths"]["max_abs_delta_angstrom"] < 1e-12
    assert audit["bonded_one_three_distances"]["max_abs_delta_angstrom"] < 1e-12


def test_distorted_endpoint_is_not_exactly_representable():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
    ])
    endpoint = reference.copy()
    endpoint[3, 2] += 0.2
    audit = endpoint_internal_geometry(reference, endpoint, ENDPOINT_EDGE_INDEX)
    assert not audit["necessary_condition_passed"]
    assert audit["bond_lengths"]["max_abs_delta_angstrom"] > 0.1


def test_projected_rigid_endpoint_roundtrips_at_flow_time_one():
    native_ligand = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0], [-2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    ligand_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    poi_rotation = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    endpoint_ligand = apply_rigid(
        native_ligand, ligand_rotation,
        torch.tensor([3.0, -2.0, 4.0], dtype=torch.float64),
    )
    endpoint_poi = apply_rigid(
        native_poi, poi_rotation,
        torch.tensor([-4.0, 5.0, 2.0], dtype=torch.float64),
    )
    ligand_data = {
        "coords": native_ligand,
        "edge_index": torch.as_tensor(ENDPOINT_EDGE_INDEX),
        "edge_mask": torch.zeros(ENDPOINT_EDGE_INDEX.shape[1], dtype=torch.bool),
        "mask_rotate": np.zeros((0, native_ligand.shape[1]), dtype=bool),
    }
    coupling, projected, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
        no_torsion=True,
    )
    ligand, poi, body, targets = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report["endpoint_exactly_represented"]
    assert np.allclose(projected["ligand"], endpoint_ligand[0].numpy(), atol=1e-8)
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-8)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)
    assert body.shape == (1, 3, 3)
    assert all(torch.isfinite(value).all() for branch in targets.values()
               for value in (branch.values() if isinstance(branch, dict) else (branch,)))


def test_projected_torsion_endpoint_roundtrips_at_flow_time_one():
    native_ligand = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0], [3.0, 1.0, 1.0], [3.5, 2.0, 1.5],
    ]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [-2.0, 0.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0], [-2.0, 1.0, 1.0],
    ]], dtype=torch.float64)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    edge_mask[4] = True
    masks = np.array([[False, False, False, True, True]])
    ligand_data = {
        'coords': native_ligand, 'edge_index': edge_index,
        'edge_mask': edge_mask, 'mask_rotate': masks,
    }
    torsioned = _raw_torsions(
        native_ligand[0].numpy(), np.array([[2, 3]]), masks, np.array([0.7]),
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    endpoint_ligand = apply_rigid(
        torch.as_tensor(torsioned)[None], rotation,
        torch.tensor([2.0, -3.0, 1.0], dtype=torch.float64),
    )
    endpoint_poi = native_poi + torch.tensor([[[4.0, 2.0, -1.0]]])
    coupling, _, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
    )
    ligand, poi, _, _ = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report['endpoint_exactly_represented']
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-6)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)


def test_periodic_multitorsion_endpoint_projection_recovers_reachable_targets():

    flow, data, _, poi, coupling = flow_example()
    rng = np.random.default_rng(631)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                            dtype=torch.float64)
    for _ in range(20):
        angles = rng.uniform(-np.pi, np.pi, len(coupling['bonds']))
        shape = _raw_torsions(data['coords'][0].numpy(), coupling['bonds'],
                              coupling['masks'], angles)
        endpoint = torch.as_tensor(shape)[None] @ rotation.T + 7.
        projected_coupling, projected, report = project_endpoint_coupling(
            data, poi, endpoint, poi)
        assert report['ligand_projection_rmsd_angstrom'] < 1e-8, (
            'a periodic torsion boundary/local minimum distorted an exactly reachable target')
        assert np.all(np.abs(projected_coupling['angles']) <= np.pi)
        ligand, _, _, _ = flow.path(data, poi, projected_coupling,
                                    torch.ones(1, dtype=torch.float64))
        assert torch.allclose(ligand, endpoint, atol=1e-8)
        assert np.allclose(projected['ligand'], endpoint[0].numpy(), atol=1e-8)


def test_projection_retries_exhausted_budget_without_accepting_failure():
    from unittest.mock import patch
    import tricoopdiff.initialization.endpoints as endpoints

    _, data, _, poi, _ = flow_example()
    solve = endpoints.least_squares
    calls = []
    def exhausted_once(fun, x, **kwargs):
        result = solve(fun, x, **kwargs)
        calls.append(kwargs['max_nfev'])
        if len(calls) == 1:
            result.success, result.status = False, 0
        return result
    with patch.object(endpoints, 'least_squares', side_effect=exhausted_once):
        _, _, report = endpoints.project_endpoint_coupling(data, poi, data['coords'], poi)
    assert calls[:2] == [400, 1200]
    assert report['torsion_optimization']['success']
    assert report['ligand_projection_rmsd_angstrom'] < 1e-8
    assert report['torsion_optimization']['attempts'][0]['status'] == 0


def test_single_atom_fallback_has_translation_only_roundtrip():
    native_ligand = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64)
    endpoint_ligand = torch.tensor([[[4.0, -2.0, 7.0]]], dtype=torch.float64)
    native_poi = torch.tensor([[
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ]], dtype=torch.float64)
    endpoint_poi = native_poi + torch.tensor([[[2.0, 3.0, 4.0]]])
    ligand_data = {
        'coords': native_ligand,
        'edge_index': torch.empty((2, 0), dtype=torch.long),
        'edge_mask': torch.empty(0, dtype=torch.bool),
        'mask_rotate': np.zeros((0, 1), dtype=bool),
    }
    coupling, _, report = project_endpoint_coupling(
        ligand_data, native_poi, endpoint_ligand, endpoint_poi,
        no_torsion=True,
    )
    ligand, poi, _, _ = SDPIFlow().path(
        ligand_data, native_poi, coupling, torch.ones(1, dtype=torch.float64),
    )
    assert report['ligand_orientation_underdetermined']
    assert report['endpoint_exactly_represented']
    assert torch.allclose(ligand, endpoint_ligand, atol=1e-8)
    assert torch.allclose(poi, endpoint_poi, atol=1e-8)


def test_torsion_projection_handles_reversed_reordered_rotors_and_global_frame():

    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                            dtype=torch.float64)
    for smiles in ('CC(C)CCOC(C)CC', 'c1ccccc1CCOCC'):
        _, data, _, poi, _ = flow_example(smiles)
        selected = torch.where(data['edge_mask'])[0]
        edges = data['edge_index'].clone()
        edges[:, selected] = edges[:, selected].flip(0).flip(1)
        masks = ~np.asarray(data['mask_rotate'])[::-1].copy()
        source = data['coords'] @ rotation.T + 11.
        data = {**data, 'coords': source, 'edge_index': edges, 'mask_rotate': masks}
        angles = np.linspace(-3.1, 2.9, len(selected))
        target = _raw_torsions(source[0].numpy(), edges[:, selected].T.numpy(),
                               masks, angles) @ rotation.numpy().T - 6.
        _, projected, report = project_endpoint_coupling(data, poi, target, poi)
        assert report['endpoint_exactly_represented']
        assert np.allclose(projected['ligand'], target, atol=1e-8), (
            'projection phase signs depended on rotor order, side or absolute frame')


# Pose anchors

def test_surface_frame_is_right_handed_and_orthonormal():
    frame = surface_frame(
        torch.tensor([[0.0, 0.0, 2.0]]),
        torch.tensor([[2.0, 1.0, 1.0]]),
    )
    assert torch.allclose(frame @ frame.transpose(-1, -2),
                          torch.eye(3).unsqueeze(0), atol=1e-6)
    assert torch.allclose(torch.linalg.det(frame), torch.ones(1), atol=1e-6)


def test_joint_anchor_maps_contact_point_and_opposes_normals():
    source_point = torch.tensor([[1.0, 2.0, 3.0]])
    source_normal = torch.tensor([[0.0, 0.0, 1.0]])
    source_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    target_point = torch.tensor([[5.0, -2.0, 1.0]])
    target_normal = torch.tensor([[0.0, 1.0, 0.0]])
    target_tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        source_point, source_normal, source_tangent,
        target_point, target_normal, target_tangent,
        torch.tensor([0.0]),
    )
    assert torch.allclose(anchors.place(source_point)[0, 0], target_point[0], atol=1e-6)
    mapped_normal = source_normal @ anchors.rotations[0]
    assert torch.allclose(mapped_normal, -target_normal, atol=1e-6)


def test_twists_create_distinct_orientations_at_the_same_position():
    point = torch.zeros(1, 3)
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    tangent = torch.tensor([[1.0, 0.0, 0.0]])
    anchors = joint_surface_anchors(
        point, normal, tangent, point, normal, tangent,
        torch.tensor([0.0, math.pi / 2]),
    )
    assert anchors.rotations.shape == (2, 3, 3)
    assert torch.allclose(anchors.place(point)[:, 0], point.expand(2, -1), atol=1e-6)
    assert not torch.allclose(anchors.rotations[0], anchors.rotations[1])


def test_proper_rigid_fit_recovers_exact_transform():
    source = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
    ])
    rotation = torch.tensor([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
    ])
    translation = torch.tensor([3.0, -2.0, 1.0])
    target = source @ rotation.T + translation
    fitted_rotation, fitted_translation, rmsd = proper_rigid_fit(source, target)
    fitted = source @ fitted_rotation.T + fitted_translation
    assert torch.allclose(fitted, target, atol=1e-5)
    assert torch.det(fitted_rotation) > 0
    assert rmsd < 1e-5


def test_pose_consensus_selects_dense_ternary_mode():
    def candidate(offset):
        coords = torch.tensor(offset, dtype=torch.float32).reshape(1, 1, 3)
        return {
            "ligand": coords, "poi": coords, "penalty": (0, 0, 0.0, 0.0),
        }

    candidates = [
        candidate([0.0, 0.0, 0.0]), candidate([0.2, 0.0, 0.0]),
        candidate([-0.2, 0.0, 0.0]), candidate([20.0, 0.0, 0.0]),
    ]
    selected = select_pose_consensus(candidates, distance_threshold=1.0)
    assert any(selected is candidate for candidate in candidates[:3])
    assert selected["consensus_neighbors_10A"] == 3


def test_balanced_pair_subsampling_has_equal_marginal_degree():
    for size, count in ((64, 256), (16, 64), (4, 16)):
        pairs = select_balanced_pairs(size, count, seed=31_000)
        assert pairs == select_balanced_pairs(size, count, seed=31_000)
        assert len(pairs) == len(set(pairs)) == count
        degree = count // size
        assert [sum(left == i for left, _ in pairs) for i in range(size)] == [degree] * size
        assert [sum(right == i for _, right in pairs) for i in range(size)] == [degree] * size


def test_partial_pair_subsampling_is_deterministic_and_diverse():
    pairs = select_partial_pairs(7, 8, 48, seed=31_000)
    assert pairs == select_partial_pairs(7, 8, 48, seed=31_000)
    assert len(pairs) == len(set(pairs)) == 48
    left_degree = [sum(left == i for left, _ in pairs) for i in range(7)]
    right_degree = [sum(right == i for _, right in pairs) for i in range(8)]
    assert max(left_degree) - min(left_degree) <= 1
    assert max(right_degree) - min(right_degree) <= 1


def test_hybrid_walker_selection_protects_sources_and_diversity():
    def candidate(source, index, offset, penalty):
        coords = torch.tensor([[[float(offset), 0.0, 0.0]]])
        return {
            "candidate_id": [source, index],
            "penalty": (0, 0, float(penalty), 0.0),
            "ligand": coords,
            "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        }

    candidates = [
        candidate("bitseed", 0, 0.0, 0),
        candidate("sdpi", 0, 0.2, 5),  # Same basin, but SDPI quota is protected.
        candidate("bitseed", 1, 10.0, 1),
        candidate("bitseed", 2, 20.0, 2),
        candidate("bitseed", 3, 30.0, 3),
        candidate("sdpi", 1, 40.0, 6),
    ]
    selected, metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert metadata["actual_source_counts"] == {"bitseed": 3, "sdpi": 1}
    assert metadata["unique_clusters_selected"] == 4
    assert selected[0]["candidate_id"] == ["sdpi", 0]
    repeated, repeated_metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert [item["candidate_id"] for item in repeated] == [
        item["candidate_id"] for item in selected
    ]
    assert repeated_metadata == metadata


def test_hybrid_walker_selection_second_pass_fills_exact_budget():
    candidates = []
    for index in range(4):
        coords = torch.tensor([[[0.1 * index, 0.0, 0.0]]])
        candidates.append({
            "candidate_id": ["bitseed" if index else "sdpi", index],
            "penalty": (0, 0, float(index), 0.0),
            "ligand": coords, "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        })
    selected, metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
    )
    assert len(selected) == 4
    assert metadata["cluster_count"] == 1
    assert metadata["unique_clusters_selected"] == 1


def test_hybrid_walker_selection_component_switches_and_random_seed():
    candidates = []
    for index in range(8):
        source = "sdpi" if index >= 6 else "bitseed"
        coords = torch.tensor([[[float(index) * 10, 0.0, 0.0]]])
        candidates.append({
            "candidate_id": [source, index],
            "penalty": (0, 0, float(index), 0.0),
            "ligand": coords, "poi": coords + 10,
            "rotation": torch.eye(3).unsqueeze(0),
        })
    geometry, geometry_metadata = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3,
        use_diversity=False, protect_sources=False,
    )
    assert all(item["candidate_id"][0] == "bitseed" for item in geometry)
    assert not geometry_metadata["use_diversity"]
    assert not geometry_metadata["protect_sources"]
    random_a, _ = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3, use_diversity=False,
        protect_sources=True, random_seed=123,
    )
    random_b, _ = select_hybrid_walkers(
        candidates, 4, bitseed_quota=3, use_diversity=False,
        protect_sources=True, random_seed=123,
    )
    assert [item["candidate_id"] for item in random_a] == [
        item["candidate_id"] for item in random_b
    ]
    assert sum(item["candidate_id"][0] == "sdpi" for item in random_a) == 1


def test_sdpi_fallback_preserves_candidate_budget_and_seed_schedule():
    class StubInitializer:
        def __init__(self):
            self.seeds = []

        def initialize(self, e3, poi, ligand_data, seed):
            self.seeds.append(seed)
            ligand = torch.tensor([[[4.0, 0.0, 0.0]]])
            placed_poi = torch.tensor([[[8.0, 0.0, 0.0]]])
            rotation = torch.eye(3).unsqueeze(0)
            return ligand, placed_poi, rotation, {"init_mode": "sdpi"}

    initializer = StubInitializer()
    candidates, records = generate_sdpi_candidates(
        initializer, torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]), {}, count=3, seed=31_000,
    )
    assert len(candidates) == len(records) == 3
    assert len(set(initializer.seeds)) == 3
    assert all(candidate["fallback"] == "sdpi_v1" for candidate in candidates)
    assert [candidate["candidate_id"] for candidate in candidates] == [
        ["sdpi", 0], ["sdpi", 1], ["sdpi", 2],
    ]


def test_hybrid_replaces_only_failed_conformer_with_sdpi():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            ligand = torch.tensor([[[4.0, 0.0, 0.0]]])
            placed_poi = torch.tensor([[[8.0, 0.0, 0.0]]])
            return ligand, placed_poi, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 2
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, seed, **kwargs):
            if seed == 11:
                raise FloatingPointError("bad conformer")
            candidates = []
            for left in range(2):
                for right in range(2):
                    candidates.append({
                        "attempt": [left, right],
                        "candidate_id": [0, left, right],
                    })
            record = {"conformer_index": 0, "binary_retry_counts": [[], []],
                      "binary_draw_counts": [2, 2],
                      "binary_failure_counts": [0, 0],
                      "binary_valid_pose_counts": [2, 2]}
            return candidates, [record], None

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=2, bitseed_candidate_budget=4,
        sdpi_candidate_budget=1,
    )
    assert len(candidates) == 5
    assert metadata["actual_bitseed"] == 2
    assert metadata["actual_sdpi"] == 3
    assert len(metadata["failed_conformers"]) == 1
    assert sum(candidate.get("fallback") == "sdpi_v1"
               for candidate in candidates) == 3


def test_hybrid_replaces_unfittable_conformer_with_sdpi():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            coords = torch.tensor([[[4.0, 0.0, 0.0]]])
            return coords, coords + 4, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 1
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, **kwargs):
            raise ValueError("rigid fit requires at least three points")

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=1, bitseed_candidate_budget=1,
        sdpi_candidate_budget=0,
    )
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == ["sdpi", 0]
    assert metadata["actual_bitseed"] == 0
    assert metadata["actual_sdpi"] == 1
    assert metadata["failed_conformers"][0]["error"].startswith("ValueError:")


def test_hybrid_keeps_partial_pairs_and_fills_only_the_deficit():
    class StubBase:
        def initialize(self, e3, poi, ligand_data, seed):
            coords = torch.tensor([[[float(seed), 0.0, 0.0]]])
            return coords, coords + 4, torch.eye(3).unsqueeze(0), {"seed": seed}

    class StubDual:
        proposal_attempts = 2
        base = StubBase()

        @staticmethod
        def _conformer_seed(seed, index):
            return seed + index

        def generate_candidates(self, *args, seed, **kwargs):
            candidates = [{
                "attempt": [0, 0], "candidate_id": [0, 0, 0],
            }]
            record = {
                "conformer_index": 0,
                "binary_retry_counts": [[0], [0]],
                "binary_draw_counts": [4, 1],
                "binary_failure_counts": [3, 0],
                "binary_valid_pose_counts": [1, 1],
            }
            return candidates, [record], None

    candidates, metadata = generate_hybrid_candidates(
        StubDual(), torch.zeros(1, 3), torch.ones(1, 3), {}, seed=10,
        conformer_count=1, bitseed_candidate_budget=2,
        sdpi_candidate_budget=1,
    )
    assert len(candidates) == 3
    assert metadata["actual_bitseed"] == 1
    assert metadata["actual_sdpi"] == 2
    assert metadata["partial_conformers"][0]["replacement_count"] == 1
    assert candidates[0]["candidate_id"] == ["bitseed", 0, 0, 0]


def test_fixed_candidate_initializer_clones_pose_and_records_source():
    candidate = {
        "candidate_id": ["sdpi", 7],
        "ligand": torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
        "poi": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
        "rotation": torch.eye(3).unsqueeze(0),
        "geometry": {"minimum_protein_ca_distance": 4.0},
        "acceptable": True,
    }
    initializer = FixedCandidateInitializer(candidate, no_torsion=False)
    ligand, poi, rotation, metadata = initializer.initialize(None, None, None, seed=91)
    assert torch.equal(ligand, candidate["ligand"])
    assert torch.equal(poi, candidate["poi"])
    assert torch.equal(rotation, candidate["rotation"])
    assert ligand.data_ptr() != candidate["ligand"].data_ptr()
    assert poi.data_ptr() != candidate["poi"].data_ptr()
    assert metadata["candidate_source"] == "sdpi"
    assert metadata["native_information_used"] is False
    assert metadata["diffusion_seed"] == 91


# Initialization audit

def test_conditional_gaussian_kl():
    assert conditional_translation_kl([0, 0, 0], [0, 0, 0], 2, 2) == 0
    assert math.isclose(conditional_translation_kl([2, 0, 0], [0, 0, 0], 2, 2), 0.5)
    expected = 0.5 * 3 * (4 - 1 - math.log(4))
    assert math.isclose(conditional_translation_kl([0, 0, 0], [0, 0, 0], 4, 2), expected)
    # Numeric expectation of log q - log p checks direction and dimension.
    rng = np.random.default_rng(7)
    x = rng.normal(size=(200000, 3)) * 2 + [1, 2, 3]
    empirical = np.mean(-3 * math.log(2 / 3)
                        - np.sum((x - [1, 2, 3]) ** 2, axis=1) / 8
                        + np.sum(x ** 2, axis=1) / 18)
    analytic = conditional_translation_kl([1, 2, 3], [0, 0, 0], 2, 3)
    assert abs(empirical - analytic) < 0.01


def test_invariant_pairs_and_rmsd_bound():
    from rdkit import Chem
    mol = Chem.MolFromSmiles('CCCC')
    pairs = invariant_atom_pairs(mol, [(1, 2)])
    assert set(map(tuple, pairs)) == {(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)}
    native = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [1., 1, 1]])
    moved = native.copy()
    moved[0, 0] -= 0.4
    gap = invariant_geometry_gap(native, moved, pairs)
    rmsd = np.sqrt(np.mean(np.sum((native - moved) ** 2, axis=1)))
    assert gap['exact_reachability_ruled_out_at_tolerance']
    assert 0 < gap['indexed_atom_rmsd_lower_bound_angstrom'] <= rmsd
    unchanged = invariant_geometry_gap(native, native + [10, -3, 2], pairs)
    assert not unchanged['exact_reachability_ruled_out_at_tolerance']
    # With no allowed torsions, all six pair distances are invariant.
    assert len(invariant_atom_pairs(mol, [])) == 6


def test_real_torsion_updates_preserve_audited_invariants():
    import torch
    from rdkit import Chem
    from tricoopdiff.initialization import validate_ligand_graph
    from tricoopdiff.diffdock.utils.torsion import modify_conformer_torsion_angles

    data = ligand_input('CCCOCC')
    mol = Chem.Mol(data['mol'])
    bonds, masks = validate_ligand_graph(mol, data['edge_index'], data['edge_mask'], data['mask_rotate'])
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    pairs = invariant_atom_pairs(mol, bonds.tolist())
    for seed in range(5):
        updates = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(bonds))
        moved = modify_conformer_torsion_angles(coords, bonds, masks, updates, as_numpy=False)
        # Vendored as_numpy=False returns float32, even from float64 input.
        gap = invariant_geometry_gap(coords.numpy(), moved.numpy(), pairs, atol=1e-6)
        assert not gap['exact_reachability_ruled_out_at_tolerance'], gap


def test_audit_rejects_nonfinite_and_bad_shapes():
    checks = [
        lambda: conditional_translation_kl([0, 0, 0], [0, 0, 0], 0, 1),
        lambda: conditional_translation_kl([0, 0, 0], [0, 0, 0], 1, float('nan')),
        lambda: conditional_translation_kl([float('nan'), 0, 0], [0, 0, 0], 1, 1),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((3, 3)), np.array([[0, 1]])),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((2, 3)), np.array([[0, 2]])),
        lambda: invariant_geometry_gap(np.zeros((2, 3)), np.zeros((2, 3)), np.array([[0., 1.]])),
    ]
    for check in checks:
        try:
            check()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid audit inputs must fail')


def test_binary_anchor_integrates_every_real_schedule_interval():
    """The last low-noise interval must not be merged into its predecessor."""
    from types import SimpleNamespace
    from tricoopdiff.initialization.candidates import DualExpertAnchorInitializer

    class ZeroExpert:
        model = SimpleNamespace(dynamic_max_cross=False, no_torsion=False)

        def __call__(self, graph):
            return torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(0)

    for steps in (2, 5, 20):
        for dtype in (torch.float32, torch.float64):
            coords = torch.zeros(1, 3, 3, dtype=dtype)
            receptor = {'coords': coords, 'features': torch.zeros(1, 3, 1)}
            ligand = {'features': torch.zeros(1, 3, 1),
                      'edge_index': None, 'edge_attr': None,
                      'edge_mask': None, 'mask_rotate': None}
            initializer = DualExpertAnchorInitializer(
                ZeroExpert(), receptor, receptor, binary_steps=steps)
            calls = []

            def record_step(position, tr, rot, tor, t, rt, tt, dt, rdt, tdt,
                            *, add_noise):
                calls.append((float(t), float(dt), add_noise, t.dtype, dt.dtype))
                return position + dt, position + dt

            initializer.process.reverse_step_protac = record_step
            crop = {'coords': coords[0], 'features': receptor['features'][0],
                    'edge_index': None}
            with patch('tricoopdiff.initialization.candidates.crop_receptor_to_ligand', return_value=crop), \
                 patch('tricoopdiff.initialization.candidates.build_subgraph_e3l', return_value=object()), \
                 patch('tricoopdiff.initialization.candidates._set_time_embeddings'):
                result = initializer._dock_rigid(receptor, ligand, coords, seed=73)
            schedule = initializer.process.get_inference_schedule(steps)['tr_schedule']
            assert len(calls) == steps, 'binary_steps must count actual expert updates'
            assert np.allclose([row[0] for row in calls], schedule[:-1])
            assert np.allclose([row[1] for row in calls], -np.diff(schedule))
            assert [row[2] for row in calls] == [True] * (steps - 1) + [False]
            assert all(row[3] == dtype and row[4] == dtype for row in calls)
            assert torch.allclose(result, torch.ones_like(result)), 'full schedule must integrate to t=0'


def test_binary_anchor_rejects_invalid_step_counts():
    from types import SimpleNamespace
    from tricoopdiff.initialization.candidates import DualExpertAnchorInitializer

    expert = SimpleNamespace(model=SimpleNamespace(no_torsion=False))
    for invalid in (0, 1, 2.5, True):
        try:
            DualExpertAnchorInitializer(expert, {}, {}, binary_steps=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid binary step count was accepted: {invalid!r}')


def test_reflect_minimization_excludes_topological_neighbours():
    from tricoopdiff.geometry.reflect import local_minimization
    coords = np.array([[0., 0., 0.], [1.54, 0., 0.], [2., 1., 0.], [3., 1., 0.]])
    result = local_minimization(coords, np.array([6] * 4), np.ones(4, dtype=bool),
                                max_rounds=1, bonds=[(0, 1), (1, 2), (2, 3)])
    assert np.array_equal(result, coords), 'repulsion moved atoms only separated by 1-3 bonds'


def test_reflect_rechecks_bond_anomalies_on_final_coordinates():
    from tricoopdiff.geometry.reflect import tricoop_reflect, detect_bond_anomalies
    coords = np.array([[0., 0., 0.], [2.30, 0., 0.], [0.5, 0., 0.]])
    numbers = np.array([6, 6, 6])
    result = tricoop_reflect(coords, numbers, bonds=[(0, 1)], bond_types=[1])
    expected = detect_bond_anomalies(result['coords'], [(0, 1)], [1], numbers)
    assert np.array_equal(result['bond_anomalies'], expected), 'returned pre-repair bond status'
    assert result['n_bond_anomalies'] == expected.sum()
    assert result['is_valid'] == (result['n_clashes'] == 0 and expected.sum() == 0)


def test_protac_training_split_keeps_disjoint_warheads_and_linker():
    from tricoopdiff.initialization.protac import split_training_warheads

    molecule = Chem.MolFromSmiles('CCCCCCCCCCCC')
    indices = np.arange(molecule.GetNumAtoms(), dtype=float)
    ligand = np.stack((1.5*indices, .6*(-1.)**indices, .4*np.sin(indices)), axis=1)
    protein1 = ligand[:3] + np.array([0., 0., 2.])
    protein2 = ligand[-3:] + np.array([0., 0., 2.])
    split = split_training_warheads(molecule, ligand, protein1, protein2,
                                    min_anchor_atoms=5)
    assert split is not None, 'two separated bound ends were rejected'
    first, second, linker = (set(split[key]) for key in
                             ('anchor1_atoms', 'anchor2_atoms', 'linker_atoms'))
    assert first.isdisjoint(second) and first.isdisjoint(linker) and second.isdisjoint(linker)
    assert first | second | linker == set(range(molecule.GetNumAtoms()))
    assert {0, 1, 2} <= first and {9, 10, 11} <= second
    assert len(linker) >= 2
    assert split_training_warheads(molecule, ligand, protein1, protein1,
                                   min_anchor_atoms=5) is None, \
        'single shared pocket was mislabeled as two warheads'


def test_protac_linker_closure_returns_the_selected_conformer():
    from rdkit.Chem import AllChem
    from tricoopdiff.initialization.protac import close_linker_to_binary_anchors

    ligand = Chem.MolFromSmiles('CCCCCCCCCC')
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 29
    assert AllChem.EmbedMolecule(ligand, parameters) == 0
    reference = ligand.GetConformer().GetPositions()
    candidates = close_linker_to_binary_anchors(
        ligand, [0, 1, 2], reference[[0, 1, 2]],
        [7, 8, 9], reference[[7, 8, 9]], starts=3, max_iterations=100)
    assert any(row['conformer_id'] != 0 for row in candidates)
    for row in candidates:
        assert row['ligand'].GetNumConformers() == 1
        assert np.allclose(row['ligand'].GetConformer().GetPositions(),
                           row['coordinates'], atol=1e-6), \
            'linker refinement must receive the conformer selected by closure'


def test_protac_binary_anchor_assembly_preserves_both_local_binding_poses():
    from tricoopdiff.geometry.rigid import apply_rigid, invert_rigid, rotation_exp
    from tricoopdiff.initialization.protac import assemble_from_binary_anchors

    dtype = torch.float64
    ligand = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [4., 0., 0.], [5., 0., 0.], [4., 1., 0.], [4., 0., 1.]],
                          dtype=dtype)
    protein2 = torch.tensor([[0., 3., 0.], [0., 4., 1.], [1., 3., 1.]], dtype=dtype)
    body_rotation = rotation_exp(torch.tensor([.2, -.3, .1], dtype=dtype))
    body_translation = torch.tensor([2., -4., 3.], dtype=dtype)
    inverse_rotation, inverse_translation = invert_rigid(body_rotation, body_translation)
    local_anchor2 = apply_rigid(ligand[4:], inverse_rotation, inverse_translation)
    global_rotation = rotation_exp(torch.tensor([-.4, .2, .3], dtype=dtype))
    conformer = apply_rigid(ligand, global_rotation,
                            torch.tensor([7., -5., 2.], dtype=dtype)).requires_grad_()
    result = assemble_from_binary_anchors(conformer, protein2, ligand[:4],
                                          local_anchor2, (0, 1, 2, 3), (4, 5, 6, 7))
    expected_protein2 = apply_rigid(protein2, body_rotation, body_translation)
    assert torch.allclose(result['ligand'], ligand, atol=1e-8)
    assert torch.allclose(result['protein2'], expected_protein2, atol=1e-8)
    assert torch.allclose(result['anchor_rmsd'], torch.zeros(2, dtype=dtype), atol=1e-8)
    result['protein2'].square().sum().backward()
    assert conformer.grad is not None and torch.isfinite(conformer.grad).all()
    assert conformer.grad.abs().sum() > 0, 'linker/warhead geometry cannot steer protein pose'


def test_protac_binary_mapping_keeps_all_native_blind_hypotheses():
    from tricoopdiff.initialization.protac import assemble_binary_mapping_hypotheses

    ligand = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [4., 0., 0.], [5., 0., 0.], [4., 1., 0.]])
    protein2 = torch.tensor([[4., 3., 0.], [5., 3., 0.], [4., 3., 1.]])
    condition = {'mapping_hypotheses': [
        {'anchor1_atoms': (0, 1, 2), 'anchor2_atoms': (3, 4, 5),
         'anchor1_xyz': ligand[:3].numpy(), 'anchor2_xyz': ligand[3:].numpy()},
        {'anchor1_atoms': (0, 1, 2), 'anchor2_atoms': (3, 4, 5),
         'anchor1_xyz': ligand[:3].numpy(), 'anchor2_xyz': ligand[3:].numpy()},
    ]}
    results = assemble_binary_mapping_hypotheses(ligand, protein2, condition)
    assert len(results) == 2, 'a symmetry-related mapping was silently discarded'
    assert results[0]['anchor_fit_rmsd'] <= results[1]['anchor_fit_rmsd']
    assert 'DockQ' not in results[0], 'oracle metric entered mapping feasibility ranking'


def test_protac_training_binary_condition_erases_native_relative_pose():
    from tricoopdiff.initialization.protac import (atom_order_graph_signature,
                                                   make_training_binary_condition)

    ligand = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [8., 0., 0.], [9., 0., 0.], [8., 1., 0.]])
    protein2 = torch.tensor([[8., 2., 0.], [9., 2., 0.], [8., 2., 1.]])
    molecule = Chem.MolFromSmiles('CCCCCC')
    split = {'contract': 'synthetic_binary_anchor_split_v1',
             'anchor1_atoms': (0, 1, 2), 'anchor2_atoms': (3, 4, 5),
             'atom_order_graph_sha256': atom_order_graph_signature(molecule)}
    condition = make_training_binary_condition(
        split, ligand, protein2, generator=torch.Generator().manual_seed(17))
    assert condition['native_full_ligand_conformer_is_input'] is False
    assert torch.allclose(condition['anchor1_xyz'], ligand[:3])
    assert torch.allclose(torch.cdist(condition['anchor2_xyz'],
                                      condition['protein2_reference']),
                          torch.cdist(ligand[3:], protein2), atol=1e-5)
    assert not torch.allclose(condition['anchor2_xyz'], ligand[3:]), \
        'native relative binary frames leaked into training input'


def test_protac_external_anchor_mapping_uses_graph_not_ternary_conformer():
    from tricoopdiff.initialization.protac import match_external_binary_anchors

    full = Chem.MolFromSmiles('NCCCOCCCl')
    fragments = [Chem.PathToSubmol(full, bond_path) for bond_path in ((0, 1), (5, 6))]
    for fragment, offset in zip(fragments, (0., 10.)):
        conformer = Chem.Conformer(fragment.GetNumAtoms())
        for atom, point in enumerate(((offset, 0., 0.), (offset+1., 0., 0.),
                                      (offset+1., 1., 0.))):
            conformer.SetAtomPosition(atom, point)
        fragment.AddConformer(conformer)
    before = match_external_binary_anchors(full, *fragments)
    assert before is not None, 'unique external warheads were not mapped'
    assert set(before['anchor1_atoms']).isdisjoint(before['anchor2_atoms'])
    assert before['native_ternary_coordinates_used'] is False
    assert len(before['mapping_hypotheses']) == before['symmetry_mapping_count']
    limited = match_external_binary_anchors(full, *fragments, hypothesis_limit=1)
    assert limited['symmetry_mapping_count'] == before['symmetry_mapping_count']
    assert len(limited['mapping_hypotheses']) == 1
    assert limited['attachment_atoms'] == before['attachment_atoms']
    assert limited['mapping_hypotheses'][0]['anchor1_atoms'] == \
        before['mapping_hypotheses'][0]['anchor1_atoms']
    assert limited['mapping_hypotheses'][0]['anchor2_atoms'] == \
        before['mapping_hypotheses'][0]['anchor2_atoms']
    projected = match_external_binary_anchors(
        full, *fragments, geometry_limit=2,
        geometry_fragment_atoms=(tuple(range(fragments[0].GetNumAtoms())),
                                 tuple(range(fragments[1].GetNumAtoms()))))
    assert projected['symmetry_mapping_count'] == before['symmetry_mapping_count']
    assert 1 <= len(projected['mapping_hypotheses']) <= 2
    assert all(len(row['anchor1_fragment_atoms']) == fragments[0].GetNumAtoms()
               for row in projected['mapping_hypotheses'])
    assert before['selected_mapping_policy'] == 'inspection_only_enumerate_for_inference'
    from tricoopdiff.initialization.protac import validate_anchor_graph_identity
    assert validate_anchor_graph_identity(before, full) == before['atom_order_graph_sha256']
    renumbered = Chem.RenumberAtoms(full, list(reversed(range(full.GetNumAtoms()))))
    try:
        validate_anchor_graph_identity(before, renumbered)
    except ValueError as error:
        assert 'atom indices' in str(error)
    else:
        raise AssertionError('chemically identical but differently ordered ligand was accepted')
    random_conformer = Chem.Conformer(full.GetNumAtoms())
    for atom in range(full.GetNumAtoms()):
        random_conformer.SetAtomPosition(atom, (atom*7., atom*3., -atom*5.))
    full.AddConformer(random_conformer)
    after = match_external_binary_anchors(full, *fragments)
    assert before['attachment_atoms'] == after['attachment_atoms']
    assert before['anchor1_atoms'] == after['anchor1_atoms']
    assert before['anchor2_atoms'] == after['anchor2_atoms']


def test_protac_bridge_ignores_unmapped_warhead_substituent():
    from tricoopdiff.initialization.protac import _bridge_attachment

    molecule = Chem.MolFromSmiles('CC(C)CCC')
    bridge = _bridge_attachment(molecule, (1, 2), (4, 5))
    assert bridge == (1, 4, frozenset({3})), \
        'a peripheral binary-warhead substituent invalidated the true linker'


def test_protac_linker_closure_holds_both_binary_poses_fixed():
    from rdkit.Chem import AllChem
    from tricoopdiff.initialization.protac import close_linker_to_binary_anchors

    molecule = Chem.MolFromSmiles('c1ccccc1CCc1ccccc1')
    molecule = Chem.AddHs(molecule)
    assert AllChem.EmbedMolecule(molecule, randomSeed=31) == 0
    target = molecule.GetConformer().GetPositions()
    heavy = Chem.RemoveHs(molecule).GetNumAtoms()
    first, second = list(range(6)), list(range(8, 14))
    candidates = close_linker_to_binary_anchors(
        molecule, first, target[first], second, target[second],
        seed=47, starts=2, max_iterations=100)
    assert candidates
    result = candidates[0]
    assert result['coordinates'].shape == (heavy, 3)
    assert result['anchor1_rmsd'] < 1e-6
    assert result['anchor2_rmsd'] < 1e-6
    assert np.isfinite(result['strain_energy'])
    assert np.isfinite(result['coordinates']).all()


def test_protac_linker_step_moves_protein_without_breaking_binary_anchors():
    from tricoopdiff.initialization.protac import anchored_flow_step, linker_torsion_mask

    ligand = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                            [4., 0., 0.], [5., 0., 0.], [4., 1., 0.], [4., 0., 1.]]],
                          dtype=torch.float64)
    protein2 = torch.tensor([[[4., 3., 0.], [5., 3., 1.], [4., 4., 1.]]],
                            dtype=torch.float64)
    data = {'coords': ligand, 'edge_index': torch.tensor([[3], [4]]),
            'edge_mask': torch.tensor([True]),
            'mask_rotate': torch.tensor([[False]*4+[True]*4])}
    condition = {'anchor1_atoms': (0, 1, 2, 3), 'anchor2_atoms': (4, 5, 6, 7),
                 'anchor1_xyz': ligand[0, :4], 'anchor2_xyz': ligand[0, 4:]}
    assert linker_torsion_mask(data, condition['anchor1_atoms'],
                               condition['anchor2_atoms']).tolist() == [True]
    speed = torch.tensor([[1.]], dtype=ligand.dtype, requires_grad=True)
    zero = torch.zeros((1, 3), dtype=ligand.dtype)
    velocities = {'e3_ligand': {'tr': zero, 'rot': zero},
                  'poi_ligand': {'tr': zero, 'rot': zero}, 'torsion': speed}
    moved_ligand, moved_protein, body, errors = anchored_flow_step(
        ligand, protein2, torch.eye(3, dtype=ligand.dtype)[None], velocities,
        torch.tensor([.1], dtype=ligand.dtype), data, protein2, condition)
    assert torch.allclose(moved_ligand[0, :4], condition['anchor1_xyz'], atol=1e-6)
    assert torch.allclose(errors, torch.zeros(2, dtype=ligand.dtype), atol=1e-6)
    assert not torch.allclose(moved_protein, protein2)
    assert torch.isfinite(body).all()
    moved_protein.square().sum().backward()
    assert speed.grad is not None and torch.isfinite(speed.grad).all()
    assert speed.grad.abs().sum() > 0, 'protein loss cannot train linker torsion'
