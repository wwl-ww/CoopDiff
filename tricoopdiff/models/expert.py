"""
Frozen DiffDock two-body expert wrapper.

Wraps DiffDock's CGModel with all parameters frozen. Production ternary models
read independently computed node features from E3-L and POI-L. Legacy per-layer
FiLM orchestration remains available for historical checkpoint implementations.

Each subgraph contains one protein and ligand, preserving DiffDock's input
schema. This does not guarantee that generated poses are in-distribution.
"""

import warnings
import math
import os
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import numpy as np
from e3nn import o3

# Vendored DiffDock so3/torus score norms — single source of truth so the
# FiLM-path score computation matches native CGModel.forward() exactly
# (required for zero_film_verification to pass).
from tricoopdiff.diffdock.utils import so3, torus
from tricoopdiff.runtime import sha256_file
from tricoopdiff.geometry.equivariant_clamp import clamp_irrep_features, validate_clamp_mode


EXPERT_SIGMA_BOUNDS = ((0.1, 19.0), (0.03, 1.55), (0.0314, 3.14))


def configure_expert_policy(expert, limit, mode):
    """Own numerical stabilization on the expert instance, never process state."""
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('expert activation clamp must be finite and positive')
    mode = validate_clamp_mode(mode)
    if getattr(expert, 'backend', None) in ('matcha', 'fabind'):
        return  # Transformer expert has no DiffDock activation-clamp contract.
    model = getattr(expert, 'model', expert)
    model.activation_clamp = float(limit)
    model.activation_clamp_mode = mode


def expert_runtime_identity(expert):
    """Generation identity includes configuration and actual numerical policy."""
    if getattr(expert, 'backend', None) in ('matcha', 'fabind'):
        return dict(expert.provenance)
    model = getattr(expert, 'model', None)
    identity = {**expert_base_identity(expert),
                'model_parameters_sha256': getattr(expert, 'provenance', {}).get('model_parameters_sha256'),
                'effective_model_config_sha256': getattr(expert, 'provenance', {}).get('effective_model_config_sha256'),
                'activation_clamp': getattr(model, 'activation_clamp', None),
                'activation_clamp_mode': getattr(model, 'activation_clamp_mode', 'componentwise'),
                'sigma_bounds': [list(pair) for pair in EXPERT_SIGMA_BOUNDS]}
    # Preserve the historical identity schema for the pretrained hard-edge
    # policy; record the opt-in runtime policy only when it is active.
    if getattr(model, 'smooth_edges', False):
        identity['smooth_edges'] = True
    if getattr(model, 'smooth_cross_edges', False):
        identity['smooth_cross_edges'] = True
    return identity


def validate_expert_sigma_args(args):
    """Reject configurations that disagree with the production preconditioner."""
    for name, bounds in zip(('tr', 'rot', 'tor'), EXPERT_SIGMA_BOUNDS):
        for suffix, expected in zip(('min', 'max'), bounds):
            value = getattr(args, f'{name}_sigma_{suffix}', None)
            if value != expected:
                raise ValueError(f'incompatible expert {name}_sigma_{suffix}: {value}; expected {expected}')


def expert_t_to_sigma(t_tr, t_rot, t_tor):
    """DiffDock-L input preconditioning, not a ternary diffusion process."""
    bounds = EXPERT_SIGMA_BOUNDS
    return tuple(low * (high / low) ** time
                 for time, (low, high) in zip((t_tr, t_rot, t_tor), bounds))


def so3_score_norm(sigma):
    """Reuse the expert's SO(3) table; keep native float32 output units."""
    value = torch.as_tensor(sigma)
    return so3.score_norm(value.detach().cpu().reshape(-1)).reshape(value.shape).to(value.device)


def torus_score_norm(sigma):
    """Torsion score norm — matches DiffDock's torus.score_norm lookup table.

    NOT 1/sigma^2: that approximation holds only for small sigma and diverges
    from the wrapped-Gaussian score norm as sigma approaches pi (relevant at
    high noise during diffusion, where tor_sigma reaches 1.55 rad).
    """
    return torus.score_norm(np.asarray(sigma))


class FrozenMatcha(nn.Module):
    """Frozen staged binary flow in a protein-defined, differentiable rigid frame.

    Stages supply evidence to the ternary controller, not a second sampler.
    No angular reinitialization or DiffDock score preconditioner is applied.
    """

    backend = 'matcha'
    crop_policy = 'full_protein_anchor_frame_v1'
    node_scalar_dim = 320
    node_vector_dim = 3
    no_torsion = False

    def __init__(self, source_dir, weights_dir, device='cuda'):
        super().__init__()
        import inspect
        import sys
        source_dir, weights_dir = Path(source_dir).resolve(), Path(weights_dir).resolve()
        sys.path.insert(0, str(source_dir))
        from matcha.models import MatchaModel
        from safetensors.torch import load_file
        if Path(inspect.getfile(MatchaModel)).resolve() != source_dir / 'matcha/models/models.py':
            raise ValueError('imported Matcha source differs from the requested frozen-prior source')
        conf = SimpleNamespace(num_heads=8, feature_dim=320, num_transformer_blocks=12,
            llm_emb_dim=480, use_time=True, dropout_rate=0., num_kernel_pos_encoder=128,
            predict_torsion_angles=True)
        self.weights_dir = weights_dir
        self.stages = nn.ModuleList()
        hashes = []
        for stage in (1, 2, 3):
            model = MatchaModel(conf, use_qk_bias=stage != 2)
            path = weights_dir / f'matcha_pipeline/stage{stage}/model.safetensors'
            model.load_state_dict(load_file(str(path)), strict=True)
            self.stages.append(model)
            hashes.append(sha256_file(path))
        self.provenance = {'backend': 'matcha', 'stage_weights_sha256': hashes,
            'coordinate_contract': 'protein_fixed_anchor_frame_v1',
            'time_contract': 'reverse_global_to_three_equal_local_forward_phases_v1',
            'advice_contract': 'forward_velocity_negative_reverse_clock_tr_div10_v1',
            'torsion_contract': 'canonical_executor_bonds_and_fragments_v2',
            'source_model_sha256': sha256_file(source_dir / 'matcha/models/models.py')}
        # Node evidence depends on all imported model/preprocessing code and on
        # the sequence encoder, not only on the three motion-head checkpoints.
        source_files = sorted((source_dir / 'matcha').rglob('*.py'))
        self.provenance['source_tree_sha256'] = hashlib.sha256(json.dumps([
            (str(path.relative_to(source_dir)), sha256_file(path)) for path in source_files
        ]).encode()).hexdigest()
        esm_files = sorted(path for path in (weights_dir / 'esm2_t12_35M').iterdir()
                           if path.suffix in ('.json', '.txt', '.safetensors', '.bin'))
        self.provenance['sequence_encoder_sha256'] = hashlib.sha256(json.dumps([
            (path.name, sha256_file(path)) for path in esm_files
        ]).encode()).hexdigest()
        self.provenance['adapter_sha256'] = sha256_file(__file__)
        self.requires_grad_(False).to(device).eval()

    @property
    def model(self):
        # Compatibility for existing trainer's no_torsion inspection only.
        return self

    def train(self, mode=True):
        super().train(False)
        return self

    def prepare_case(self, e3_data, poi_data, ligand_data, protein_paths):
        """Add immutable chemical/sequence features; never store target positions."""
        import numpy as np
        from Bio.PDB import PDBParser
        from Bio.SeqUtils import seq1
        from rdkit import Chem
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from matcha.utils.preprocessing import allowable_features, lig_feature_dims

        if len(protein_paths) != 2:
            raise ValueError('Matcha requires exactly two indexed protein paths')

        device = next(self.parameters()).device
        esm_path = str(self.weights_dir / 'esm2_t12_35M')
        tokenizer = AutoTokenizer.from_pretrained(esm_path, local_files_only=True)
        esm = AutoModelForMaskedLM.from_pretrained(esm_path, local_files_only=True).to(device).eval()
        aa = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        proteins = []
        for data, path in zip((e3_data, poi_data), protein_paths):
            structure = PDBParser(QUIET=True).get_structure('p', str(path))
            ca, embeddings, sequence = [], [], ''
            for chain in structure[0]:
                residues = [r for r in chain if r.id[0] == ' ' and 'CA' in r]
                if not residues:
                    continue
                seq = ''.join(seq1(r.resname, custom_map={'MSE': 'M'}) for r in residues)
                with torch.no_grad():
                    tokens = tokenizer(seq, return_tensors='pt').to(device)
                    embeddings.append(esm(**tokens, output_hidden_states=True).hidden_states[-1][0, 1:-1])
                sequence += seq
                ca.extend(r['CA'].coord for r in residues)
            positions = data['coords'].reshape(-1, 3).to(device)
            reference = torch.as_tensor(np.asarray(ca), device=device)
            if reference.shape != positions.shape:
                raise ValueError('Matcha protein sequence/coordinate correspondence failed')
            difference = reference - positions
            if (difference - difference.mean(0)).abs().max() > 1e-3:
                raise ValueError('Matcha preparation requires the indexed protein reference frame')
            centered = positions - positions.mean(0)
            first = int(centered.square().sum(-1).argmax())
            axis = centered[first] / centered[first].norm()
            perpendicular = centered - (centered @ axis)[:, None] * axis
            second = int(perpendicular.square().sum(-1).argmax())
            if perpendicular[second].norm() < 1e-3:
                raise ValueError('protein cannot define a nondegenerate rigid reference frame')
            proteins.append({**data, 'matcha': {
                'features': torch.cat(embeddings).unsqueeze(0).detach(),
                'sequence': torch.tensor([[aa.index(a) if a in aa else 0 for a in sequence]],
                                         device=device).unsqueeze(-1),
                'anchors': (first, second)}})
        del esm
        mol = ligand_data['mol']
        mol = Chem.Mol(mol) if isinstance(mol, bytes) else Chem.Mol(mol)
        keys = ('possible_atomic_num_list', 'possible_chirality_list', 'possible_degree_list',
                'possible_formal_charge_list', 'possible_implicit_valence_list', 'possible_numH_list',
                'possible_number_radical_e_list', 'possible_hybridization_list', 'possible_is_aromatic_list',
                'possible_numring_list', *[f'possible_is_in_ring{i}_list' for i in range(3, 9)])
        features = []
        for atom in mol.GetAtoms():
            ring = mol.GetRingInfo()
            values = (atom.GetAtomicNum(), str(atom.GetChiralTag()), atom.GetTotalDegree(),
                atom.GetFormalCharge(), atom.GetImplicitValence(), atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(), str(atom.GetHybridization()), atom.GetIsAromatic(),
                ring.NumAtomRings(atom.GetIdx()), *[ring.IsAtomInRingOfSize(atom.GetIdx(), i) for i in range(3, 9)])
            features.append([allowable_features[k].index(v) + 1 if v in allowable_features[k]
                             else len(allowable_features[k]) for k, v in zip(keys, values)])
        assert len(features[0]) == len(lig_feature_dims[0])
        bonds, masks = self.canonical_torsions(mol, ligand_data, device)
        ligand = {**ligand_data, 'matcha': {
            'features': torch.tensor(features, device=device).unsqueeze(0),
            'bonds': bonds, 'masks': masks}}
        return *proteins, ligand

    @staticmethod
    def canonical_torsions(mol, ligand_data, device):
        """Query the learned head on the executor's actual directed fragments.

        This is a prior read, not the official Matcha sampler. Its filtered or
        collinearity-adjusted rotor masks are not interchangeable with ours;
        neither bond reversal alone nor zero padding is a valid conversion.
        The head accepts arbitrary counts, including zero canonical rotors.
        """
        from tricoopdiff.initialization import validate_ligand_graph

        bonds, masks = validate_ligand_graph(mol, ligand_data['edge_index'],
            ligand_data['edge_mask'], ligand_data['mask_rotate'])
        return (bonds.to(device=device, dtype=torch.long),
                torch.as_tensor(masks, device=device, dtype=torch.bool))

    def read_pair(self, receptor, ligand, reverse_t, modulation=None):
        """Return world-frame motion and invariant tokens, ligand then protein."""
        rp, lp = receptor['coords'][0], ligand['coords'][0]
        data, chem = receptor['matcha'], ligand['matcha']
        origin = rp.mean(0)
        centered = rp - origin
        first, second = data['anchors']
        e1 = centered[first] / centered[first].norm()
        e2 = centered[second] - (centered[second] @ e1) * e1
        e2 = e2 / e2.norm()
        frame = torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)
        progress = (1 - reverse_t).clamp(0, 1) * 3
        index = min(2, int(progress.detach().item()))
        local_time = (progress - index).clamp(0, 1)
        batch = SimpleNamespace(protein=SimpleNamespace(
            pos=((rp - origin) @ frame).unsqueeze(0), x=data['features'], seq=data['sequence'],
            is_padded_mask=torch.zeros(1, len(rp), device=rp.device, dtype=torch.bool)),
            ligand=SimpleNamespace(pos=((lp - origin) @ frame).unsqueeze(0), x=chem['features'],
                t=local_time, num_atoms=torch.tensor([len(lp)], device=rp.device),
                is_padded_mask=torch.zeros(1, len(lp), device=rp.device, dtype=torch.bool),
                num_rotatable_bonds=torch.tensor([len(chem['bonds'])], device=rp.device),
                rotatable_bonds=chem['bonds'], mask_rotate=[chem['masks']],
                tor_ptr=torch.tensor([0, len(chem['bonds'])], device=rp.device)))
        model = self.stages[index]
        hook = None
        if modulation is not None:
            gamma, beta = modulation
            def film(module, args, output):
                return output * (1 + gamma) + beta
            hook = model.protein_embedder.register_forward_hook(film)
        try:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                lig_tokens, prot_tokens, tor = model.encode_complex(batch, predict_torsion=True)
                tr, rot = model.tr_head(lig_tokens[:, 0]), model.rot_head(lig_tokens[:, 1])
        finally:
            if hook is not None:
                hook.remove()
        scalar = torch.cat((lig_tokens[0, 2:], prot_tokens[0]), 0)
        # Tokens are invariant in the protein frame. Preserve direction through
        # physical atom offsets rather than inventing vector irreps from scalars.
        nodes = torch.cat((lp, rp), 0)
        vectors = torch.stack(((nodes - origin) / 10.,
                               (nodes - lp.mean(0)) / 10.,
                               frame[:, 0].expand(len(nodes), -1)), 1)
        tr, rot = tr @ frame.T, rot @ frame.T
        tor = tor if tor is not None else lp.new_empty(0)
        # Evidence convention matches the reverse-clock controller, not scores.
        return -tr / 10., -rot, -tor.reshape(1, -1), scalar, vectors


class FrozenFABind(nn.Module):
    """FABind+ node/pair prior read at the actual pose, not its docking sampler.

    Predict a pocket without target coordinates, then evaluate the pretrained
    refinement network on the current ligand. Atom suggestions are projected
    onto rigid/torsion tangents; only SDPI executes motions. Hard edge/pocket
    membership is piecewise constant, not a globally smooth expert contract.
    """

    backend = 'fabind'
    crop_policy = 'predicted_pocket_union_current_contacts_full_protein_evidence_v1'
    node_scalar_dim = 512
    node_vector_dim = 3
    no_torsion = False

    def __init__(self, source_dir, weights_dir, device='cpu'):
        super().__init__()
        import inspect
        import sys
        source_dir, weights_dir = Path(source_dir).resolve(), Path(weights_dir).resolve()
        root = source_dir / 'FABind_plus/fabind'
        sys.path.insert(0, str(root))
        from models.model import FABindPlus
        from utils.parsing import parse_train_args
        if Path(inspect.getfile(FABindPlus)).resolve() != root / 'models/model.py':
            raise ValueError('FABind import does not match requested source')
        _, parser = parse_train_args(test=True)
        flags = ('--n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 '
                 '--rm-layernorm --add-attn-pair-bias --explicit-pair-embed '
                 '--add-cross-attn-layer --random-n-iter --use-ln-mlp '
                 '--mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand '
                 '--dropout 0.1 --use-esm2-feat --dis-map-thres 15 '
                 '--pocket-radius-buffer 5 --min-pocket-radius 20')
        args = parser.parse_args(flags.split())
        self.prior = FABindPlus(args, embedding_channels=512, pocket_pred_embedding_channels=128)
        path = weights_dir / 'fabind_plus_best_ckpt.bin'
        self.prior.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        self.provenance = {
            'backend': self.backend, 'weights_sha256': sha256_file(path),
            'source_tree_sha256': hashlib.sha256(json.dumps([
                (str(p.relative_to(root)), sha256_file(p)) for p in sorted(root.rglob('*.py'))
            ]).encode()).hexdigest(),
            'coordinate_contract': 'protein_fixed_anchor_frame_current_pose_functional_refinement_v2',
            'advice_contract': 'ridge_horizontal_rigid_torsion_tangent_reverse_clock_v1',
            'topology_contract': 'native_hard_radius_and_predicted_pocket_piecewise_constant',
            'sequence_contract': 'caller_cached_esm2_t33_650M_last_hidden_1280',
            'time_contract': 'time_independent_regression_advice_not_a_flow_velocity',
            'adapter_sha256': sha256_file(__file__),
        }
        self.requires_grad_(False).to(device).eval()

    @property
    def model(self):
        return self

    def train(self, mode=True):
        super().train(False)
        return self

    def prepare_case(self, e3_data, poi_data, ligand_data, protein_paths=None):
        """Prepare indexed chemistry; ESM must be supplied, never downloaded."""
        from rdkit import Chem
        from tricoopdiff.initialization import validate_ligand_graph
        device = next(self.parameters()).device
        proteins = []
        for data in (e3_data, poi_data):
            features = data.get('esm_features')
            if features is None:
                raise ValueError('FABind requires explicit esm_features [N,1280]; do not guess feature slices')
            features = torch.as_tensor(features, device=device)
            if features.shape != (data['coords'].shape[1], 1280) or not torch.isfinite(features).all():
                raise ValueError('FABind ESM rows/dimension differ from protein coordinates')
            coords = data['coords'][0]
            centered = coords - coords.mean(0)
            first = int(centered.square().sum(-1).argmax())
            axis = centered[first] / centered[first].norm().clamp_min(1e-10)
            perpendicular = centered - (centered @ axis)[:, None] * axis
            second = int(perpendicular.square().sum(-1).argmax())
            if perpendicular[second].norm() < 1e-3:
                raise ValueError('FABind protein cannot define a nondegenerate rigid frame')
            proteins.append({**data, 'fabind': {'features': features.detach(),
                                              'anchors': (first, second)}})
        mol = Chem.Mol(ligand_data['mol'])
        bonds, masks = validate_ligand_graph(mol, ligand_data['edge_index'],
            ligand_data['edge_mask'], ligand_data['mask_rotate'])
        vocabularies = (['H', 'B', 'C', 'N', 'O', 'F', 'Mg', 'Si', 'P', 'S',
                         'Cl', 'Cu', 'Zn', 'Se', 'Br', 'Sn', 'I'],
                        list(range(7)), list(range(7)), list(range(8)), list(range(-5, 6)))
        features = []
        for atom in mol.GetAtoms():
            values = (atom.GetSymbol(), atom.GetDegree(), atom.GetTotalNumHs(),
                      atom.GetTotalValence(), atom.GetFormalCharge())
            row = []
            for value, vocabulary in zip(values, vocabularies):
                encoded = [0.] * (len(vocabulary) + 1)
                encoded[vocabulary.index(value) if value in vocabulary else -1] = 1.
                row.extend(encoded)
            features.append(row + [float(atom.GetIsAromatic())])
        adjacency = torch.as_tensor(Chem.GetAdjacencyMatrix(mol), device=device, dtype=torch.float32)
        las = (adjacency + adjacency @ adjacency) > 0
        for ring in Chem.GetSymmSSSR(mol):
            ids = torch.tensor(list(ring), device=device)
            las[ids[:, None], ids[None]] = True
        las.fill_diagonal_(False)
        ligand = {**ligand_data, 'fabind': {
            'features': torch.tensor(features, device=device),
            'edges': torch.tensor([(a, b) for bond in mol.GetBonds()
                for a, b in ((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                             (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))],
                device=device, dtype=torch.long).reshape(-1, 2).T + 1,
            'las': las.nonzero().T + 1,
            'bonds': bonds.to(device), 'masks': torch.as_tensor(masks, device=device),
        }}
        return *proteins, ligand

    def prepare_cached_case(self, e3_data, poi_data, ligand_data):
        """Consume the repository's indexed AA + ESM2-650M schema explicitly."""
        proteins = []
        for data in (e3_data, poi_data):
            features = data['features']
            if features.ndim != 3 or features.shape[0] != 1 or features.shape[-1] != 1281:
                raise ValueError('FABind cached inputs require B=1 AA+ESM650M features')
            proteins.append({**data, 'esm_features': features[0, :, 1:]})
        return self.prepare_case(*proteins, ligand_data)

    @staticmethod
    def _refine(module, coords, features, segment, mask, global_nodes, edges, las, reference):
        """Official refine_coord equations with functional, fully tracked updates.

        Native first iterations intentionally use no_grad; invoking that sampler
        would silently truncate our endpoint derivative. Its weights and formulas
        are retained here, but not its inference autograd policy.
        """
        from torch_scatter import scatter_add
        batch = segment.new_zeros(len(segment), dtype=torch.long)
        p, c = features[segment == 1][None], features[segment == 0][None]
        pair, pair_mask = module.inter_layer(p, c,
            torch.ones(p.shape[:2], device=p.device, dtype=torch.bool),
            torch.ones(c.shape[:2], device=c.device, dtype=torch.bool))
        pair = pair * pair_mask[..., None]
        def fixed_topology_step(coords, features, pair, ctx, inter, reduced):
            # Recompute only continuous tensor operations. Re-extracting hard
            # edges after CUDA scatter roundoff can select a different graph.
            gnn = module.gnn
            h, x = gnn.dropout(gnn.linear_in(features)), coords
            current_pair = pair
            for index in range(gnn.n_layers):
                h, moved = gnn._modules[f'gcl_{index}'](h, ctx, x, batch_id=batch)
                x = torch.where(mask[:, None, None], moved, x) if gnn.args.fix_pocket else moved
                h, moved, _, current_pair = gnn._modules[f'att_{index}'](
                    h, inter, x, segment_id=segment, batch_id=batch, reduced_tuple=reduced,
                    pair_embed_batched=current_pair, pair_mask=pair_mask,
                    LAS_mask=None, p_p_dist_embed=None, c_c_dist_embed=None)
                x = torch.where(mask[:, None, None], moved, x) if gnn.args.fix_pocket else moved
                if not gnn.args.rm_LAS_constrained_optim and not gnn.args.only_last_LAS:
                    for _ in range(gnn.geom_reg_steps):
                        delta = x[las[0], 0] - x[las[1], 0]
                        target = reference[las[0], 0] - reference[las[1], 0]
                        force = 4 * (delta.square().sum(-1) - target.square().sum(-1))[:, None] * delta
                        update = scatter_add(force, las[1], dim=0, dim_size=len(x))
                        x = x + (update * gnn.geometry_reg_step_size).clamp(
                            min=gnn.normalize_coord(-15), max=gnn.normalize_coord(15))[:, None]
            h, moved = gnn.out_layer(h, ctx, x, batch_id=batch)
            x = torch.where(mask[:, None, None], moved, x) if gnn.args.fix_pocket else moved
            if gnn.dense:
                raise ValueError('unsupported FABind dense configuration')
            output = gnn.linear_out(gnn.dropout(h))
            return torch.where(mask[:, None, None], x, coords), output, current_pair

        from torch.utils.checkpoint import checkpoint
        for _ in range(module.n_iter):
            ctx, inter, reduced = module.extract_edges(coords, batch, segment, global_nodes)
            ctx = torch.cat((edges, ctx), 1)
            args = (coords, features, pair, ctx, inter, reduced)
            if torch.is_grad_enabled():
                coords, output, current_pair = checkpoint(
                    fixed_topology_step, *args, use_reentrant=False)
            else:
                coords, output, current_pair = fixed_topology_step(*args)
        return coords, output, current_pair

    @staticmethod
    def project_motion(coords, displacement, bonds, masks):
        """Regularized joint fit in the executor's horizontal torsion gauge."""
        x = coords.double()
        centered = x - x.mean(0)
        eye = torch.eye(3, device=x.device, dtype=x.dtype)
        spin_basis = torch.linalg.cross(eye[:, None].expand(3, len(x), 3),
                                         centered[None].expand(3, -1, -1), dim=-1)
        columns = [eye[:, None].expand(3, len(x), 3), spin_basis]
        inertia = centered.square().sum() * eye - centered.T @ centered
        ridge = inertia.trace().clamp_min(torch.finfo(x.dtype).tiny) * 1e-6
        torsions = []
        for (u, v), mask in zip(bonds, masks):
            axis = x[u] - x[v]
            length = torch.linalg.vector_norm(axis)
            if bool(length <= 1e-10):
                raise ValueError('degenerate FABind executor torsion axis')
            axis = axis / length
            tangent = torch.linalg.cross(axis.expand_as(x), x - x[v], dim=-1) * mask[:, None]
            torque = torch.linalg.cross(centered, tangent, dim=-1).sum(0)
            spin = torch.linalg.solve(inertia + ridge * eye, torque + .5 * ridge * axis)
            tangent = tangent - tangent.mean(0) - torch.linalg.cross(spin.expand_as(x), centered, dim=-1)
            torsions.append(tangent)
        if torsions:
            columns.append(torch.stack(torsions))
        jacobian = torch.cat(columns).flatten(1).T
        gram = jacobian.T @ jacobian
        damping = gram.diagonal().mean().clamp_min(1e-8) * 1e-4
        action = torch.linalg.solve(gram + damping * torch.eye(len(gram), device=x.device, dtype=x.dtype),
                                    jacobian.T @ displacement.double().flatten()).to(coords.dtype)
        return action[:3][None], action[3:6][None], action[6:][None]

    def read_pair(self, receptor, ligand, reverse_t, modulation=None):
        rp, lp = receptor['coords'][0], ligand['coords'][0]
        # Official xyz-wise clamps break rotation equivariance. Keep the exact
        # frozen equations in a protein-defined frame, not an arbitrary world
        # frame. Anchor indices are chosen once, without ligand/target input.
        origin = rp.mean(0)
        centered = rp - origin
        first, second = receptor['fabind']['anchors']
        e1 = centered[first] / centered[first].norm()
        e2 = centered[second] - (centered[second] @ e1) * e1
        e2 = e2 / e2.norm()
        frame = torch.stack((e1, e2, torch.linalg.cross(e1, e2)), -1)
        rp, lp = (rp - origin) @ frame, (lp - origin) @ frame
        chem, model = ligand['fabind'], self.prior
        lig_h = model.compound_linear_whole_protein(chem['features'].to(lp))
        prot_h = model.protein_linear_whole_protein(receptor['fabind']['features'].to(rp))
        if modulation is not None:
            gamma, beta = modulation
            if gamma.ndim == 3 and gamma.shape[0] == 1:
                gamma, beta = gamma[0], beta[0]
            if gamma.shape != prot_h.shape or beta.shape != prot_h.shape:
                raise ValueError('FABind requires node-aligned protein FiLM, not broadcast context')
            prot_h = prot_h * (1 + gamma) + beta
        n = len(lp)
        def run(module, prot_coords, ligand_coords, protein_h, ligand_h):
            origin = prot_coords.mean(0, keepdim=True)
            coords = torch.cat((origin, ligand_coords,
                                origin, prot_coords)) / model.coordinate_scale
            reference = torch.cat((lp.new_zeros(1, 3), lp - lp.mean(0),
                                    lp.new_zeros(1 + len(prot_coords), 3))) / model.coordinate_scale
            h = torch.cat((model.glb_c, ligand_h, model.glb_p, protein_h))
            if module is model.pocket_pred_model:
                h = model.embedding_shrink(h)
            segment = torch.zeros(len(h), dtype=torch.long, device=lp.device)
            segment[n + 1:] = 1
            global_nodes = torch.zeros(len(h), dtype=torch.bool, device=lp.device)
            global_nodes[0] = global_nodes[n + 1] = True
            mask = segment == 0
            mask[n + 1] = True
            return self._refine(module, coords[:, None], h, segment, mask,
                                global_nodes, chem['edges'], chem['las'], reference[:, None])
        _, whole, _ = run(model.pocket_pred_model, rp, lp - lp.mean(0) + rp.mean(0), prot_h, lig_h)
        whole = model.embedding_enlarge(whole)
        proteins, ligands = whole[n + 2:], whole[1:n + 1]
        probability = model.protein_to_pocket(proteins).sigmoid().squeeze(-1)
        # Official inference defaults gs_hard=False, gs_tau=1: deterministic
        # two-class softmax(log probabilities) gives these soft weights.
        probability = probability.clamp(1e-6, 1 - 1e-6)
        center = (probability[:, None] * rp).sum(0) / probability.sum().clamp_min(1e-6)
        radius = (model.pocket_radius_head(ligands.sum(0, keepdim=True)).relu().squeeze() + 5).clamp_min(20)
        keep = self.pocket_membership(rp, lp, center, radius, model.args.inter_cutoff)
        if int(keep.sum()) < 5:
            keep = torch.ones_like(keep)
        predicted, local, _ = run(model.complex_model, rp[keep], lp, proteins[keep], ligands)
        displacement = predicted[1:n + 1, 0] * model.coordinate_scale - lp
        tr, rot, tor = self.project_motion(lp, displacement, chem['bonds'], chem['masks'])
        proteins = proteins.index_copy(0, keep.nonzero().flatten(), local[n + 2:])
        scalar = torch.cat((local[1:n + 1], proteins))
        nodes = torch.cat((lp, rp))
        suggestion = torch.cat((displacement, torch.zeros_like(rp)))
        vectors = torch.stack(((nodes - rp.mean(0)) / 10,
                               (nodes - lp.mean(0)) / 10, suggestion / 10), 1)
        return -tr @ frame.T / 10, -rot @ frame.T, -tor, scalar, vectors @ frame.T

    @staticmethod
    def pocket_membership(protein, ligand, center, radius, contact_cutoff):
        """A long ligand must retain every currently nearby protein residue.

        Predicted pocket alone can omit the far end of a PROTAC. Full-protein
        evidence is retained regardless, and local refinement additionally sees
        the union of predicted pocket and actual native-cutoff contacts.
        """
        predicted = (protein - center).norm(dim=-1) <= radius
        nearby = torch.cdist(protein, ligand,
                             compute_mode='donot_use_mm_for_euclid_dist').amin(-1) <= contact_cutoff
        return (predicted | nearby).detach()


class FrozenDiffDock(nn.Module):
    """
    Frozen weights with differentiable scores and SO(3)-invariant node features.

    ``forward_features`` reads native final-layer scalars without changing the
    expert computation. Per-layer FiLM APIs are retained for legacy models.
    """

    def __init__(self, cg_model: nn.Module, num_conv_layers: int = 11):
        """
        Args:
            cg_model: DiffDock's CGModel with pretrained weights loaded
            num_conv_layers: number of conv layers (11 for DiffDock-L, NOT 6)
        """
        super().__init__()
        self.model = cg_model
        # Snapshot legacy environment defaults only once. Production owners
        # subsequently configure this instance explicitly.
        self.model.activation_clamp = (float(os.environ['NAN_CLAMP'])
                                       if os.environ.get('NAN_CLAMP') else None)
        self.model.activation_clamp_mode = validate_clamp_mode(
            os.environ.get('NAN_CLAMP_MODE', 'componentwise'))
        self.num_conv_layers = num_conv_layers

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Store model references for convenience
        self.ns = self.model.ns
        self.nv = self.model.nv
        self.sh_irreps = self.model.sh_irreps
        self.device = self.model.device
        self.eval()

    def embedding(self, data):
        """
        Run the embedding phase of CGModel.
        Returns initial ligand and receptor node features plus edge info.

        This is a thin wrapper around CGModel.embedding().
        """
        return self.model.embedding(data)

    def train(self, mode: bool = True):
        """Keep the frozen expert in evaluation mode.

        ``requires_grad=False`` prevents parameter updates but does not disable
        dropout or other training-mode behavior. Parent ``model.train()`` calls
        recurse into children, so override it here to keep the pretrained expert
        deterministic while still allowing TriFuse to train.
        """
        super().train(False)
        self.model.eval()
        return self

    def build_cross_graph(self, data, cross_cutoff):
        """Build cross edges between ligand and receptor."""
        return self.model.build_cross_conv_graph(data, cross_cutoff)

    def build_center_graph(self, data):
        """Build center convolution graph for tr/rot score prediction."""
        return self.model.build_center_conv_graph(data)

    def build_bond_graph(self, data):
        """Build bond convolution graph for torsion score prediction."""
        return self.model.build_bond_conv_graph(data)

    def step_conv_layer(
        self,
        layer_idx: int,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr,
        edge_sh: torch.Tensor,
        edge_weight=None,
    ) -> torch.Tensor:
        """
        Run a single conv layer. This is the core per-layer interface
        that allows FiLM modulation between layers.

        Args:
            layer_idx: which conv layer to run (0-indexed)
            node_attr: [N_total, D] current node features (ligand + receptor)
            edge_index: [2, E] combined edge index
            edge_attr: edge features (tensor or list of tensors for multi-group)
            edge_sh: spherical harmonic edge embeddings
            edge_weight: optional edge weights

        Returns:
            updated node_attr [N_total, D']
        """
        return self.model.conv_layers[layer_idx](
            node_attr, edge_index, edge_attr, edge_sh, edge_weight=edge_weight
        )

    def compute_scores(
        self,
        data,
        lig_node_attr: torch.Tensor,
        graph_sigma_emb: torch.Tensor,
        tr_sigma: torch.Tensor,
        rot_sigma: torch.Tensor,
        tor_sigma: torch.Tensor,
    ) -> dict:
        """
        Compute final tr/rot/torsion scores from ligand node features.

        Returns:
            dict with keys: 'tr_score', 'rot_score', 'tor_score'
        """
        model = self.model

        # Build center convolution for tr/rot scores
        center_edge_index, center_edge_attr, center_edge_sh = model.build_center_conv_graph(data)
        center_edge_attr = model.center_edge_embedding(center_edge_attr)
        if model.fixed_center_conv:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1
            )
        else:
            center_edge_attr = torch.cat(
                [center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1
            )
        global_pred = model.final_conv(
            lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh,
            out_nodes=data.num_graphs
        )

        tr_pred = global_pred[:, :3] + (
            global_pred[:, 6:9] if not model.odd_parity else 0
        )
        rot_pred = global_pred[:, 3:6] + (
            global_pred[:, 9:] if not model.odd_parity else 0
        )

        # Fix magnitude of tr/rot scores. The vendored CGModel uses the same
        # normalization pattern but does not guard zero norms; in TriCoopDiff that
        # can surface during smoke/training as NaNs in score loss / Tweedie phys
        # loss when a pathological graph makes final_conv emit an all-zero vector.
        # Keep the raw norm for the magnitude head (native semantics) but clamp the
        # divisor so direction normalization itself never divides by zero.
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_unit = tr_pred / tr_norm.clamp_min(1e-8)
        tr_pred = tr_unit * model.tr_final_layer(
            torch.cat([tr_norm, graph_sigma_emb], dim=1)
        )
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_unit = rot_pred / rot_norm.clamp_min(1e-8)
        rot_pred = rot_unit * model.rot_final_layer(
            torch.cat([rot_norm, graph_sigma_emb], dim=1)
        )

        if model.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = (
                rot_pred
                * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(lig_node_attr.device)
            )

        # Torsion scores
        if model.no_torsion or data['ligand'].edge_mask.sum() == 0:
            tor_pred = torch.empty(0, device=lig_node_attr.device)
        else:
            tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = \
                model.build_bond_conv_graph(data)
            tor_bond_vec = (
                data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
            )
            tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

            tor_bonds_sh = o3.spherical_harmonics(
                "2e", tor_bond_vec, normalize=True, normalization='component'
            )
            tor_edge_sh = model.final_tp_tor(
                tor_edge_sh, tor_bonds_sh[tor_edge_index[0]]
            )

            tor_edge_attr = torch.cat([
                tor_edge_attr,
                lig_node_attr[tor_edge_index[1], :self.ns],
                tor_bond_attr[tor_edge_index[0], :self.ns],
            ], -1)
            tor_pred = model.tor_bond_conv(
                lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                out_nodes=data['ligand'].edge_mask.sum(), reduce='mean',
                edge_weight=tor_edge_weight
            )
            tor_pred = model.tor_final_layer(tor_pred).squeeze(1)
            # Align with patched cg_model.py: index sigma by the source atoms of
            # the rotatable bonds (tor_bonds[0]), which is guaranteed to match
            # tor_pred's out_nodes, rather than the edge_index→edge_mask chain.
            edge_sigma = tor_sigma[data['ligand'].batch][tor_bonds[0]]

            if model.scale_by_sigma:
                tor_pred = tor_pred * torch.sqrt(
                    torch.tensor(
                        torus_score_norm(edge_sigma.cpu().numpy())
                    ).float().to(lig_node_attr.device)
                )

        return {
            'tr_score': tr_pred,
            'rot_score': rot_pred,
            'tor_score': tor_pred,
        }

    def forward(self, data):
        """Native score forward; feature reads and legacy checks use this path."""
        return self.model(data)

    @property
    def node_scalar_dim(self) -> int:
        """Number of final-layer 0e and 0o channels (proper-rotation scalars)."""
        irreps = o3.Irreps(self.model.conv_layers[-1].out_irreps)
        return sum(multiplicity for multiplicity, irrep in irreps if irrep.l == 0)

    @property
    def node_vector_dim(self) -> int:
        """Number of final-layer l=1 channels available at each expert node."""
        irreps = o3.Irreps(self.model.conv_layers[-1].out_irreps)
        return sum(multiplicity for multiplicity, irrep in irreps if irrep.l == 1)

    def forward_features(self, data, *, include_vectors=False):
        """Return native scores and scalar nodes in ligand/cropped-receptor order.

        Coordinates retain autograd through the frozen expert. The temporary
        read-only hook does not replace native forward or alter checkpoint keys.
        Odd scalars are SO(3) invariant, not invariant under reflections.
        When requested, the final-layer l=1 irreps are returned as [N, V, 3].
        A componentwise clamp is rejected for this path because it breaks
        rotational equivariance; the legacy scalar-only API keeps that mode.
        """
        layer = self.model.conv_layers[-1]
        irreps = o3.Irreps(layer.out_irreps)
        sections = [section for (_, irrep), section in zip(irreps, irreps.slices())
                    if irrep.l == 0]
        if not sections:
            raise ValueError('expert final layer has no scalar node channels')
        captured = []
        hook = layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
        try:
            tr, rot, tor = self.model(data)[:3]
        finally:
            hook.remove()
        if len(captured) != 1 or captured[0].ndim != 2 or captured[0].shape[1] != irreps.dim:
            raise ValueError('expert final-layer node output does not match its irreps')
        limit = self.model.activation_clamp
        clamp_mode = self.model.activation_clamp_mode
        if include_vectors and limit and clamp_mode == 'componentwise':
            raise ValueError('equivariant vector features require irrep_norm expert activation clamping')
        scalars = torch.cat([captured[0][:, section] for section in sections], dim=-1)
        vector_blocks = []
        for (multiplicity, irrep), section in zip(irreps, irreps.slices()):
            if include_vectors and irrep.l == 1:
                block = captured[0][:, section].reshape(-1, multiplicity, 3)
                vector_blocks.append((multiplicity, irrep, block))
        # Native CGModel applies its clamp after the hook. Scalar norm clipping
        # is mathematically equivalent to component clipping, but its floating
        # point arithmetic and non-finite propagation must also match native.
        if limit:
            if clamp_mode == 'irrep_norm':
                scalars = clamp_irrep_features(scalars, f'{scalars.shape[-1]}x0e', float(limit))
            else:
                scalars = scalars.clamp(min=-float(limit), max=float(limit))
            if include_vectors:
                vector_blocks = [
                    (multiplicity, irrep,
                     clamp_irrep_features(block.reshape(len(block), -1),
                                           f"{multiplicity}x1{'o' if irrep.p == -1 else 'e'}", float(limit)
                                           ).reshape_as(block))
                    for multiplicity, irrep, block in vector_blocks]
        if include_vectors:
            vectors = (torch.cat([block for _, _, block in vector_blocks], dim=1)
                       if vector_blocks else scalars.new_zeros(len(scalars), 0, 3))
            return tr, rot, tor, scalars, vectors
        return tr, rot, tor, scalars


def _set_time_embeddings(data, t, model):
    """
    Set diffusion time information on the HeteroData object, matching DiffDock's
    `set_time` semantics: `complex_t` holds per-graph [batchsize] tensors and
    `ligand.node_t` holds per-node [num_nodes] tensors. CGModel's
    `timestep_emb_func` (GaussianFourierProjection) requires 1-D tensor inputs,
    so these must be tensors, not Python floats.
    """
    device = data['ligand'].pos.device
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=torch.float32)
    t = t.to(device).reshape(-1)  # [1] (single-complex mode)

    batchsize = int(data.num_graphs)
    num_nodes = data['ligand'].pos.shape[0]

    data.complex_t = {
        'tr': t.expand(batchsize),
        'rot': t.expand(batchsize),
        'tor': t.expand(batchsize),
    }
    data['ligand'].node_t = {
        'tr': t.expand(num_nodes),
        'rot': t.expand(num_nodes),
        'tor': t.expand(num_nodes),
    }
    if 'receptor' in data.node_types:
        num_rec_nodes = data['receptor'].pos.shape[0]
        data['receptor'].node_t = {
            'tr': t.expand(num_rec_nodes),
            'rot': t.expand(num_rec_nodes),
            'tor': t.expand(num_rec_nodes),
        }


def build_diffdock_expert(
    model_dir: str = "checkpoints/score_model",
    device: str = "cuda",
    ckpt: str = "best_ema_inference_epoch_model.pt",
    num_conv_layers: int = None,
    crop_beyond="native",
) -> FrozenDiffDock:
    """
    Build and freeze a DiffDock score model from pretrained weights.

    Args:
        model_dir: directory containing `model_parameters.yml` and the checkpoint.
        device: 'cuda' or 'cpu'.
        ckpt: checkpoint filename inside model_dir.
        num_conv_layers: optional assertion of the checkpoint architecture;
            must match model_parameters.yml (11 for DiffDock-L).
        crop_beyond: receptor-crop policy for the frozen expert. ``"native"``
            (default) preserves the checkpoint's `model_parameters.yml` setting so
            the expert stays on vendored DiffDock's native inference path.
            Pass ``None`` to force the old full-receptor fallback, or a numeric
            value to override the crop radius explicitly.

    Returns:
        FrozenDiffDock wrapping the loaded CGModel (all params frozen).
    """
    from argparse import Namespace
    from functools import partial
    from pathlib import Path

    import yaml
    from tricoopdiff.diffdock.factory import build_cg_model
    from tricoopdiff.diffdock.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl

    model_dir = Path(model_dir)
    with open(model_dir / "model_parameters.yml") as f:
        args = Namespace(**yaml.full_load(f))
    validate_expert_sigma_args(args)

    checkpoint_layers = getattr(args, "num_conv_layers", 11)
    if num_conv_layers is None:
        num_conv_layers = checkpoint_layers
    if (not isinstance(num_conv_layers, int) or isinstance(num_conv_layers, bool)
            or num_conv_layers < 1 or num_conv_layers != checkpoint_layers):
        raise ValueError('num_conv_layers must match the frozen checkpoint architecture')

    # Keep the frozen expert on the same crop/dynamic-cross regime as the
    # vendored DiffDock checkpoint unless the caller explicitly overrides it.
    # Previous versions forced ``args.crop_beyond = None`` here, which silently
    # downgraded native inference from
    #   crop_beyond(...) + dynamic_max_cross=True
    # to
    #   full receptor + dynamic_max_cross=False,
    # creating a persistent parity gap for the zero-FiLM baseline.
    if crop_beyond == "native":
        pass
    elif crop_beyond is None:
        args.crop_beyond = None
    else:
        args.crop_beyond = crop_beyond

    t_to_sigma = partial(t_to_sigma_compl, args=args)
    cg_model = build_cg_model(args, device, t_to_sigma)

    ckpt_path = model_dir / ckpt
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    cg_model.load_state_dict(state_dict, strict=True)
    cg_model = cg_model.to(device)
    cg_model.eval()

    expert = FrozenDiffDock(cg_model, num_conv_layers=num_conv_layers)
    expert.provenance = {
        "diffdock_commit": "47eb91a",
        "diffdock_source": "vendored with repository stability patches",
        "score_checkpoint": str(ckpt_path.resolve()),
        "score_checkpoint_sha256": sha256_file(ckpt_path),
        "model_parameters_sha256": sha256_file(model_dir / 'model_parameters.yml'),
        "effective_model_config_sha256": hashlib.sha256(
            json.dumps(vars(args), sort_keys=True).encode()).hexdigest(),
        "no_torsion": bool(getattr(args, "no_torsion", False)),
    }
    return expert


_EXPERT_IDENTITY_KEYS = ('diffdock_commit', 'score_checkpoint_sha256', 'no_torsion')


def expert_base_identity(expert):
    """Return the stable identity fields for the external frozen expert."""
    provenance = getattr(expert, 'provenance', {})
    if getattr(expert, 'backend', None) in ('matcha', 'fabind'):
        return dict(provenance)
    missing = [key for key in _EXPERT_IDENTITY_KEYS if key not in provenance]
    if missing:
        raise ValueError(f'frozen expert has incomplete provenance: {missing}')
    return {key: provenance[key] for key in _EXPERT_IDENTITY_KEYS}


def validate_expert_base_identity(checkpoint, expert):
    """Reject known base/adapter mismatches and label legacy uncertainty."""
    actual = expert_base_identity(expert)
    saved = checkpoint.get('expert_base_identity')
    if saved is None:
        warnings.warn(
            'Checkpoint has no frozen-expert identity; compatibility with the '
            'runtime DiffDock base cannot be proven.', RuntimeWarning, stacklevel=2)
        return 'legacy_unrecorded'
    if saved != actual:
        raise ValueError(f'checkpoint frozen expert identity {saved!r} does not '
                         f'match runtime {actual!r}')
    return saved
