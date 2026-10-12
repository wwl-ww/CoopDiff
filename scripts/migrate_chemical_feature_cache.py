#!/usr/bin/env python
"""Rebuild ligand chemistry while reusing verified, unchanged protein features.

Writes a separate v10 cache; never modifies the v9 cache, labels or old banks.
No encoder or expert weights are loaded and no GPU is required.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
               str(Path(__file__).resolve().parents[1])]
import featurize_common as features  # installs diffdock compatibility first
import numpy as np
import torch
import yaml
from rdkit import Chem
from torch_geometric.data import Batch, HeteroData

from tricoopdiff.diffdock.datasets.process_mols import get_lig_graph_with_matching, read_molecule
from tricoopdiff.initialization import validate_ligand_graph
from tricoopdiff.models.expert import FrozenMatcha
from tricoopdiff.runtime import atomic_torch_save


def migrate(old_index, new_index, old_cache, out_cache, score_model_args):
    def entries(path):
        data = json.loads(Path(path).read_text())
        return data['entries'] if isinstance(data, dict) else data
    original = {row['case']: row for row in entries(old_index)}
    out_cache = Path(out_cache).resolve()
    if out_cache == Path(old_cache).resolve():
        raise ValueError('chemistry migration requires a separate output cache')
    out_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries(new_index):
        case = entry['case']
        source = original[case]
        if [source[k] for k in ('protein1', 'protein2')] != [entry[k] for k in ('protein1', 'protein2')]:
            raise ValueError(f'{case}: protein inputs changed')
        cached = torch.load(Path(old_cache) / (case + '.pt'), map_location='cpu', weights_only=False)
        identity = cached.get('esm_encoder_identity')
        if cached.get('cache_version') != 9 or not features._esm_identity_verified(identity):
            raise ValueError(f'{case}: migration requires a verified v9 source cache')
        old_signature = features._featurize_signature(
            source['protein1'], source['protein2'], source['ligand'], score_model_args,
            cache_version=9, encoder_identity=identity)
        if cached.get('source_signature') != old_signature:
            raise ValueError(f'{case}: old source identity mismatch')
        if features._source_cache_version(entry['ligand']) != 10:
            raise ValueError(f'{case}: corrected ligand must carry bond orders in SDF/MOL')
        mol = read_molecule(entry['ligand'], sanitize=True, remove_hs=score_model_args.remove_hs)
        old_mol = Chem.Mol(cached['ligand_data']['mol'])
        if ([a.GetAtomicNum() for a in mol.GetAtoms()] !=
                [a.GetAtomicNum() for a in old_mol.GetAtoms()]):
            raise ValueError(f'{case}: retained atom order/count changed')
        error = np.max(np.abs(mol.GetConformer().GetPositions() - old_mol.GetConformer().GetPositions()))
        if error > 0.00006:
            raise ValueError(f'{case}: ligand coordinates changed by {error} Angstrom')
        graph = HeteroData()
        get_lig_graph_with_matching(mol, graph, popsize=None, maxiter=None,
                                   matching=False, keep_original=False,
                                   num_conformers=1, remove_hs=score_model_args.remove_hs)
        graph.mol = mol
        ligand = features.extract_ligand(Batch.from_data_list([graph]), 'cpu')
        # The deposited atom order and coordinates were verified above. Keep
        # the exact old float32 labels in the common E3 frame.
        ligand['coords'] = cached['ligand_data']['coords'].unsqueeze(0).clone()
        actual = Chem.Mol(ligand['mol'])
        validate_ligand_graph(actual, ligand['edge_index'], ligand['edge_mask'], ligand['mask_rotate'])
        expert_bonds, expert_masks = FrozenMatcha.canonical_torsions(actual, ligand, 'cpu')
        assert torch.equal(expert_bonds, ligand['edge_index'][:, ligand['edge_mask']].T)
        assert np.array_equal(expert_masks.numpy(), ligand['mask_rotate'])
        result = dict(cached, cache_version=10,
                      ligand_data=features._strip_batch_and_cpu(ligand),
                      source_signature=features._featurize_signature(
                          entry['protein1'], entry['protein2'], entry['ligand'], score_model_args,
                          encoder_identity=identity),
                      chemistry_contract=ligand['chemistry_contract'],
                      chemistry_migration={'source_signature_v9': old_signature,
                                           'max_coordinate_roundtrip_error_A': float(error)})
        atomic_torch_save(result, out_cache / (case + '.pt'))
        # Exercise the real cache-only path and its version/source/encoder checks.
        e3, poi, loaded, _ = features.featurize_complex(
            case, entry['protein1'], entry['protein2'], entry['ligand'], score_model_args,
            'cpu', cache_dir=out_cache, cache_only=True, cache_wait_seconds=0,
            expected_esm_encoder_identity=identity)
        assert torch.equal(e3['features'][0], cached['e3_data']['features'])
        assert torch.equal(poi['features'][0], cached['poi_data']['features'])
        assert torch.equal(loaded['coords'][0], cached['ligand_data']['coords'])
        rows.append({'case': case, 'torsions': int(loaded['edge_mask'].sum()),
                     'max_coordinate_roundtrip_error_A': float(error)})
    manifest = {'source_index': str(Path(old_index).resolve()),
                'corrected_index': str(Path(new_index).resolve()), 'rows': rows}
    manifest_path = out_cache / f'migration_manifest_{Path(old_index).stem}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-index', required=True)
    parser.add_argument('--new-index', required=True)
    parser.add_argument('--old-cache', required=True)
    parser.add_argument('--out-cache', required=True)
    parser.add_argument('--model-parameters', required=True)
    args = parser.parse_args()
    from argparse import Namespace
    score_args = Namespace(**yaml.full_load(Path(args.model_parameters).read_text()))
    print(migrate(args.old_index, args.new_index, args.old_cache, args.out_cache, score_args))


if __name__ == '__main__':
    main()
