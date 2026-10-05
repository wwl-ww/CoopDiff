"""Native-informed rigid/torsion reachability diagnostic; never prediction scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import train_entry as entry
from tricoopdiff.initialization import TernaryInitializer
from tricoopdiff.initialization.endpoints import (
    _fit_numpy_rigid, project_endpoint_coupling, endpoint_internal_geometry,
)
from tricoopdiff.sdpi_flow import _raw_torsions
from tricoopdiff.metrics.dockq import compute_fullatom_dockq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='1AFQ_B_C_0FG')
    parser.add_argument('--seeds', nargs='+', type=int, default=[73, 74, 10001])
    parser.add_argument('--restarts', type=int, default=6)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/model45_root_cause/reachability')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    item = next(x for x in entry._load_entries(ROOT / 'results/model45_partition/ternary_train.json') if x['case'] == args.case)
    score_args = entry._load_score_model_args(ROOT / 'checkpoints/score_model')
    ed, pd, ld, _ = entry.featurize_complex(args.case, item['protein1'], item['protein2'], item['ligand'], score_args, 'cpu', cache_dir=ROOT.parent / 'CoopDiff/results/featurize_cache_proposal_oracle', cache_read_only=True, cache_only=True)
    context = entry.validation_metric_context(item, ROOT / 'DeepTernary/data/TernaryDB/pdbs')
    native_ligand, native_poi = ld['coords'][0].numpy(), pd['coords'][0].numpy()
    bonds = ld['edge_index'].T[ld['edge_mask']].numpy()
    masks = ld['mask_rotate']
    if isinstance(masks, (list, tuple)):
        masks = masks[0]
    masks = np.asarray(masks)
    def score(ligand, poi):
        return compute_fullatom_dockq(np.asarray(ligand), np.asarray(poi), ed['coords'][0].numpy(), **context)
    reports = []
    for seed in args.seeds:
        ligand, poi, body, provenance = TernaryInitializer().initialize(ed['coords'][0], pd['coords'][0], ld, seed=seed)
        source = ligand[0].numpy().astype(np.float64)
        coupling, projected, projection = project_endpoint_coupling({**ld, 'coords': ligand}, poi, native_ligand, pd['coords'])
        def decode(angles):
            shape = _raw_torsions(source, bonds, masks, angles)
            rotation, translation, _, _ = _fit_numpy_rigid(shape, native_ligand)
            return shape @ rotation.T + translation
        def residual(angles):
            return (decode(angles) - native_ligand).reshape(-1)
        rng = np.random.RandomState(seed)
        starts = [coupling['angles'], np.zeros(len(bonds))] + [rng.uniform(-np.pi, np.pi, len(bonds)) for _ in range(args.restarts)]
        fits = []
        for i, start in enumerate(starts):
            fit = least_squares(residual, start, max_nfev=4000, ftol=1e-10, xtol=1e-10, gtol=1e-10) if len(bonds) else None
            angles = fit.x if fit is not None else start
            rmsd = float(np.sqrt(np.mean(np.sum(residual(angles).reshape(-1, 3)**2, axis=1))))
            fits.append(dict(start=i, success=bool(fit.success) if fit else True, nfev=int(fit.nfev) if fit else 0, ligand_rmsd=rmsd, angles=angles.tolist()))
        best = min(fits, key=lambda x: x['ligand_rmsd'])
        fitted_ligand = decode(np.asarray(best['angles']))
        # Rigid projection of the initial protein is the closed-form global optimum.
        rotation, translation, protein_rmsd, _ = _fit_numpy_rigid(poi[0].numpy(), native_poi)
        fitted_poi = poi[0].numpy() @ rotation.T + translation
        representable = endpoint_internal_geometry(source, fitted_ligand, ld['edge_index'])
        state_hash = hashlib.sha256(b''.join(value.detach().cpu().contiguous().numpy().tobytes() for value in (ligand, poi, body))).hexdigest()
        record = dict(initialization_device='cpu', initial_state_sha256=state_hash, production_target_reconstruction_rmsd=float(np.sqrt(np.mean(np.sum((decode(coupling["angles"]) - projected["ligand"])**2, axis=-1)))), seed=seed, initial=score(source, poi[0].numpy()), direct_oracle=score(fitted_ligand, fitted_poi), native_self_check=score(native_ligand, native_poi), protein_rigid_rmsd=protein_rmsd, production_target_projection=projection, multistart=fits, best_ligand_native_rmsd=best['ligand_rmsd'], reconstructed_internal_geometry=representable, max_initial_native_protein_pair_distance_difference=float(np.abs(torch.cdist(poi.double(), poi.double()).numpy()-torch.cdist(pd['coords'].double(), pd['coords'].double()).numpy()).max()))
        torch.save(dict(case=args.case, e3_data=ed, poi_data=pd, ligand_data=ld, initial_state=dict(ligand=ligand, poi=poi, body_frame=body), target_ligand=torch.as_tensor(projected['ligand'], dtype=ligand.dtype)[None], oracle_ligand=torch.tensor(fitted_ligand)[None], oracle_poi=torch.tensor(fitted_poi)[None], fullatom_targets=entry.fullatom_targets(item, ed, pd), metric_context=context, provenance=provenance), args.out / f'seed{seed}.pt')
        reports.append(record)
        print(json.dumps(record, default=lambda x: x.item()), flush=True)
        (args.out / 'summary.json').write_text(json.dumps(dict(case=args.case, native_used=True, prediction_claim=False, initialization_note='CPU torch.Generator differs from CUDA at equal seed; use saved actual state tensors for cross-diagnostic comparisons. These are not the earlier CUDA training initial states.', protocol='Closed-form globally optimal protein rigid fit; ligand torsions optimized with exact rigid fit and multistart least squares. No global optimum claim for ligand.', criteria=dict(protein_rmsd_tolerance_angstrom=1e-3, native_dockq_tolerance=1e-4, ligand_native_exact_tolerance_angstrom=1e-3), reports=reports), indent=2, default=lambda x: x.item()))


if __name__ == '__main__':
    main()
