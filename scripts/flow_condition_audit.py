"""Intervene on opposite-interface coordinates seen by TriFuse only.

The expert inputs and original targets stay fixed. This isolates the TriFuse
conditioning route, not the separate cross-expert bridge route. Only the E3-L
branch is tested: POI gets an independently drawn relative pose with the same
shared ligand conformation and E3-L pose.
"""

import copy

import torch


@torch.no_grad()
def audit_conditioning(trainer, validation):
    model, flow = trainer.model, trainer.diffusion
    model.eval()
    rows = []
    for batch, coupling, value in validation:
        captured = []
        velocities = []
        time = torch.tensor([value], device=trainer.device)

        def capture_velocity(module, inputs, output):
            normalized = flow.normalized_predictions(output, time)
            velocity = flow.velocities(
                normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
            velocities.append({k: v.detach().clone()
                               for k, v in velocity['e3_ligand'].items()})

        def capture_film(module, inputs, output):
            captured.append(torch.cat([
                tensor.detach().flatten()
                for pair in output['film_params'][:model.num_conv_layers]
                for tensor in pair]))

        capture = model.trifuse.register_forward_hook(capture_film)
        capture_output = model.register_forward_hook(capture_velocity)
        try:
            baseline = trainer.compute_loss(
                **copy.deepcopy(batch), coupling_override=copy.deepcopy(coupling),
                t_override=time, diagnostics=True)
            donor = flow.sample_coupling(
                batch['e3_data']['coords'], batch['poi_data']['coords'],
                batch['ligand_data'], no_torsion=bool(model.expert.model.no_torsion))
            donor['angles'] = coupling['angles'].copy()
            donor['omegas'][0] = coupling['omegas'][0]
            donor['centers'][0] = coupling['centers'][0]
            original_ligand = flow.path(
                batch['ligand_data'], batch['poi_data']['coords'], coupling, time)[0]
            donor_ligand, donor_poi, _, _ = flow.path(
                batch['ligand_data'], batch['poi_data']['coords'], donor, time)
            if not torch.equal(original_ligand, donor_ligand):
                raise AssertionError('conditioning intervention changed own ligand input')

            def replace_other_state(module, inputs, kwargs):
                coords = kwargs['node_coords'].clone()
                coords[kwargs['node_types'] == 1] = donor_poi[0]
                return inputs, {**kwargs, 'node_coords': coords}

            intervention = model.trifuse.register_forward_pre_hook(
                replace_other_state, with_kwargs=True)
            try:
                shuffled = trainer.compute_loss(
                    **copy.deepcopy(batch), coupling_override=copy.deepcopy(coupling),
                    t_override=time, diagnostics=True)
            finally:
                intervention.remove()
            def own_loss(result):
                return sum(result['components'][name].item()
                           for name in ('loss_e3l_tr', 'loss_e3l_rot'))
            rows.append({
                'case': batch['case'], 'time': value,
                'correct_e3l_loss': own_loss(baseline),
                'shuffled_e3l_loss': own_loss(shuffled),
                'film_change_rms': (captured[1] - captured[0]).square().mean().sqrt().item(),
                'velocity_change_rms': {
                    k: (velocities[1][k] - velocities[0][k]).square().mean().sqrt().item()
                    for k in ('tr', 'rot')},
            })
        finally:
            capture.remove()
            capture_output.remove()
    return {'scope': 'E3-L TriFuse conditioning only; independent POI state, fixed expert inputs',
            'rows': rows,
            'correct_mean': sum(r['correct_e3l_loss'] for r in rows) / len(rows),
            'shuffled_mean': sum(r['shuffled_e3l_loss'] for r in rows) / len(rows)}
