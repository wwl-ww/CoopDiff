"""Intervene on opposite-interface coordinates seen by TriFuse only.

The expert inputs and original targets stay fixed. This isolates the TriFuse
conditioning route, not the separate cross-expert bridge route. Either branch
can be tested while its own pose and the shared conformation remain fixed.
"""

import copy

import torch
from scipy.spatial.transform import Rotation


def opposite_state(flow, batch, coupling, time, branch):
    """Return the intervened protein coordinates in the original global frame."""
    if branch not in ('e3_ligand', 'poi_ligand'):
        raise ValueError(f'unknown conditioning branch: {branch}')
    donor = flow.sample_coupling(
        batch['e3_data']['coords'], batch['poi_data']['coords'], batch['ligand_data'])
    own = 0 if branch == 'e3_ligand' else 1
    donor['angles'] = coupling['angles'].copy()
    donor['omegas'][own] = coupling['omegas'][own]
    donor['centers'][own] = coupling['centers'][own]
    original, _ = flow.path(batch['ligand_data'], batch['poi_data']['coords'], coupling, time)
    if own == 1:
        donor['omegas'][2] = coupling['omegas'][2]
        donor['poi_center'] = coupling['poi_center'].copy()
    changed, _ = flow.path(
        batch['ligand_data'], batch['poi_data']['coords'], donor, time)
    if own == 0:
        if not torch.equal(original['a'], changed['a']):
            raise AssertionError('conditioning intervention changed own ligand input')
        return 1, changed['poi'][0]
    if not torch.allclose(original['b'], changed['b']) or not torch.allclose(original['poi'], changed['poi']):
        raise AssertionError('conditioning intervention changed own B binary')
    ligand, donor_ligand = original['a'], changed['a']
    # Vary E3-L while holding POI-L fixed, then align the donor's common rigid
    # frame back to the original one. Only E3 coordinates are passed to TriFuse.
    value = float(time.item())
    original_rotation = Rotation.from_rotvec(value * coupling['omegas'][0]).as_matrix()
    donor_rotation = Rotation.from_rotvec(value * donor['omegas'][0]).as_matrix()
    rotation = ligand.new_tensor(original_rotation @ donor_rotation.T)

    def align(coords):
        return (coords - donor_ligand.mean(1, keepdim=True)) @ rotation.T + ligand.mean(1, keepdim=True)

    if not torch.allclose(align(donor_ligand), ligand, atol=1e-4, rtol=1e-5):
        raise AssertionError('POI conditioning intervention changed own ligand input')
    return 0, align(batch['e3_data']['coords'])[0]


@torch.no_grad()
def audit_conditioning(trainer, validation, branch='e3_ligand'):
    model, flow = trainer.model, trainer.diffusion
    model.eval()
    rows = []
    prefix = 'e3l' if branch == 'e3_ligand' else 'pol'
    for batch, coupling, value in validation:
        captured = []
        velocities = []
        time = torch.tensor([value], device=trainer.device)

        def capture_velocity(module, inputs, output):
            normalized = flow.normalized_predictions(output, time)
            velocity = flow.velocities(
                normalized, batch['e3_data']['coords'], batch['poi_data']['coords'])
            velocities.append({k: v.detach().clone()
                               for k, v in velocity[branch].items()})

        def capture_film(module, inputs, output):
            films = (output['film_params'][:model.num_conv_layers] if prefix == 'e3l'
                     else output['film_params'][model.num_conv_layers:])
            captured.append(torch.cat([
                tensor.detach().flatten()
                for pair in films
                for tensor in pair]))

        capture = model.trifuse.register_forward_hook(capture_film)
        capture_output = model.register_forward_hook(capture_velocity)
        try:
            baseline = trainer.compute_loss(
                **copy.deepcopy(batch), coupling_override=copy.deepcopy(coupling),
                t_override=time, diagnostics=True)
            protein_type, donor_coords = opposite_state(flow, batch, coupling, time, branch)

            def replace_other_state(module, inputs, kwargs):
                coords = kwargs['node_coords'].clone()
                coords[kwargs['node_types'] == protein_type] = donor_coords
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
                           for name in (f'loss_{prefix}_tr', f'loss_{prefix}_rot'))
            rows.append({
                'case': batch['case'], 'time': value,
                f'correct_{prefix}_loss': own_loss(baseline),
                f'shuffled_{prefix}_loss': own_loss(shuffled),
                'film_change_rms': (captured[2 if prefix == 'e3l' else 3] - captured[0 if prefix == 'e3l' else 1]).square().mean().sqrt().item(),
                'velocity_change_rms': {
                    k: (velocities[1][k] - velocities[0][k]).square().mean().sqrt().item()
                    for k in ('tr', 'rot')},
            })
        finally:
            capture.remove()
            capture_output.remove()
    return {'scope': f'{branch} TriFuse conditioning only; opposite state changed, fixed expert inputs',
            'rows': rows,
            'correct_mean': sum(r[f'correct_{prefix}_loss'] for r in rows) / len(rows),
            'shuffled_mean': sum(r[f'shuffled_{prefix}_loss'] for r in rows) / len(rows)}
