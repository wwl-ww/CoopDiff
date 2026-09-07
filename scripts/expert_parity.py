"""Separate CUDA repeatability from native/FiLM implementation parity."""

import hashlib

import torch


def _fingerprint(value):
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        return (str(tensor.dtype), tuple(tensor.shape),
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
    if isinstance(value, (tuple, list)):
        return tuple(_fingerprint(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _fingerprint(item)) for key, item in sorted(value.items()))
    return value


@torch.no_grad()
def audit_binary_preservation(model, batch, state, time, trace_layers=False):
    """Repeat each path; optionally locate the first differing convolution.

    Inputs and model weights are fixed. Stage is restored even on failure.
    Outputs are JSON-safe diagnostics; callers own the acceptance decision.
    """
    previous_stage, previous_training = model.dual_stage, model.training
    captured, handles, label = {}, [], None
    def hook(index):
        def record(module, inputs, kwargs, output):
            captured[label].setdefault(index, []).append({
                'inputs': _fingerprint((inputs, kwargs)),
                'output': output.detach().cpu().clone(),
            })
        return record
    outputs = {}
    def forward(stage):
        model.dual_stage = stage
        return model(batch['e3_data'], {**batch['poi_data'], 'coords': state['poi']},
                     {**batch['ligand_data'], 'coords': state['a']}, t_e3l=time,
                     ligand_b_coords=state['b'], r_e3_to_poi=state['body'])['interface_scores']
    try:
        model.eval()
        # e3nn's TorchScript graphs specialize after their initial calls. Warm
        # both schedules before attributing cold/hot rounding to path changes.
        forward('independent')
        forward('annealed')
        if trace_layers:
            for group in ('rec_emb_layers', 'lig_emb_layers', 'conv_layers'):
                for index, layer in enumerate(getattr(model.expert.model, group, ())):
                    handles.append(layer.register_forward_hook(hook(f'{group}.{index}'), with_kwargs=True))
        for label, stage in (('native_0', 'independent'), ('native_1', 'independent'),
                             ('film_0', 'annealed'), ('film_1', 'annealed')):
            captured[label] = {}
            outputs[label] = forward(stage)
    finally:
        for handle in handles:
            handle.remove()
        model.dual_stage = previous_stage
        model.train(previous_training)
    report = {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'comparisons': {}}
    for name, left, right in (('native_repeat', 'native_0', 'native_1'),
                              ('film_repeat', 'film_0', 'film_1'),
                              ('native_vs_film', 'native_0', 'film_0')):
        fields = {}
        for branch in outputs[left]:
            for component, x in outputs[left][branch].items():
                y = outputs[right][branch][component]
                fields[f'{branch}.{component}'] = {
                    'max_abs_error': float((x - y).abs().max()) if x.numel() else 0.,
                    'equal': torch.equal(x, y),
                    'allclose': torch.allclose(x, y, atol=2e-4, rtol=2e-4),
                }
        report['comparisons'][name] = {'fields': fields,
            'max_abs_error': max(item['max_abs_error'] for item in fields.values()),
            'equal': all(item['equal'] for item in fields.values()),
            'allclose': all(item['allclose'] for item in fields.values())}
        if trace_layers:
            report['comparisons'][name]['layers'] = [
                {'layer': index, 'branch': branch, 'identical_inputs': x['inputs'] == y['inputs'],
                 'output_max_abs_error': float((x['output'] - y['output']).abs().max())}
                for index in captured[left]
                for branch, x, y in zip(('sub1', 'sub2'), captured[left][index], captured[right][index])]
    return report
