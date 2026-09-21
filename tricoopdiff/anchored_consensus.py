"""Experimental adapter reusing consensus heads for the anchored torsion flow.

Explicit source is still required by the existing network. A production training
entry must version its new endpoint/source policy; old weights are not evidence
of training on this process. This module does not load or reinterpret checkpoints.
"""

import math

import torch

from .sdpi_flow import SDPIFlow
from .anchored_joint_motion import joint_anchored_velocity
from .bounded_binary_flow import BoundedBinaryTorsionFlow
from .bounded_joint_motion import bounded_joint_velocity
from .single_anchor_flow import SingleAnchorTorsionFlow
from .single_anchor_joint_motion import single_joint_velocity
from .anchored_training_metric import PARAMETER_METRIC, PHYSICAL_METRIC, binary_offset_metric


def predict_anchored_velocity(model, flow, angles, e3_data, poi_data, ligand_data,
                              time, dt, *, conditions, source):
    """Run the unchanged consensus forward on a decoded conditional state."""
    ligand, poi, body = flow.decode(angles)
    t = angles.new_tensor([float(time)])
    step = angles.new_tensor([float(dt)])
    if not 0 <= float(time) <= 1 or not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError('invalid anchored model clock')
    if ligand_data['coords'].shape != (1, *ligand.shape) or poi_data['coords'].shape != (1, *poi.shape):
        raise ValueError('model data and anchored reference disagree')
    # Reuse pretrained feature/output units, not SDPI's motion path or step.
    units = SDPIFlow()
    conditioning = units.conditioning(t, step, e3_data['coords'], poi_data['coords'])
    sigma, _, _ = units.t_to_sigma(t, t, t)
    crop = (sigma * 3 + 20 if getattr(model.expert.model, 'dynamic_max_cross', False)
            else None)
    output = model(
        e3_data, {**poi_data, 'coords': poi[None]},
        {**ligand_data, 'coords': ligand[None]}, t_e3l=t, t_pol=t,
        r_e3_to_poi=body[None], receptor_crop_cutoff=crop,
        diffusion_state=conditioning, source=source, conditions=conditions)
    if isinstance(flow, SingleAnchorTorsionFlow):
        convert = single_joint_velocity
    else:
        convert = bounded_joint_velocity if isinstance(flow, BoundedBinaryTorsionFlow) else joint_anchored_velocity
    velocity, diagnostics = convert(
        flow, angles, output['normalized_velocity'],
        units.scales(e3_data['coords'], poi_data['coords']))
    adapter = getattr(model.trifuse, 'anchored_action_adapter', None)
    if adapter is not None:
        context = {**output['anchored_action_context'], 'conditions': conditions,
                   'e3_coords': e3_data['coords'][0]}
        correction = adapter(flow, angles, context, velocity,
                             units.scales(e3_data['coords'], poi_data['coords']), time)
        count = len(flow.plan.bonds)
        velocity = (velocity + correction if getattr(adapter, 'full_state_action', False)
                    else torch.cat((velocity[:count] + correction, velocity[count:])))
        diagnostics = {**diagnostics, 'body_aware_residual_rms':
                       float(correction.detach().square().mean().sqrt()) if count else 0.}
    output = {**output, 'anchored_joint_motion': diagnostics}
    return velocity, output


def anchored_velocity_loss(model, flow, data_angles, prior_angles,
                           e3_data, poi_data, ligand_data, time, dt, *,
                           conditions, source, cartesian_weight=.1, binary_metric=PARAMETER_METRIC):
    """Train allowed torsions and the Cartesian motion they induce on both bodies."""
    if not math.isfinite(cartesian_weight) or cartesian_weight < 0:
        raise ValueError('Cartesian weight must be finite and nonnegative')
    if binary_metric not in (PARAMETER_METRIC, PHYSICAL_METRIC):
        raise ValueError('unsupported binary velocity loss metric')
    if not len(flow.plan.bonds) and not isinstance(flow, SingleAnchorTorsionFlow):
        raise ValueError('fully rigid chart has no trainable torsion target')
    angles, _, target = flow.path(data_angles, prior_angles, time)
    angles, target = angles.detach(), target.detach()
    predicted, output = predict_anchored_velocity(
        model, flow, angles, e3_data, poi_data, ligand_data, time, dt,
        conditions=conditions, source=source)
    count = len(flow.plan.bonds)
    torsion_loss = (((predicted[:count] - target[:count]) / math.pi).square().mean()
                    if count else predicted.sum() * 0)
    offset_loss, free_pose_loss = torsion_loss.new_zeros(()), torsion_loss.new_zeros(())
    if isinstance(flow, SingleAnchorTorsionFlow):
        free_pose_loss = (predicted[count + 6:] - target[count + 6:]).square().mean()
    if isinstance(flow, (SingleAnchorTorsionFlow, BoundedBinaryTorsionFlow)):
        offset_loss = binary_offset_metric(flow, angles, predicted - target,
            SDPIFlow.scales(e3_data['coords'], poi_data['coords']), policy=binary_metric)
    cartesian_loss = torsion_loss.new_zeros(())
    if cartesian_weight:
        delta_ligand, delta_poi = flow.cartesian_velocity(
            angles, predicted - target, create_graph=torch.is_grad_enabled())
        # Give each body equal aggregate weight; normalize units by receptor size.
        scale = SDPIFlow.scales(e3_data['coords'], poi_data['coords']).mean()
        cartesian_loss = .5 * ((delta_ligand / scale).square().mean()
                               + (delta_poi / scale).square().mean())
    total = torsion_loss + offset_loss + free_pose_loss + cartesian_weight * cartesian_loss
    from .assembly_flow import AssemblyFlow
    assembly_loss = total.new_zeros(())
    if isinstance(flow, AssemblyFlow):
        from .prior_joint_action import prior_energies
        endpoint = angles - float(time) * predicted
        predicted_ligand, predicted_poi, _ = flow.decode(endpoint)
        # Reconstruct local binary evidence at the denoised endpoint. Conditions
        # contain no native inter-protein transform; the flow loss supplies the
        # native assembly supervision separately.
        energies = prior_energies(predicted_ligand, predicted_poi, e3_data['coords'][0], conditions)
        assembly_loss = energies[:2].mean() + .1 * energies[2]
        total = total + .2 * assembly_loss
    if not torch.isfinite(total):
        raise FloatingPointError('non-finite anchored flow loss')
    return {'loss_total': total, 'loss_torsion': torsion_loss, 'loss_binary_offset': offset_loss,
            'loss_assembly': assembly_loss,
            'loss_free_pose': free_pose_loss,
            'loss_cartesian': cartesian_loss, 'model_output': output}


def anchored_target_loss(model, flow, target_ligand, target_poi, prior_angles,
                         e3_data, poi_data, ligand_data, time, dt, *,
                         conditions, source, target_tolerance=1e-3, cartesian_weight=.1,
                         binary_metric=PARAMETER_METRIC):
    """Encode an actual target before forwarding; never silently project it.

    Target coordinates are supervision only, already in the anchored E3 frame.
    An independent binary assembly can have different fixed internal geometry;
    such a sample is rejected rather than relabeled as a reachable endpoint.
    No condition dropout is performed here: changing constraints requires a new
    chart, prior and encoded target, not just blanking the feature tensors.
    """
    with torch.no_grad():
        data_angles, audit = flow.encode_target(
            target_ligand, target_poi, ligand_data['edge_index'],
            tolerance=target_tolerance)
    losses = anchored_velocity_loss(
        model, flow, data_angles, prior_angles, e3_data, poi_data, ligand_data,
        time, dt, conditions=conditions, source=source,
        cartesian_weight=cartesian_weight, binary_metric=binary_metric)
    return {**losses, 'target_audit': audit}


@torch.no_grad()
def sample_anchored(model, flow, initial_angles, e3_data, poi_data, ligand_data,
                    *, steps, conditions, source):
    """Integrate in angle space using the same forward and decoder as training."""
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    angles = initial_angles.detach().clone()
    state = flow.decode(angles)
    was_training = model.training
    model.eval()
    try:
        for index in range(steps):
            velocity, _ = predict_anchored_velocity(
                model, flow, angles, e3_data, poi_data, ligand_data,
                1 - index / steps, 1 / steps, conditions=conditions, source=source)
            angles, state = flow.step(angles, velocity, 1 / steps)
    finally:
        model.train(was_training)
    return {'angles': angles, 'ligand_coords': state[0], 'poi_coords': state[1],
            'body': state[2], 'generative_process': flow.version}
