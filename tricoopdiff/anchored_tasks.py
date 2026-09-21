"""Explicit conditioning tasks; missing binary information is not fabricated."""

from .anchored_flow import FLOW_VERSION
from .anchored_fragments import FRAGMENT_POLICY
from .anchored_joint_motion import ACTION_POLICY
from .anchored_training import TRAINING_CONTRACT, prepare_augmented_anchored_sample
from .bounded_binary_flow import BOUNDED_FLOW_VERSION
from .bounded_binary_training import BOUNDED_TRAINING_CONTRACT, validate_binary_bounds
from .bounded_joint_motion import BOUNDED_ACTION_POLICY
from .single_anchor_flow import SingleAnchorTorsionFlow
from .single_anchor_joint_motion import SINGLE_ACTION_POLICY
from .single_anchor_training import (SINGLE_FRAGMENT_POLICY, SINGLE_TRAINING_CONTRACT,
                                     prepare_single_anchor_sample)


def task_contract(task, bounds):
    validate_binary_bounds(bounds)
    if task == 'dual_assembly':
        from .assembly_flow import ASSEMBLY_FLOW, ASSEMBLY_TRAINING
        if bounds is not None:
            raise ValueError('soft assembly has no binary displacement bounds')
        return {'generative_process': ASSEMBLY_FLOW, 'training_contract': ASSEMBLY_TRAINING,
                'fragment_policy': FRAGMENT_POLICY, 'action_policy': 'unbounded_joint_interface_v1'}
    if task in ('dual_core', 'dual_contact_core'):
        from .binding_core import CORE_POLICY, CONTACT_CORE_POLICY
        from .core_training import CORE_TRAINING_CONTRACT, CONTACT_CORE_TRAINING_CONTRACT
        if bounds is None:
            raise ValueError('core training requires explicit binary bounds')
        return {'generative_process': BOUNDED_FLOW_VERSION,
                'training_contract': CONTACT_CORE_TRAINING_CONTRACT if task == 'dual_contact_core' else CORE_TRAINING_CONTRACT,
                'fragment_policy': CONTACT_CORE_POLICY if task == 'dual_contact_core' else CORE_POLICY,
                'action_policy': BOUNDED_ACTION_POLICY}
    if task == 'single_anchor':
        if bounds is None or any(values[0] != values[1] for values in bounds.values()):
            raise ValueError('single-anchor task requires explicit shared binary uncertainty bounds')
        return {'generative_process': SingleAnchorTorsionFlow.version,
                'training_contract': SINGLE_TRAINING_CONTRACT,
                'fragment_policy': SINGLE_FRAGMENT_POLICY, 'action_policy': SINGLE_ACTION_POLICY}
    if task != 'dual_binary':
        raise ValueError('unsupported anchored conditioning task')
    return {'generative_process': BOUNDED_FLOW_VERSION if bounds else FLOW_VERSION,
            'training_contract': BOUNDED_TRAINING_CONTRACT if bounds else TRAINING_CONTRACT,
            'fragment_policy': FRAGMENT_POLICY,
            'action_policy': BOUNDED_ACTION_POLICY if bounds else ACTION_POLICY}


def prepare_task_sample(task, e3, poi, ligand, *, generator, evidence_id, binary_bounds):
    task_contract(task, binary_bounds)
    if task == 'dual_assembly':
        from .assembly_flow import prepare_assembly_sample
        return prepare_assembly_sample(e3, poi, ligand, generator=generator, evidence_id=evidence_id)
    if task in ('dual_core', 'dual_contact_core'):
        from .core_training import prepare_augmented_core_sample
        return prepare_augmented_core_sample(e3, poi, ligand, generator=generator,
            evidence_id=evidence_id, binary_bounds=binary_bounds,
            all_contact_blocks=task == 'dual_contact_core')
    if task == 'single_anchor':
        return prepare_single_anchor_sample(e3, poi, ligand, generator=generator, evidence_id=evidence_id,
            translation_bound=binary_bounds['translation'][0], rotation_bound=binary_bounds['rotation'][0])
    return prepare_augmented_anchored_sample(e3, poi, ligand, generator=generator,
                                             evidence_id=evidence_id, binary_bounds=binary_bounds)


def validate_task_sample(sample, task, bounds, prior_config):
    contract = task_contract(task, bounds)
    if (sample.audit.get('training_contract') != contract['training_contract']
            or sample.audit.get('fragment_augmentation', {}).get('policy') != contract['fragment_policy']
            or sample.flow.version != contract['generative_process']
            or sample.audit.get('prior', {}).get('config') != prior_config):
        raise ValueError('training sample does not follow the conditioning task/prior contract')
    present = sample.conditions['sides'][:, 2] > 0
    if task in ('dual_binary', 'dual_core', 'dual_contact_core', 'dual_assembly'):
        if sample.audit.get('binary_bounds') != bounds or not bool(present.all()):
            raise ValueError('dual-binary sample requires both complete binary conditions and declared bounds')
    else:
        expected = {'translation': bounds['translation'][0], 'rotation': bounds['rotation'][0]}
        if (sample.audit.get('known_side_bounds') != expected or not bool(present[0])
                or bool(present[1])):
            raise ValueError('single-anchor sample must constrain only its known first side')
