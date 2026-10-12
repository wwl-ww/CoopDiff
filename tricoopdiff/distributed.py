"""Explicit synchronous gradient reduction for the custom rollout trainer.

The rollout invokes model submodules directly, so wrapping only its top-level
forward in DistributedDataParallel would not express the training contract.
"""
import torch
import torch.distributed as dist


def all_reduce_and_wait(tensor, op=dist.ReduceOp.SUM):
    """Retain a Work handle so NCCL's blocking host wait is honored.

    New NCCL synchronous collectives return no Work and only order CUDA
    streams. A subsequent item()/cpu() then spins in CUDA synchronization.
    With TORCH_NCCL_BLOCKING_WAIT=1, the explicit Work.wait sleeps instead
    and retains the process group's timeout/error handling.
    """
    work = dist.all_reduce(tensor, op=op, async_op=True)
    work.wait()


class GradientSynchronizer:
    """Reduce mean local gradients into the sample-weighted global mean."""

    def __init__(self, device):
        self.device = torch.device(device)

    def check_error(self, error):
        """All ranks must call, including ranks whose local backward failed."""
        failed = torch.tensor(int(error is not None), device=self.device)
        all_reduce_and_wait(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError('distributed update aborted: a rank failed before gradient synchronization') from error

    def __call__(self, parameters, local_samples):
        if local_samples < 1:
            raise ValueError('each rank must have at least one sample')
        count = torch.tensor(local_samples, dtype=torch.int64, device=self.device)
        all_reduce_and_wait(count)
        total = int(count.item())
        # Parameters can be unused on one rank (e.g. no ligand torsions). A
        # shared presence mask keeps collective order identical; globally
        # unused gradients stay None, preserving AdamW's skip semantics.
        used = torch.tensor([int(p.grad is not None) for p in parameters],
                            dtype=torch.int64, device=self.device)
        all_reduce_and_wait(used, op=dist.ReduceOp.MAX)
        active = [p for p, present in zip(parameters, used.cpu().tolist()) if present]
        # Bucket small parameter tensors to avoid hundreds of latency-bound
        # collectives. Each rank derives the same buckets from the shared mask.
        bucket, elements, signature = [], 0, None

        def reduce_bucket(items):
            flat = torch.cat([(p.grad.detach() if p.grad is not None else
                               torch.zeros_like(p)).reshape(-1) for p in items])
            flat.mul_(local_samples)
            all_reduce_and_wait(flat)
            flat.div_(total)
            offset = 0
            for parameter in items:
                reduced = flat[offset:offset + parameter.numel()].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = reduced.clone()
                else:
                    parameter.grad.copy_(reduced)
                offset += parameter.numel()

        for parameter in active:
            current = (parameter.device, parameter.dtype)
            if bucket and (current != signature or
                           (elements + parameter.numel()) * parameter.element_size() > 25 * 1024 ** 2):
                reduce_bucket(bucket)
                bucket, elements = [], 0
            bucket.append(parameter)
            elements += parameter.numel()
            signature = current
        if bucket:
            reduce_bucket(bucket)
        return total


def shard_global_batch(entries, global_batch_size, rank, world_size):
    """Partition one complete global batch into disjoint contiguous shards."""
    if not 0 <= rank < world_size or global_batch_size < world_size:
        raise ValueError('invalid rank, world size or global batch size')
    if global_batch_size % world_size:
        raise ValueError('global batch size must be divisible by world size')
    if not world_size <= len(entries) <= global_batch_size:
        raise ValueError('global batch must have at least one real sample per rank; no padding')
    size, remainder = divmod(len(entries), world_size)
    start = rank * size + min(rank, remainder)
    return entries[start:start + size + int(rank < remainder)]


def validate_distributed_checkpoint(payload, world_size, global_batch_size,
                                    *, train_index_sha256=None, seed=None,
                                    phase=None, random_flip_proteins=None,
                                    epoch_size=None, checkpoint_ternary=None):
    """Validate rank identity and data cursor before restoring optimization.

    Legacy smoke artifacts lack data hashes; their manifest must be checked by
    the caller. New checkpoints pin the data and sampling policy explicitly.
    """
    state = payload.get('distributed_state')
    if not isinstance(state, dict):
        raise ValueError('distributed resume requires a distributed checkpoint')
    if state.get('contract') != 'explicit_sample_weighted_allreduce_v1':
        raise ValueError('unsupported distributed synchronization contract')
    if state.get('world_size') != world_size or state.get('global_batch_size') != global_batch_size:
        raise ValueError('distributed resume requires matching world size and global batch size')
    rngs = state.get('rank_rng_states')
    if not isinstance(rngs, list) or len(rngs) != world_size:
        raise ValueError('distributed checkpoint has incomplete rank RNG states')
    for rng in rngs:
        if not isinstance(rng, dict) or not {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= rng.keys():
            raise ValueError('distributed checkpoint has malformed rank RNG state')
    cursor = state.get('next_data_cursor')
    if not isinstance(cursor, dict):
        raise ValueError('distributed checkpoint has no data cursor')
    for key in ('epoch', 'offset'):
        if isinstance(cursor.get(key), bool) or not isinstance(cursor.get(key), int) or cursor[key] < 0:
            raise ValueError('distributed checkpoint has invalid data cursor')
    if cursor['offset'] % global_batch_size:
        raise ValueError('distributed checkpoint cursor must be on a global batch boundary')
    if epoch_size is not None:
        if epoch_size <= 0 or cursor['offset'] >= epoch_size:
            raise ValueError('distributed checkpoint cursor is incompatible with dataset size')
        step = payload.get('global_step')
        if step != cursor['epoch'] * ((epoch_size + global_batch_size - 1) // global_batch_size) + cursor['offset'] // global_batch_size:
            raise ValueError('distributed data cursor differs from optimizer update count')
        if payload.get('training_state', {}).get('samples_seen') != cursor['epoch'] * epoch_size + cursor['offset']:
            raise ValueError('distributed sample count differs from optimizer update count')
    if state.get('resume_contract') is not None:
        if state['resume_contract'] != 'rank_rng_cursor_dataset_v1':
            raise ValueError('unsupported distributed resume contract')
        expected = {'train_index_sha256': train_index_sha256, 'seed': seed,
                    'phase': phase, 'random_flip_proteins': random_flip_proteins,
                    'epoch_size': epoch_size, 'checkpoint_ternary': checkpoint_ternary}
        for key, value in expected.items():
            if value is not None and state.get(key) != value:
                raise ValueError(f'distributed checkpoint {key} differs')
    return state


def balanced_sample_assignment(costs, world_size):
    """Deterministic LPT assignment; retain each shard's global sample order."""
    if len(costs) < world_size or world_size < 1:
        raise ValueError('balanced assignment requires one real sample per rank')
    import math
    if any(not math.isfinite(float(cost)) or float(cost) <= 0 for cost in costs):
        raise ValueError('sample costs must be positive and finite')
    assigned, loads = [[] for _ in range(world_size)], [0.] * world_size
    for index in sorted(range(len(costs)), key=lambda index: (-costs[index], index)):
        rank = min(range(world_size), key=lambda rank: (loads[rank], rank))
        assigned[rank].append(index)
        loads[rank] += costs[index]
    return [sorted(indices) for indices in assigned]


def move_batch_tensors(value, device):
    """Move immutable nested cache payloads without changing atom ordering."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_tensors(item, device) for item in value)
    return value


def endpoint_near_policy(trainer, sample_index):
    if trainer.training_recipe == 'endpoint_near_target_v2':
        return ((sample_index // 2) % 2 == 0,
                trainer.near_target_supervision['start_times'][(sample_index // 4) % 2])
    return sample_index % 2 == 0, trainer.near_target_supervision['start_t']


def presample_endpoint_randomness(trainer, batch, sample_index):
    """Advance the original logical rank's legacy CPU/CUDA streams exactly.

    Valid for the endpoint zero-dropout forward contract: SDPI consumes one CPU
    integer, followed by the near branch's CUDA normals in their original order.
    Protein flip RNG was already consumed by the original rank's loader.
    """
    from tricoopdiff.sdpi_flow import INDEPENDENT_RIGID_MODE, CONTINUOUS_TORSION_MODE
    # Imported locally to avoid pulling model dependencies into CPU shard tests.
    from tricoopdiff.train import VECTOR_EXPERT_MODEL_PROFILE
    result = dict(batch)
    result['initialization_seed_override'] = int(torch.randint(0, 2**31 - 1, (1,)).item())
    if trainer.near_target_weight:
        zero, _ = endpoint_near_policy(trainer, sample_index)
        if not zero:
            ligand = batch['ligand_data']['coords']
            no_torsion = bool(getattr(trainer.model.expert.model, 'no_torsion', False))
            result['near_noise_override'] = {
                'rigid_output_mode': INDEPENDENT_RIGID_MODE,
                **({'torsion_update_mode': CONTINUOUS_TORSION_MODE}
                   if trainer.model.trifuse.config.get('model_profile') in
                   ('ternary_endpoint_v4', VECTOR_EXPERT_MODEL_PROFILE) else {}),
                'torsion': ligand.new_empty((1, 0 if no_torsion else int(batch['ligand_data']['edge_mask'].sum()))).normal_(
                    0, trainer.near_target_supervision['torsion_std_radians']),
                **{branch: {'tr': ligand.new_empty((1, 3)).normal_(
                        0, trainer.near_target_supervision['translation_std_angstrom']),
                    'rot': ligand.new_empty((1, 3)).normal_(
                        0, trainer.near_target_supervision['rotation_std_radians'])}
                   for branch in ('e3_ligand', 'poi_ligand')},
            }
    return result
