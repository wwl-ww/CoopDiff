"""Explicit synchronous gradient reduction for the custom rollout trainer.

The rollout invokes model submodules directly, so wrapping only its top-level
forward in DistributedDataParallel would not express the training contract.
"""
import torch
import torch.distributed as dist


class GradientSynchronizer:
    """Reduce mean local gradients into the sample-weighted global mean."""

    def __init__(self, device):
        self.device = torch.device(device)

    def check_error(self, error):
        """All ranks must call, including ranks whose local backward failed."""
        failed = torch.tensor(int(error is not None), device=self.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError('distributed update aborted: a rank failed before gradient synchronization') from error

    def __call__(self, parameters, local_samples):
        if local_samples < 1:
            raise ValueError('each rank must have at least one sample')
        count = torch.tensor(local_samples, dtype=torch.int64, device=self.device)
        dist.all_reduce(count)
        total = int(count.item())
        # Parameters can be unused on one rank (e.g. no ligand torsions). A
        # shared presence mask keeps collective order identical; globally
        # unused gradients stay None, preserving AdamW's skip semantics.
        used = torch.tensor([int(p.grad is not None) for p in parameters],
                            dtype=torch.int64, device=self.device)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        active = [p for p, present in zip(parameters, used.cpu().tolist()) if present]
        # Bucket small parameter tensors to avoid hundreds of latency-bound
        # collectives. Each rank derives the same buckets from the shared mask.
        bucket, elements, signature = [], 0, None

        def reduce_bucket(items):
            flat = torch.cat([(p.grad.detach() if p.grad is not None else
                               torch.zeros_like(p)).reshape(-1) for p in items])
            flat.mul_(local_samples)
            dist.all_reduce(flat)
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
    if len(entries) != global_batch_size:
        raise ValueError('global batch must be complete; padding is not permitted')
    size = global_batch_size // world_size
    return entries[rank * size:(rank + 1) * size]


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
        if epoch_size <= 0 or epoch_size % global_batch_size or cursor['offset'] >= epoch_size:
            raise ValueError('distributed checkpoint cursor is incompatible with dataset size')
        step = payload.get('global_step')
        if step != cursor['epoch'] * (epoch_size // global_batch_size) + cursor['offset'] // global_batch_size:
            raise ValueError('distributed data cursor differs from optimizer update count')
        if payload.get('training_state', {}).get('samples_seen') != step * global_batch_size:
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
