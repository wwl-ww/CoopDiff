"""Bounded runtime keys for immutable tensor views, with mutation detection."""


def tensor_view_key(tensor):
    """Return a storage/version key; callers retain tensors while caching it.

    Retaining source tensors prevents allocator pointer reuse. Inference tensors
    lack a version counter and must take the uncached path instead.
    """
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), str(tensor.device),
            str(tensor.dtype), version)
