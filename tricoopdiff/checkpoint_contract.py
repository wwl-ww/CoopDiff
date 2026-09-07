"""Checkpoint dependencies that are not contained in adapter-only payloads."""

import warnings


_EXPERT_IDENTITY_KEYS = ('diffdock_commit', 'score_checkpoint_sha256', 'no_torsion')


def expert_base_identity(expert):
    """Return the stable identity fields for the external frozen expert."""
    provenance = getattr(expert, 'provenance', {})
    missing = [key for key in _EXPERT_IDENTITY_KEYS if key not in provenance]
    if missing:
        raise ValueError(f'frozen expert has incomplete provenance: {missing}')
    return {key: provenance[key] for key in _EXPERT_IDENTITY_KEYS}


def validate_expert_base_identity(checkpoint, expert):
    """Reject known base/adapter mismatches and label legacy uncertainty."""
    actual = expert_base_identity(expert)
    saved = checkpoint.get('expert_base_identity')
    if saved is None:
        warnings.warn(
            'Checkpoint has no frozen-expert identity; compatibility with the '
            'runtime DiffDock base cannot be proven.', RuntimeWarning, stacklevel=2)
        return 'legacy_unrecorded'
    if saved != actual:
        raise ValueError(f'checkpoint frozen expert identity {saved!r} does not '
                         f'match runtime {actual!r}')
    return saved
