"""
Score merging strategies for combining two-body expert outputs.

Default: Residual-gated merge -> s_q = s_q1 + g * (s_q2 - s_q1)

The residual gate ``g`` (one scalar per score type: tr / rot / tor) is predicted
by TriBrain's zero-initialized ``MergeGateHead``. At init ``g=0`` → ``s_q = s_q1``
(pure E3-L), which is what makes the zero-FiLM baseline transparent: without any
learned coordination the model reproduces the frozen E3-L two-body expert exactly
instead of double-counting both subgraphs. As training moves ``g`` away from 0,
the POI-L force field (``s_q2 - s_q1``) is blended in — the same adaLN-Zero
philosophy as FiLM.

``merge_direct_addition`` (s_q = s_q1 + s_q2) is kept only as an L0 ablation: it
double-counts the shared PROTAC DOFs and breaks zero-FiLM transparency, which is
exactly the bug the residual-gated merge fixes.
"""

import torch


def merge_direct_addition(scores_1: dict, scores_2: dict) -> dict:
    """
    Direct addition of scores from two subgraphs (L0 ablation, NOT default).

    s_q = s_q1 + s_q2

    Each two-body expert independently predicts how PROTAC should move relative
    to its protein; adding them double-counts the shared PROTAC DOFs. Kept only
    to measure the effect of the transparent residual-gated merge against it.

    Args:
        scores_1: {'tr_score': [B,3], 'rot_score': [B,3], 'tor_score': [B,K]}
        scores_2: same format

    Returns:
        merged scores dict
    """
    merged = {}
    for key in scores_1:
        if scores_1[key] is not None and scores_2[key] is not None:
            merged[key] = scores_1[key] + scores_2[key]
        elif scores_1[key] is not None:
            merged[key] = scores_1[key]
        else:
            merged[key] = scores_2[key]
    return merged


def merge_residual_gated(scores_1: dict, scores_2: dict, gate: torch.Tensor) -> dict:
    """
    Residual-gated merge (default).

    s_q = s_q1 + g * (s_q2 - s_q1)

    ``gate`` = [B, 3] residual gate (tr, rot, tor), zero-initialized in TriBrain,
    so at init ``gate=0`` → ``s_q = s_q1`` (pure E3-L). ``g=1`` → ``s_q = s_q2``
    (pure POI-L); intermediate ``g`` interpolates the two two-body force fields.

    Args:
        scores_1: {'tr_score': [B,3], 'rot_score': [B,3], 'tor_score': [B,K]}
        scores_2: same format (tr/rot already rotated back to the E3 frame by
                  the caller)
        gate: [B, 3] residual gate per score type (tr, rot, tor)

    Returns:
        merged scores dict
    """
    g_tr, g_rot, g_tor = gate[:, 0], gate[:, 1], gate[:, 2]
    gates = {'tr_score': g_tr, 'rot_score': g_rot, 'tor_score': g_tor}

    merged = {}
    for key in scores_1:
        s1, s2 = scores_1[key], scores_2[key]
        # Empty / None scores (e.g. tor when the conformer has no rotatable bonds)
        # pass through ungated — gating an empty tensor is a no-op and would only
        # change its shape via broadcasting.
        if s1 is None or s1.numel() == 0:
            merged[key] = s1 if s1 is not None else s2
        elif s2 is None or s2.numel() == 0:
            merged[key] = s1
        else:
            g = gates[key]
            merged[key] = s1 + g.unsqueeze(-1) * (s2 - s1)

    merged['merge_gates'] = {'tr': g_tr, 'rot': g_rot, 'tor': g_tor}
    return merged
