"""Per-head, per-token gating across NSA's three branches:
out = g_c * out_c + g_s * out_s + g_w * out_w. Logits come from a small
projection in NSAAttention; this module just turns them into the
combination. Supports sigmoid (independent) and softmax (partition).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


def combine(
    out_c: Tensor,
    out_s: Tensor,
    out_w: Tensor,
    gate_logits: Tensor,
    activation: Literal["sigmoid", "softmax"] = "sigmoid",
) -> Tensor:
    """Combine three branch outputs via per-head learned gates.

    Args:
        out_c, out_s, out_w: branch outputs, each [B, H, Tq, D].
        gate_logits: [B, H, Tq, 3], pre-activation gate scores in branch order
            (compressed, selected, sliding).
        activation: "sigmoid" for independent branches (default) or "softmax"
            for a 3-way partition.

    Returns:
        out: [B, H, Tq, D], same dtype as the branch outputs.
    """
    if activation == "sigmoid":
        g = torch.sigmoid(gate_logits)
    elif activation == "softmax":
        g = torch.softmax(gate_logits, dim=-1)
    else:
        raise ValueError(f"unknown activation: {activation}")
    return g[..., 0:1] * out_c + g[..., 1:2] * out_s + g[..., 2:3] * out_w


__all__ = ["combine"]
