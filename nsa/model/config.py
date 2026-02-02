"""Model configs for NSA-100M, NSA-150M, and the dense-100M baseline.

Each config carries both architecture (depth, width, heads) and NSA-specific
knobs. The dense baseline shares everything except `attention="dense"`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    hidden_size: int = 768  # n_heads * head_dim
    intermediate_size: int = 2048  # SwiGLU expanded; ~2.67x hidden_size pre-glu split
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tied_embeddings: bool = True

    # Attention switch: "dense" uses torch SDPA; "nsa" uses the NSA combined forward.
    attention: Literal["dense", "nsa"] = "nsa"

    # NSA hyperparameters (ignored when attention == "dense").
    nsa_block_size_c: int = 64
    nsa_block_size_n: int = 64
    nsa_block_size_m: int = 64
    nsa_top_k: int = 16
    nsa_window_size: int = 512
    nsa_pool: Literal["mean", "learned"] = "mean"
    nsa_gate_activation: Literal["sigmoid", "softmax"] = "sigmoid"

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count, ignoring biases and norm scales."""
        embed = self.vocab_size * self.hidden_size  # tied: counted once.
        attn = self.n_layers * 4 * self.hidden_size * self.hidden_size  # q, k, v, o
        # SwiGLU has gate + up + down: 3 matrices.
        ffn = self.n_layers * 3 * self.hidden_size * self.intermediate_size
        return embed + attn + ffn


def nsa_100m() -> TransformerConfig:
    return TransformerConfig(
        n_layers=12,
        n_heads=12,
        head_dim=64,
        hidden_size=768,
        intermediate_size=2048,
        attention="nsa",
    )


def dense_100m() -> TransformerConfig:
    cfg = nsa_100m()
    cfg.attention = "dense"
    return cfg


def nsa_150m() -> TransformerConfig:
    return TransformerConfig(
        n_layers=14,
        n_heads=14,
        head_dim=64,
        hidden_size=896,
        intermediate_size=2400,
        attention="nsa",
    )
