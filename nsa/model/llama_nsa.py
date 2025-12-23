"""Llama-style transformer with NSA attention. Wires the combined NSA
forward into the same block structure as llama_dense.py. The Triton
kernel paths live in nsa.triton.*; this module imports the reference
path so that train.py smoke tests can run on CPU.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nsa.model.config import TransformerConfig
from nsa.model.llama_dense import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    _apply_rope,
    _build_rope_cache,
)
from nsa.reference import NSAConfig, nsa_attention_reference


def _nsa_config_from_transformer(cfg: TransformerConfig) -> NSAConfig:
    return NSAConfig(
        block_size_c=cfg.nsa_block_size_c,
        block_size_n=cfg.nsa_block_size_n,
        block_size_m=cfg.nsa_block_size_m,
        top_k=cfg.nsa_top_k,
        window_size=cfg.nsa_window_size,
        causal=True,
        pool=cfg.nsa_pool,
        gate_activation=cfg.nsa_gate_activation,
    )


class NSAAttention(nn.Module):
    """NSA combined attention. The forward dispatches to either:
      - the reference implementation (CPU + initial GPU smoke), or
      - the Triton fused forward (nsa.triton.forward.combined).

    The attention operator is selected at module-init via cfg.attention;
    this class is only constructed when cfg.attention == "nsa".
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.nsa_cfg = _nsa_config_from_transformer(cfg)
        H, D = cfg.n_heads, cfg.head_dim
        hidden = cfg.hidden_size
        self.q_proj = nn.Linear(hidden, H * D, bias=False)
        self.k_proj = nn.Linear(hidden, H * D, bias=False)
        self.v_proj = nn.Linear(hidden, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, hidden, bias=False)
        # Per-head per-token gate logits over the three branches.
        self.gate = nn.Linear(hidden, H * 3, bias=False)
        # Learned compressed-pool projections (only used when pool == "learned").
        if cfg.nsa_pool == "learned":
            self.pool_proj_k = nn.Parameter(torch.randn(cfg.nsa_block_size_c, D) / D**0.5)
            self.pool_proj_v = nn.Parameter(torch.randn(cfg.nsa_block_size_c, D) / D**0.5)
        else:
            self.pool_proj_k = None
            self.pool_proj_v = None

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape
        H, D = self.cfg.n_heads, self.cfg.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)

        gate_logits = self.gate(x).view(B, T, H, 3).permute(0, 2, 1, 3).contiguous()  # [B, H, T, 3]

        if x.is_cuda:
            from nsa.triton.forward import nsa_forward
            # The Triton kernels require fp16 or bf16. RoPE upcasts to fp32
            # under autocast (mul is not on bf16's downcast list); pin to
            # bf16 for the kernel call, then cast back to the residual
            # stream's dtype so the o_proj works regardless of autocast.
            kdtype = torch.bfloat16
            out = nsa_forward(
                q.to(kdtype), k.to(kdtype), v.to(kdtype),
                self.nsa_cfg, gate_logits=gate_logits.to(kdtype),
            ).to(x.dtype)
        else:
            out = nsa_attention_reference(
                q, k, v, self.nsa_cfg,
                gates=gate_logits,
                pool_proj_k=self.pool_proj_k,
                pool_proj_v=self.pool_proj_v,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class LlamaNSA(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.attention == "nsa"
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, NSAAttention(cfg)) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if cfg.tied_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self._rope_cache: Optional[tuple[Tensor, Tensor]] = None

    def _rope(self, T: int, device, dtype) -> tuple[Tensor, Tensor]:
        if self._rope_cache is None or self._rope_cache[0].shape[0] < T or self._rope_cache[0].device != device:
            self._rope_cache = _build_rope_cache(
                max(T, self.cfg.max_position_embeddings), self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        cos, sin = self._rope_cache
        return cos[:T], sin[:T]

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embed(ids)
        cos, sin = self._rope(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        if self.lm_head is not None:
            return self.lm_head(x)
        return x @ self.embed.weight.T
