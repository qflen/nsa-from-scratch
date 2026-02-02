"""Llama-style transformer with full (dense) attention. Baseline for the
training comparison. Shares everything with llama_nsa.py except the attention
operator: this one calls torch.nn.functional.scaled_dot_product_attention
(which dispatches to FA-3 on Hopper when available).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nsa.model.config import TransformerConfig


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden, intermediate, bias=False)
        self.w_up = nn.Linear(hidden, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DenseAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        H, D = cfg.n_heads, cfg.head_dim
        hidden = cfg.hidden_size
        self.q_proj = nn.Linear(hidden, H * D, bias=False)
        self.k_proj = nn.Linear(hidden, H * D, bias=False)
        self.v_proj = nn.Linear(hidden, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, hidden, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        # SDPA dispatches to FA-3 on Hopper when available.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, attn: nn.Module):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = attn
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaDense(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.attention == "dense"
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg, DenseAttention(cfg)) for _ in range(cfg.n_layers)]
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
        return x @ self.embed.weight.T  # tied
