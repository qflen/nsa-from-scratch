"""Peak-memory benchmark across seq_len for nsa, fa3, full_sdpa.

For each (impl, seq_len), runs a forward and reads `torch.cuda.
max_memory_allocated()`. Where full_sdpa OOMs, the cell is recorded as
{"peak_gb": null, "note": "OOM"}.

Usage:
    python -m nsa.bench.memory --seq-lens 1024,2048,4096,8192,16384,32768,65536 \\
        --impls nsa,fa3,full_sdpa --out runs/memory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nsa.reference import NSAConfig
from nsa.triton.forward import nsa_forward


def _measure_peak(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    del out
    return peak_bytes / 1024**3


def make_qkv(B, H, T, D, dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    return Q, K, V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="1024,2048,4096,8192,16384,32768,65536")
    p.add_argument("--impls", default="nsa,fa3,full_sdpa")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--out", default="runs/memory.json")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    impls = args.impls.split(",")
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cfg = NSAConfig()

    rows = []
    for T in seq_lens:
        for impl in impls:
            try:
                Q, K, V = make_qkv(args.batch, args.heads, T, args.head_dim, dtype)
                if impl == "full_sdpa":
                    fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
                elif impl == "naive":
                    def fn():
                        scores = torch.einsum("bhqd,bhkd->bhqk", Q.float(), K.float()) / (Q.shape[-1] ** 0.5)
                        T = scores.shape[-1]
                        i = torch.arange(T, device="cuda").view(1, 1, T, 1)
                        j = torch.arange(T, device="cuda").view(1, 1, 1, T)
                        scores = scores.masked_fill(j > i, float("-inf"))
                        weights = torch.softmax(scores, dim=-1).to(V.dtype)
                        return torch.einsum("bhqk,bhkd->bhqd", weights, V)
                elif impl == "fa3":
                    from flash_attn import flash_attn_func
                    Qf = Q.transpose(1, 2).contiguous()
                    Kf = K.transpose(1, 2).contiguous()
                    Vf = V.transpose(1, 2).contiguous()
                    fn = lambda: flash_attn_func(Qf, Kf, Vf, causal=True)
                elif impl == "nsa":
                    fn = lambda: nsa_forward(Q, K, V, cfg)
                else:
                    raise ValueError(impl)
                peak = _measure_peak(fn)
                rows.append({"impl": impl, "seq_len": T, "peak_gb": peak})
                print(f"{impl:>10s} T={T:>6d}  peak {peak:>5.2f} GB")
            except torch.cuda.OutOfMemoryError:
                rows.append({"impl": impl, "seq_len": T, "peak_gb": None, "note": "OOM"})
                print(f"{impl:>10s} T={T:>6d}  OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"impl": impl, "seq_len": T, "peak_gb": None, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                print(f"{impl:>10s} T={T:>6d}  ERROR: {type(e).__name__}: {str(e)[:60]}")
                torch.cuda.empty_cache()
            del Q, K, V
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
