"""Throughput benchmark across seq_len in [1k..64k] for nsa, fa3, full_sdpa.

Measures forward-only tokens-per-second for each impl, averaged over
`iters` after `warmup` warmup runs. Writes a JSON file consumed by
`writeup/figures/plot.py` to render plot 1 (throughput vs seq_len).

Usage:
    python -m nsa.bench.throughput \\
        --seq-lens 1024,2048,4096,8192,16384,32768,65536 \\
        --impls nsa,fa3,full_sdpa --dtype bf16 \\
        --out runs/throughput.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.reference import NSAConfig
from nsa.triton.forward import nsa_forward


def _bench(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def make_qkv(B: int, H: int, T: int, D: int, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    return Q, K, V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="1024,2048,4096,8192,16384,32768,65536")
    p.add_argument("--impls", default="nsa,fa3,full_sdpa,moba")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="runs/throughput.json")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    impls = args.impls.split(",")
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cfg = NSAConfig()

    rows: list[dict] = []
    for T in seq_lens:
        for impl in impls:
            try:
                Q, K, V = make_qkv(args.batch, args.heads, T, args.head_dim, dtype)
                if impl == "full_sdpa":
                    fn = lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
                elif impl == "fa3":
                    from flash_attn import flash_attn_func
                    Qf = Q.transpose(1, 2).contiguous()
                    Kf = K.transpose(1, 2).contiguous()
                    Vf = V.transpose(1, 2).contiguous()
                    fn = lambda: flash_attn_func(Qf, Kf, Vf, causal=True)
                elif impl == "nsa":
                    fn = lambda: nsa_forward(Q, K, V, cfg)
                elif impl == "moba":
                    from nsa.triton.moba import moba_attention
                    fn = lambda: moba_attention(
                        Q, K, V, block_size_n=64, block_size_m=64,
                        top_k_per_query=16, top_k_cap=32, causal=True,
                    )[0]
                else:
                    raise ValueError(impl)

                elapsed = _bench(fn, iters=args.iters, warmup=args.warmup)
                tps = args.batch * T / elapsed
                rows.append({
                    "impl": impl, "seq_len": T, "tokens_per_sec": tps,
                    "ms_per_call": elapsed * 1000, "iters": args.iters,
                })
                print(f"{impl:>10s} T={T:>6d}  {elapsed*1000:>7.2f} ms  {tps:>10,.0f} tok/s")
            except torch.cuda.OutOfMemoryError:
                rows.append({"impl": impl, "seq_len": T, "tokens_per_sec": None, "note": "OOM"})
                print(f"{impl:>10s} T={T:>6d}  OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                rows.append({"impl": impl, "seq_len": T, "tokens_per_sec": None, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                print(f"{impl:>10s} T={T:>6d}  ERROR: {type(e).__name__}: {str(e)[:60]}")
                torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
