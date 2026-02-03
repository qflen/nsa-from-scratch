"""Memory / OOM frontier sweep across batch_size at fixed seq_lens.

Walks batch by doubling (1, 2, 4, 8, ...) until OOM, then binary-searches
the gap between the last successful batch and the first OOM batch to pin
the precise max-tolerable batch_size for each (impl, seq_len).

Used to back the "dense tolerates batch X at seq Y, NSA tolerates X'"
claim with measured numbers rather than the fixed-batch single-shot peak
that lives in `runs/memory.json`.

Usage:
    python -m nsa.bench.memory_sweep \\
        --seq-lens 4096,8192,16384,32768,65536 \\
        --impls nsa,full_sdpa --dtype bf16 \\
        --max-batch 256 \\
        --out runs/memory_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nsa.reference import NSAConfig
from nsa.triton.forward import nsa_forward


def make_qkv(B, H, T, D, dtype, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, T, D, device="cuda", dtype=dtype, generator=g)
    return Q, K, V


def _build_fn(impl: str, Q, K, V, cfg: NSAConfig):
    if impl == "full_sdpa":
        return lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    if impl == "nsa":
        return lambda: nsa_forward(Q, K, V, cfg)
    if impl == "fa3":
        from flash_attn import flash_attn_func
        Qf = Q.transpose(1, 2).contiguous()
        Kf = K.transpose(1, 2).contiguous()
        Vf = V.transpose(1, 2).contiguous()
        return lambda: flash_attn_func(Qf, Kf, Vf, causal=True)
    raise ValueError(impl)


def _try_one(impl: str, B: int, H: int, T: int, D: int, dtype, cfg: NSAConfig, *, seed: int):
    """Allocate (Q, K, V), run fwd once, return peak GB. Raises OOM on failure."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    Q, K, V = make_qkv(B, H, T, D, dtype, seed=seed)
    fn = _build_fn(impl, Q, K, V, cfg)
    out = fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    del out, Q, K, V
    torch.cuda.empty_cache()
    return peak_bytes / 1024**3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", default="4096,8192,16384,32768,65536")
    p.add_argument("--impls", default="nsa,full_sdpa")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/memory_sweep.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    impls = args.impls.split(",")
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    cfg = NSAConfig()

    rows: list[dict] = []
    for T in seq_lens:
        for impl in impls:
            steps: list[dict] = []
            last_ok = 0
            first_oom: int | None = None

            B = 1
            while B <= args.max_batch:
                try:
                    peak = _try_one(impl, B, args.heads, T, args.head_dim, dtype, cfg, seed=args.seed)
                    steps.append({"batch": B, "peak_gb": peak, "ok": True})
                    print(f"{impl:>10s} T={T:>6d} B={B:>3d}  peak {peak:>5.2f} GB")
                    last_ok = B
                    B *= 2
                except torch.cuda.OutOfMemoryError:
                    steps.append({"batch": B, "peak_gb": None, "ok": False, "note": "OOM"})
                    print(f"{impl:>10s} T={T:>6d} B={B:>3d}  OOM")
                    first_oom = B
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    steps.append({"batch": B, "peak_gb": None, "ok": False, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                    print(f"{impl:>10s} T={T:>6d} B={B:>3d}  ERROR: {type(e).__name__}")
                    first_oom = B
                    torch.cuda.empty_cache()
                    break

            # Binary search between last_ok and first_oom for the precise frontier.
            max_ok = last_ok
            if first_oom is not None and last_ok > 0 and first_oom - last_ok > 1:
                lo, hi = last_ok + 1, first_oom - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    try:
                        peak = _try_one(impl, mid, args.heads, T, args.head_dim, dtype, cfg, seed=args.seed)
                        steps.append({"batch": mid, "peak_gb": peak, "ok": True, "from": "binsearch"})
                        print(f"{impl:>10s} T={T:>6d} B={mid:>3d}  peak {peak:>5.2f} GB  [binsearch]")
                        max_ok = max(max_ok, mid)
                        lo = mid + 1
                    except torch.cuda.OutOfMemoryError:
                        steps.append({"batch": mid, "peak_gb": None, "ok": False, "note": "OOM", "from": "binsearch"})
                        print(f"{impl:>10s} T={T:>6d} B={mid:>3d}  OOM  [binsearch]")
                        torch.cuda.empty_cache()
                        hi = mid - 1
                    except Exception as e:
                        steps.append({"batch": mid, "peak_gb": None, "ok": False, "error": f"{type(e).__name__}", "from": "binsearch"})
                        torch.cuda.empty_cache()
                        hi = mid - 1

            rows.append({
                "impl": impl, "seq_len": T,
                "max_batch_ok": max_ok,
                "first_batch_oom": first_oom,
                "max_batch_tried": args.max_batch,
                "heads": args.heads, "head_dim": args.head_dim,
                "dtype": args.dtype,
                "steps": steps,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} (impl, seq_len) cells to {out_path}")


if __name__ == "__main__":
    main()
