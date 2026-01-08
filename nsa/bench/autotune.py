"""Autotune sweep for the selected-branch Triton forward. Times a grid
of (BLOCK_M, BLOCK_N, num_warps, num_stages) on a realistic shape;
writes JSON consumed by the writeup. The grid focuses on tiling and
pipeline depth since the kernel is bandwidth-bound.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from nsa.triton.forward import _resolve_block_indices
from nsa.triton.selected import selected_attention


_DEFAULT_GRID = {
    "BLOCK_M": [32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}


def _bench_one(Q, K, V, *, block_size_m, block_size_n, top_k, block_indices,
               num_warps, num_stages, iters, warmup) -> dict:
    """Time one (config) point. Returns median wall time in ms; None on OOM/compile error."""
    try:
        for _ in range(warmup):
            _ = selected_attention(
                Q, K, V,
                block_size_m=block_size_m, block_size_n=block_size_n, top_k=top_k,
                block_indices=block_indices, causal=True,
                num_warps=num_warps, num_stages=num_stages,
            )
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.time()
            _ = selected_attention(
                Q, K, V,
                block_size_m=block_size_m, block_size_n=block_size_n, top_k=top_k,
                block_indices=block_indices, causal=True,
                num_warps=num_warps, num_stages=num_stages,
            )
            torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        times.sort()
        return {"ok": True, "median_ms": times[len(times) // 2], "min_ms": times[0]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}


def sweep(shape: tuple[int, int, int, int, int, int, int], dtype, *,
          iters: int = 30, warmup: int = 5,
          grid: dict | None = None) -> list[dict]:
    """shape: (B, H, Tq, Tk, D, top_k, block_size_n_default).

    BLOCK_N in the grid overrides the default; the gather pattern uses the
    grid's BLOCK_N too so different configs are comparable. BLOCK_M can be
    set independently.
    """
    B, H, Tq, Tk, D, top_k, _ = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(B, H, Tq, D, device="cuda", dtype=dtype, generator=g)
    K = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)
    V = torch.randn(B, H, Tk, D, device="cuda", dtype=dtype, generator=g)

    grid = grid or _DEFAULT_GRID
    rows: list[dict] = []
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"sweeping {len(combos)} configs at shape {shape}, dtype={dtype}")

    for idx, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        block_size_m = cfg["BLOCK_M"]
        block_size_n = cfg["BLOCK_N"]
        # Recompute block_indices for each (BLOCK_M, BLOCK_N) so the
        # gather pattern is consistent with that tile size.
        scale = 1.0 / (D ** 0.5)
        block_indices = _resolve_block_indices(
            Q, K, block_size_m=block_size_m, block_size_n=block_size_n,
            top_k=top_k, causal=True, scale=scale,
        )
        result = _bench_one(
            Q, K, V,
            block_size_m=block_size_m, block_size_n=block_size_n, top_k=top_k,
            block_indices=block_indices,
            num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
            iters=iters, warmup=warmup,
        )
        row = {**cfg, **result}
        rows.append(row)
        if result["ok"]:
            print(f"  [{idx+1:>3}/{len(combos)}] {cfg} median={result['median_ms']:.3f} ms")
        else:
            print(f"  [{idx+1:>3}/{len(combos)}] {cfg} FAIL: {result['error']}")
        torch.cuda.empty_cache()

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="1,8,4096,8192,64,16,64",
                   help="B,H,Tq,Tk,D,top_k,block_size_n_default")
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", default="runs/autotune.json")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = tuple(int(x) for x in args.shape.split(","))
    rows = sweep(shape, dtype, iters=args.iters, warmup=args.warmup)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "shape": list(shape),
        "dtype": args.dtype,
        "rows": rows,
    }, indent=2))

    valid = [r for r in rows if r.get("ok")]
    if not valid:
        print("\nNO valid configs!")
        return
    valid.sort(key=lambda r: r["median_ms"])
    print(f"\nTop 5 configs (out of {len(valid)} valid):")
    for r in valid[:5]:
        print(f"  median={r['median_ms']:.3f} ms  "
              f"BLOCK_M={r['BLOCK_M']:>3} BLOCK_N={r['BLOCK_N']:>3} "
              f"warps={r['num_warps']} stages={r['num_stages']}")


if __name__ == "__main__":
    main()
