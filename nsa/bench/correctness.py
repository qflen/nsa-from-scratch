"""Correctness sweep across (impl, precision, seq_len) cells against
the fp32 reference. Tol: fp16 / bf16 rel_err < 1e-2, fp8 rel_err < 1e-1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nsa.reference import (
    NSAConfig,
    attention_reference,
    nsa_attention_reference,
)


_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    # fp8 needs a custom path; CUDA-only.
}


def _max_rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.float() - ref.float()).abs().max()
    denom = ref.float().abs().max() + 1e-6
    return float(diff / denom)


def sweep(impls: list[str], precisions: list[str], seq_lens: list[int], head_dim: int = 64) -> list[dict]:
    rows: list[dict] = []
    cfg = NSAConfig(block_size_c=64, block_size_n=64, block_size_m=64, top_k=16, window_size=512)
    for impl in impls:
        for prec in precisions:
            dtype = _DTYPES[prec]
            for T in seq_lens:
                Q = torch.randn(1, 4, T, head_dim, device="cuda", dtype=dtype)
                K = torch.randn(1, 4, T, head_dim, device="cuda", dtype=dtype)
                V = torch.randn(1, 4, T, head_dim, device="cuda", dtype=dtype)
                ref, _ = attention_reference(Q.float(), K.float(), V.float(), causal=True)
                if impl == "reference":
                    out = nsa_attention_reference(Q, K, V, cfg)
                elif impl == "nsa_triton":
                    from nsa.triton.forward import nsa_forward  # imported lazily, may not yet exist
                    out = nsa_forward(Q, K, V, cfg)
                else:
                    # No full-NSA CUDA forward exists (only the selected branch is
                    # ported to CUDA); see tests/test_cuda_selected.py for its correctness.
                    raise ValueError(impl)
                rel = _max_rel(out, ref)
                rows.append({"impl": impl, "precision": prec, "seq_len": T, "max_rel_err": rel})
                print(f"{impl:14s} {prec:4s} T={T:6d}  max_rel_err={rel:.3e}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impls", default="reference")
    p.add_argument("--precisions", default="fp16,bf16")
    p.add_argument("--seq-lens", default="1024,4096,16384")
    p.add_argument("--out", default="bench/results/correctness.json")
    args = p.parse_args()

    rows = sweep(
        args.impls.split(","),
        args.precisions.split(","),
        [int(x) for x in args.seq_lens.split(",")],
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
