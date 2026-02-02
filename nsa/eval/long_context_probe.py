"""Long-context inference probe (256k / 1M) for NSA-100M. The trained
context is 32k, so two RoPE-extension variants are reported: "straight"
(reuse base=10000, honestly degraded) and "ntk-aware" (base scaled by
(T / T_train) ** (D / (D - 2)), no fine-tune).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from nsa.model.config import TransformerConfig
from nsa.model.llama_nsa import LlamaNSA
from nsa.model.llama_dense import _build_rope_cache
from nsa.train.data import packed_token_stream

logger = logging.getLogger(__name__)


def _load_run(run_dir: str):
    p = Path(run_dir)
    ckpt = torch.load(p / "model_final.pt", map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    cfg = TransformerConfig(**raw["arch"])
    model = LlamaNSA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.cuda(), cfg, raw


def _set_rope(model, T: int, rope_theta: float, dtype):
    """Override the model's RoPE cache for an explicit (theta, length)."""
    D = model.cfg.head_dim
    cos, sin = _build_rope_cache(T, D, rope_theta, "cuda", dtype)
    model._rope_cache = (cos, sin)


def _eval_at(model, cfg, tokenizer, seq_len: int, num_sequences: int,
             skip_examples: int, mode: str) -> dict:
    T_train = cfg.max_position_embeddings
    if mode == "ntk":
        D = cfg.head_dim
        base = cfg.rope_theta * (max(1.0, seq_len / T_train) ** (D / (D - 2)))
    else:
        base = cfg.rope_theta
    _set_rope(model, seq_len + 1, base, torch.float32)

    stream = packed_token_stream(tokenizer, seq_len, skip_examples=skip_examples, seed=2025)
    total_loss = 0.0
    total_tokens = 0
    n_seq = 0
    for chunk in stream:
        if n_seq >= num_sequences:
            break
        x = chunk[:seq_len].unsqueeze(0).cuda()
        y = chunk[1: seq_len + 1].unsqueeze(0).cuda()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size).float(),
                    y.view(-1),
                    reduction="sum",
                )
        total_loss += float(loss.item())
        total_tokens += y.numel()
        n_seq += 1
        torch.cuda.empty_cache()

    if total_tokens == 0:
        return {"seq_len": seq_len, "mode": mode, "skipped": True}

    mean_ce = total_loss / total_tokens
    return {
        "seq_len": seq_len,
        "mode": mode,
        "rope_theta": base,
        "mean_ce": mean_ce,
        "perplexity": math.exp(mean_ce),
        "n_sequences": n_seq,
        "n_tokens": total_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seq_lens", default="65536,262144,1048576")
    p.add_argument("--num_sequences", type=int, default=2)
    p.add_argument("--skip_examples", type=int, default=10_500_000)
    p.add_argument("--modes", default="straight,ntk")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    model, cfg, raw = _load_run(args.run_dir)
    logger.info("loaded %s (~%dM params), trained at %dk context",
                cfg.attention, cfg.n_params_estimate // 1_000_000,
                cfg.max_position_embeddings // 1024)

    from transformers import AutoTokenizer
    tok_name = raw.get("tokenizer", "huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, use_fast=True, token=os.environ.get("HF_TOKEN"),
    )

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    modes = args.modes.split(",")
    results = []
    for seq_len in seq_lens:
        for mode in modes:
            logger.info("eval seq_len=%d, mode=%s", seq_len, mode)
            try:
                cell = _eval_at(model, cfg, tokenizer, seq_len,
                                args.num_sequences, args.skip_examples, mode)
            except torch.cuda.OutOfMemoryError as e:
                cell = {"seq_len": seq_len, "mode": mode, "skipped": True,
                        "reason": "OOM: " + str(e)[:200]}
            except Exception as e:
                cell = {"seq_len": seq_len, "mode": mode, "skipped": True,
                        "reason": type(e).__name__ + ": " + str(e)[:200]}
            results.append(cell)
            if "perplexity" in cell:
                logger.info("  seq_len=%d mode=%s  ce=%.4f  ppl=%.2f",
                            seq_len, mode, cell["mean_ce"], cell["perplexity"])
            else:
                logger.info("  seq_len=%d mode=%s  skipped: %s",
                            seq_len, mode, cell.get("reason", "?"))
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_dir": args.run_dir,
        "model_trained_at": cfg.max_position_embeddings,
        "results": results,
    }, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
