"""Long-context perplexity over a held-out FineWeb-Edu slice. Forward-
only bf16, packs to the target seq_len with EOT separators, reports
exp(mean_ce). LongBench wiring lives in lm_eval_setup.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from nsa.model.config import TransformerConfig
from nsa.model.llama_dense import LlamaDense
from nsa.model.llama_nsa import LlamaNSA
from nsa.train.data import packed_token_stream

logger = logging.getLogger(__name__)


def _load_run(run_dir: str) -> tuple[torch.nn.Module, TransformerConfig, dict]:
    run_dir_p = Path(run_dir)
    ckpt_path = run_dir_p / "model_final.pt"
    if not ckpt_path.exists():
        # fall back to the latest checkpoint if training was stopped early
        ckpts = sorted(run_dir_p.glob("ckpt_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoint in {run_dir_p}")
        ckpt_path = ckpts[-1]
        logger.info("using latest checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    arch = raw["arch"]
    cfg = TransformerConfig(**arch)
    if cfg.attention == "dense":
        model = LlamaDense(cfg)
    else:
        model = LlamaNSA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.cuda(), cfg, raw


def _eval_one_seq_len(
    model: torch.nn.Module,
    cfg: TransformerConfig,
    tokenizer,
    seq_len: int,
    num_sequences: int,
    skip_examples: int,
) -> dict:
    """Stream `num_sequences` packed sequences of length `seq_len` from the
    held-out FineWeb-Edu split (offset by `skip_examples` from the training
    stream), run forward-only in bf16, and return mean CE + perplexity.
    """
    if seq_len > cfg.max_position_embeddings:
        return {
            "seq_len": seq_len,
            "skipped": True,
            "reason": f"seq_len > model.max_position_embeddings ({cfg.max_position_embeddings})",
        }

    stream: Iterator = packed_token_stream(
        tokenizer, seq_len, skip_examples=skip_examples, seed=99,
    )

    total_loss = 0.0
    total_tokens = 0
    seq_count = 0

    with torch.no_grad():
        for chunk in stream:
            if seq_count >= num_sequences:
                break
            x = chunk[:seq_len].unsqueeze(0).cuda(non_blocking=True)
            y = chunk[1: seq_len + 1].unsqueeze(0).cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size).float(),
                    y.view(-1),
                    reduction="sum",
                )
            total_loss += float(loss.item())
            total_tokens += y.numel()
            seq_count += 1

    if total_tokens == 0:
        return {"seq_len": seq_len, "skipped": True, "reason": "no sequences"}

    mean_ce = total_loss / total_tokens
    return {
        "seq_len": seq_len,
        "mean_ce": mean_ce,
        "perplexity": math.exp(mean_ce),
        "n_sequences": seq_count,
        "n_tokens": total_tokens,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="path to /workspace/runs/<run_id>")
    p.add_argument(
        "--seq_lens",
        default="2048,4096,8192,16384,32768",
        help="comma-separated eval seq lens",
    )
    p.add_argument(
        "--num_sequences_per_len", type=int, default=64,
        help="number of packed sequences per (model, seq_len) cell",
    )
    p.add_argument(
        "--skip_examples", type=int, default=10_000_000,
        help="offset into FineWeb-Edu so we hit a held-out portion",
    )
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    model, cfg, raw = _load_run(args.run_dir)
    logger.info("loaded %s (~%dM params)", cfg.attention, cfg.n_params_estimate // 1_000_000)

    from transformers import AutoTokenizer
    tok_name = raw.get("tokenizer", "huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, use_fast=True, token=os.environ.get("HF_TOKEN"),
    )

    seq_lens = [int(s) for s in args.seq_lens.split(",")]
    results = []
    for seq_len in seq_lens:
        logger.info("evaluating seq_len=%d", seq_len)
        cell = _eval_one_seq_len(
            model, cfg, tokenizer, seq_len,
            num_sequences=args.num_sequences_per_len,
            skip_examples=args.skip_examples,
        )
        results.append(cell)
        if "perplexity" in cell:
            logger.info(
                "  seq_len=%d  ce=%.4f  ppl=%.4f  (n_seq=%d, n_tok=%d)",
                seq_len, cell["mean_ce"], cell["perplexity"],
                cell["n_sequences"], cell["n_tokens"],
            )
        else:
            logger.info("  seq_len=%d  skipped: %s", seq_len, cell["reason"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "tokenizer": tok_name,
        "n_params_estimate": cfg.n_params_estimate,
        "attention": cfg.attention,
        "model_max_pos": cfg.max_position_embeddings,
        "skip_examples": args.skip_examples,
        "results": results,
    }, indent=2))
    logger.info("wrote results to %s", out_path)


if __name__ == "__main__":
    main()
