"""Single-GPU training loop for the NSA-100M, dense-100M, NSA-150M, and
NSA-300M runs. Bf16 mixed precision, fp32 master weights and optimizer
state. AdamW: beta1=0.9, beta2=0.95, weight_decay=0.1, peak LR 3e-4,
min LR 3e-5, cosine schedule with 200-step warmup; grad clip 1.0.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from nsa.model.config import TransformerConfig
from nsa.model.llama_dense import LlamaDense
from nsa.model.llama_nsa import LlamaNSA
from nsa.train.data import make_loader

logger = logging.getLogger(__name__)


def _cosine_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def build_model(cfg: TransformerConfig) -> torch.nn.Module:
    if cfg.attention == "dense":
        return LlamaDense(cfg)
    if cfg.attention == "nsa":
        return LlamaNSA(cfg)
    raise ValueError(cfg.attention)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", default=None, help="overrides config.run_id")
    p.add_argument("--resume", default=None, help="path to checkpoint")
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    run_id = args.run_id or raw["run_id"]
    out_dir = Path(raw.get("out_dir", "runs")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )

    arch = raw["arch"]
    cfg = TransformerConfig(**arch)
    model = build_model(cfg).cuda()
    logger.info("model: %s, ~%dM params", cfg.attention, cfg.n_params_estimate // 1_000_000)

    from transformers import AutoTokenizer
    tok_name = raw.get("tokenizer", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True, token=os.environ.get("HF_TOKEN"))

    seq_len = raw["seq_len"]
    micro_batch = raw["micro_batch"]
    grad_accum = raw["grad_accum"]
    total_tokens = int(raw["total_tokens"])
    tokens_per_step = micro_batch * grad_accum * seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup_steps = raw.get("warmup_steps", 200)
    peak_lr = raw.get("peak_lr", 3e-4)
    floor_lr = raw.get("floor_lr", 3e-5)
    weight_decay = raw.get("weight_decay", 0.1)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"]
    else:
        start_step = 0

    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and raw.get("wandb", True)
    if use_wandb:
        import wandb
        wandb.init(
            project="nsa-from-scratch",
            name=run_id,
            config={**raw, "n_params_estimate": cfg.n_params_estimate},
            tags=raw.get("tags", []),
        )

    loader = make_loader(tokenizer, seq_len, micro_batch)
    model.train()
    t_step = time.time()
    running_loss = 0.0
    grad_norm_accum = 0.0
    save_every = raw.get("save_every", 2000)
    log_every = raw.get("log_every", 10)

    step = start_step
    while step < total_steps:
        lr = _cosine_lr(step, warmup_steps, total_steps, peak_lr, floor_lr)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            try:
                x, y = next(loader)
            except StopIteration:
                logger.warning("loader exhausted at step %d; restarting", step)
                loader = make_loader(tokenizer, seq_len, micro_batch)
                x, y = next(loader)
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size).float(), y.view(-1), reduction="mean"
                )
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optim.step()

        running_loss += loss_acc
        grad_norm_accum += gn
        step += 1

        if step % log_every == 0:
            dt = time.time() - t_step
            tps = log_every * tokens_per_step / dt
            avg_loss = running_loss / log_every
            avg_gn = grad_norm_accum / log_every
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(
                "step %d/%d loss %.4f lr %.2e gn %.3f tok/s %d peak %.1fGB",
                step, total_steps, avg_loss, lr, avg_gn, int(tps), peak_mem,
            )
            if use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": avg_gn,
                    "train/tokens_per_sec": tps,
                    "train/peak_memory_gb": peak_mem,
                    "train/step": step,
                    "train/tokens_seen": step * tokens_per_step,
                })
            running_loss = 0.0
            grad_norm_accum = 0.0
            t_step = time.time()

        if step % save_every == 0 or step == total_steps:
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(), "step": step, "config": raw},
                out_dir / "model_final.pt" if step == total_steps else out_dir / f"ckpt_{step}.pt",
            )

    if use_wandb:
        wandb.finish()
    logger.info("training complete; saved to %s", out_dir)


if __name__ == "__main__":
    main()
