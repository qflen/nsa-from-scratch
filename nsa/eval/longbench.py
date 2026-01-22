"""Likelihood-based LongBench v2 eval: per-task NLL of the gold answer
given the long-context prompt. Skips generation to stay inside budget,
so not directly comparable to LongBench F1 / ROUGE. Tasks: narrativeqa,
qasper, multifieldqa_en, gov_report, qmsum, lcc.
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

logger = logging.getLogger(__name__)


TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "gov_report",
    "qmsum",
    "lcc",
]

QA_TEMPLATE = (
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer: "
)
SUMMARY_TEMPLATE = (
    "{context}\n\n"
    "Summary: "
)
COMPLETION_TEMPLATE = "{context}"


def _load_run(run_dir: str):
    p = Path(run_dir)
    ckpt = torch.load(p / "model_final.pt", map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    cfg = TransformerConfig(**raw["arch"])
    model = LlamaNSA(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.cuda(), cfg, raw


def _score_example(model, cfg, tokenizer, prompt: str, answer: str) -> dict:
    """Compute mean NLL over answer tokens given prompt as context.

    Truncates prompt so prompt + answer fits within max_position_embeddings.
    """
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(answer_ids) == 0:
        return {"skipped": True, "reason": "empty answer"}

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_len = cfg.max_position_embeddings - len(answer_ids) - 1
    if max_len <= 0:
        return {"skipped": True, "reason": "answer longer than context"}
    prompt_ids = prompt_ids[-max_len:]

    full_ids = prompt_ids + answer_ids
    x = torch.tensor(full_ids[:-1], dtype=torch.long, device="cuda").unsqueeze(0)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device="cuda").unsqueeze(0)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)

    answer_start = len(prompt_ids) - 1  # y is shifted left by 1
    answer_token_log_probs = token_log_probs[0, answer_start:]
    if answer_token_log_probs.numel() == 0:
        return {"skipped": True, "reason": "answer slice empty"}
    mean_nll = float(-answer_token_log_probs.mean().item())
    return {
        "prompt_len": len(prompt_ids),
        "answer_len": len(answer_ids),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def _build_prompt(example: dict, task_name: str) -> tuple[str, str]:
    """Return (prompt, answer) for one LongBench example."""
    context = example.get("context") or ""
    question = example.get("input") or ""
    answers = example.get("answers") or []
    if isinstance(answers, list):
        answer = answers[0] if answers else ""
    else:
        answer = answers
    if task_name in ("gov_report", "qmsum"):
        prompt = SUMMARY_TEMPLATE.format(context=context)
    elif task_name == "lcc":
        prompt = COMPLETION_TEMPLATE.format(context=context)
    else:
        prompt = QA_TEMPLATE.format(context=context, question=question)
    return prompt, answer


def _eval_task(model, cfg, tokenizer, task_name: str, data_dir: str,
               max_examples: int) -> dict:
    import json as _json
    task_path = Path(data_dir) / f"{task_name}.jsonl"
    if not task_path.exists():
        return {"task": task_name, "skipped": True, "reason": f"missing {task_path}"}

    scores = []
    n_skipped = 0
    with task_path.open() as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            example = _json.loads(line)
            prompt, answer = _build_prompt(example, task_name)
            if not isinstance(answer, str) or not answer.strip():
                n_skipped += 1
                continue
            result = _score_example(model, cfg, tokenizer, prompt, answer)
            if result.get("skipped"):
                n_skipped += 1
                continue
            scores.append(result)

    if not scores:
        return {"task": task_name, "skipped": True, "n_skipped": n_skipped}

    return {
        "task": task_name,
        "n_examples": len(scores),
        "n_skipped": n_skipped,
        "mean_nll": sum(s["mean_nll"] for s in scores) / len(scores),
        "mean_perplexity": sum(s["perplexity"] for s in scores) / len(scores),
        "median_nll": sorted(s["mean_nll"] for s in scores)[len(scores) // 2],
        "mean_prompt_len": sum(s["prompt_len"] for s in scores) / len(scores),
        "mean_answer_len": sum(s["answer_len"] for s in scores) / len(scores),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--max_examples", type=int, default=24)
    p.add_argument("--data_dir", default="/workspace/longbench-data/data")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    model, cfg, raw = _load_run(args.run_dir)
    logger.info("loaded %s (~%dM params)", cfg.attention, cfg.n_params_estimate // 1_000_000)

    from transformers import AutoTokenizer
    tok_name = raw.get("tokenizer", "huggyllama/llama-7b")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, use_fast=True, token=os.environ.get("HF_TOKEN"),
    )

    results = []
    for task_name in TASKS:
        logger.info("evaluating %s", task_name)
        cell = _eval_task(model, cfg, tokenizer, task_name, args.data_dir, args.max_examples)
        results.append(cell)
        if "mean_nll" in cell:
            logger.info(
                "  %s: nll %.3f  ppl %.2f  (n=%d, mean_prompt_len=%d)",
                task_name, cell["mean_nll"], cell["mean_perplexity"],
                cell["n_examples"], int(cell["mean_prompt_len"]),
            )
        else:
            logger.info("  %s: skipped (%s)", task_name, cell.get("reason", "?"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "run_dir": args.run_dir,
        "scoring": "answer-token NLL given long-context prompt (likelihood, not generation)",
        "max_examples_per_task": args.max_examples,
        "model_max_pos": cfg.max_position_embeddings,
        "n_params_estimate": cfg.n_params_estimate,
        "results": results,
    }, indent=2))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
