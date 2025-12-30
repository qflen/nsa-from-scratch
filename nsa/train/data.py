"""FineWeb-Edu streaming + on-the-fly tokenization. Falls back to FineWeb
sample-10BT or SlimPajama-627B if FineWeb-Edu is unreachable on the pod.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Iterator, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_DATASET_CANDIDATES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("HuggingFaceFW/fineweb", "sample-10BT"),
    ("cerebras/SlimPajama-627B", None),
]


def _open_stream(token: Optional[str] = None):
    from datasets import load_dataset
    last_err: Optional[Exception] = None
    for name, config in _DATASET_CANDIDATES:
        try:
            ds = load_dataset(name, config, split="train", streaming=True, token=token)
            logger.info("loaded streaming dataset: %s/%s", name, config)
            return ds, name
        except Exception as e:  # network, auth, dataset moved
            logger.warning("dataset %s/%s unavailable: %s", name, config, e)
            last_err = e
    raise RuntimeError(f"no usable dataset (last error: {last_err})")


def packed_token_stream(
    tokenizer,
    seq_len: int,
    *,
    text_field: str = "text",
    eot_token_id: Optional[int] = None,
    seed: int = 0,
    skip_examples: int = 0,
) -> Iterator[Tensor]:
    """Yield int64 tensors of shape [seq_len + 1] (input_ids and shifted labels share storage).

    Strategy: stream raw text, tokenize per-document, append eot, concatenate
    into a rolling buffer, slice off seq_len+1 tokens at a time. This is the
    "packed" pretraining recipe used by GPT-NeoX, Llama, etc.
    """
    ds, _ = _open_stream(token=os.environ.get("HF_TOKEN"))
    if seed:
        ds = ds.shuffle(seed=seed, buffer_size=10000)
    if skip_examples:
        ds = ds.skip(skip_examples)

    if eot_token_id is None:
        eot_token_id = tokenizer.eos_token_id
        if eot_token_id is None:
            eot_token_id = tokenizer.pad_token_id or 0

    buf: list[int] = []
    for example in ds:
        text = example.get(text_field) or ""
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(eot_token_id)
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len:]  # keep last token as the next chunk's first.
            yield torch.tensor(chunk, dtype=torch.long)


def make_loader(tokenizer, seq_len: int, micro_batch: int, **kwargs) -> Iterator[tuple[Tensor, Tensor]]:
    """Wrap packed_token_stream into (input_ids, labels) batches of size micro_batch.

    Labels are inputs shifted left by one: the loss is computed on tokens
    [1:seq_len+1] with the last token of the chunk as the final label.
    """
    stream = packed_token_stream(tokenizer, seq_len, **kwargs)
    while True:
        batch = list(itertools.islice(stream, micro_batch))
        if len(batch) < micro_batch:
            return
        x = torch.stack([t[:-1] for t in batch], dim=0)
        y = torch.stack([t[1:] for t in batch], dim=0)
        yield x, y
