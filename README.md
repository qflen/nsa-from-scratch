# nsa-from-scratch

A from-scratch reimplementation of DeepSeek's Native Sparse Attention (Yuan et al., February 2025, [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)).

All three NSA branches (compressed, selected, sliding window) implemented as Triton kernels. The selected branch is additionally implemented in CUDA C++ on Hopper using WGMMA. Multi-precision: FP16, BF16, FP8. Validated by training a five-model fleet (NSA-100M, NSA-150M, NSA-300M, dense-100M baseline, NSA-100M at 64k context) and a perplexity sweep on the long-context-trained NSA-100M.

## Headline

- **NSA forward is 7.4x FlashAttention-3 at 64k context** on H100 NVL, batch 1, head dim 64, bf16. NSA flattens at roughly 12M tok/s from 8k onward; FA-3 drops off quadratically.
- **NSA-100M perplexity stays essentially flat from 2k to 32k evaluation context** (range 66.5 to 70.8 on a held-out FineWeb-Edu split). The long-context-stability claim of NSA lands at training-budget scale.
- **NSA trains stably at 64k context** at 100M parameters, where dense full attention OOMs the H100 80 GB even at this small a model.

Plots: `writeup/figures/01_throughput.png`, `02_memory.png`, `03_loss_curves.png`, `04_perplexity_vs_ctx.png`, `05_scaling_trend.png`, `06_longbench.png`. Full writeup at `writeup/post.md`.

## Reproducing the benchmarks

```bash
pip install -e '.[dev,train]'
pytest tests/ -q                              # 76 tests on H100 NVL, 63 on a 4090
python -m nsa.bench.throughput --seq-lens 1024,2048,4096,8192,16384,32768,65536 \
    --impls nsa,fa3,full_sdpa --dtype bf16 --out runs/throughput.json
```

The throughput benchmark needs a Hopper card (the WGMMA selected forward and the FA-3 baseline both require sm_90a). On a 4090 the suite still runs; the Hopper-only tests skip cleanly.

## Reproducing the training runs

Configurations live in `nsa/train/config_*.yaml`. The five recipes shipped:

```bash
# NSA scaling points, all 32k context
python -m nsa.train.train --config nsa/train/config_100m.yaml         # NSA-100M, 1B tokens
python -m nsa.train.train --config nsa/train/config_150m.yaml         # NSA-150M, 1B tokens
python -m nsa.train.train --config nsa/train/config_300m.yaml         # NSA-300M, 500M tokens

# Dense baseline at the largest workable context (8k on H100 PCIe at this batch shape)
python -m nsa.train.train --config nsa/train/config_dense_100m.yaml

# Long-context training demonstration (64k context, 100M parameters)
python -m nsa.train.train --config nsa/train/config_100m_64k.yaml
```

Each run logs to wandb project `nsa-from-scratch` and saves checkpoints to `runs/<run_id>/`. The token counts above are below Chinchilla-optimal because the autograd-aware bwd through the compressed and sliding branches dominates training throughput at 32k context; the writeup discusses this and the natural follow-up (hand-written Triton sliding and compressed backward kernels).

## Reproducing the perplexity eval

```bash
python -m nsa.eval.perplexity \
    --run_dir runs/nsa-100m-1b-32k \
    --seq_lens 2048,4096,8192,16384,32768 \
    --num_sequences_per_len 32 \
    --out writeup/figures/data/perplexity_nsa-100m-1b-32k.json
```

The eval streams a deterministic offset of FineWeb-Edu, packs into the target seq_len, and reports cross-entropy and perplexity per cell.

## Layout

```
nsa/
  reference.py          plain-torch reference for the three branches (correctness oracle)
  triton/               compressed.py, selected.py, sliding.py, gating.py, forward.py, backward.py, fp8.py
  cuda/                 selected_fwd.cu (Hopper WGMMA), selected_bwd.cu (in-progress), bindings.cpp
  model/                llama_nsa.py, llama_dense.py, config.py
  train/                train.py, data.py, config_*.yaml (NSA-100M, 150M, 300M, dense, 64k, sanity)
  eval/                 perplexity.py
  bench/                throughput.py, memory.py, correctness.py, autotune.py, plots.py
tests/                  test_{compressed,selected,sliding,combined,fp8,training_step,cuda_selected,cuda_selected_bwd}_forward|backward.py
writeup/                post.md, figures/{01..06}.png
notes/                  refs.bib
scripts/                fetch_wandb.py, make_plots.py
```

## License

Apache-2.0. See LICENSE.
