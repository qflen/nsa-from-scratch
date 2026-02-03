# nsa-from-scratch

[![CI](https://img.shields.io/github/actions/workflow/status/qflen/nsa-from-scratch/ci.yml?branch=main&label=CI)](https://github.com/qflen/nsa-from-scratch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton 3.0+](https://img.shields.io/badge/Triton-3.0+-9D4EDD.svg)](https://github.com/openai/triton)
[![CUDA sm_90a](https://img.shields.io/badge/CUDA-sm__90a%20(Hopper)-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv 2502.11089](https://img.shields.io/badge/arXiv-2502.11089-b31b1b.svg)](https://arxiv.org/abs/2502.11089)

From-scratch reimplementation of DeepSeek's Native Sparse Attention (Yuan et al., February 2025, [arXiv:2502.11089](https://arxiv.org/abs/2502.11089)): a sparse-attention design that holds language-model quality at long context while beating FlashAttention-3 in wall-clock time as the sequence grows.

**Full writeup:** [writeup/post.md](writeup/post.md)

## Headline results

| Result | Number | Hardware |
|---|---|---|
| NSA forward at 64k context vs FlashAttention-3 | **7.07x faster** (10.84M vs 1.53M tok/s) | H100 NVL, 94 GB |
| NSA-100M perplexity, 2k to 32k evaluation context | flat (66.5 to 70.8 on a FineWeb-Edu held-out split) | (eval) |
| NSA-100M trains stably at 64k context | 500M tokens, no loss spikes or divergence | A100 SXM, 80 GB |
| Dense full attention at 64k context, same 100M params | OOMs the card at this batch shape | A100 PCIe, 80 GB |

Throughput at 1k to 64k, perplexity at 2k to 256k, training-loss curves for a five-model fleet (NSA-100M, NSA-150M, NSA-300M, dense-100M baseline, NSA-100M at 64k context), LongBench v2 likelihood across six tasks, and a MoBA cross-comparison are all in the [writeup](writeup/post.md).

## What is implemented

- All three NSA branches (compressed, selected, sliding window) as Triton kernels.
- The selected branch additionally implemented in CUDA C++ on Hopper using inline-PTX WGMMA atoms (`m64n64k16` and `m64n128k16` at D=64 and D=128).
- Multi-precision: FP16, BF16, FP8 (per-tensor absmax, bf16 dequant in the wrapper).
- Hand-written Triton selected backward (FA-2 style streaming softmax adapted to the gather pattern, atomic-add into fp32 dK/dV buffers).
- 48-config autotune sweep (BLOCK_M, BLOCK_N, num_warps, num_stages) for the selected branch on Hopper.
- A five-model training fleet on FineWeb-Edu at 32k to 64k context with bf16 mixed precision.
- Long-context perplexity sweep across six evaluation lengths (2k to 256k) with straight and NTK-aware RoPE extension.
- LongBench v2 gold-answer-likelihood subset across six tasks.
- MoBA (Liu et al., February 2025) cross-comparison on the same Triton bench harness.

## Stack

Python, PyTorch, Triton, CUDA C++ (Hopper sm_90a, WGMMA inline-PTX), torch.utils.cpp_extension, pybind11, wandb, matplotlib. RunPod for GPU rental (H100 NVL, A100 SXM, A100 PCIe).

## Reproducing the benchmarks

```bash
pip install -e '.[dev,train]'
pytest tests/ -q                              # full suite; CUDA WGMMA tests are a Hopper-only subset, skipped on non-sm_90 cards
# Headline throughput is a 3-seed protocol (100 iters, 25 warmup each),
# pooled into a 95% bootstrap CI (this is what reports 7.07x at 64k):
for s in 0 1 2; do
  python -m nsa.bench.throughput --seq-lens 1024,2048,4096,8192,16384,32768,65536 \
      --impls nsa,fa3,full_sdpa --dtype bf16 --warmup 25 --iters 100 --seed $s \
      --out runs/throughput_seed$s.json
done
python scripts/multiseed_combine.py \
    --seeds runs/throughput_seed0.json,runs/throughput_seed1.json,runs/throughput_seed2.json \
    --out runs/throughput_multiseed.json
```

The throughput benchmark needs a Hopper card (the WGMMA selected forward and the FA-3 baseline both require sm_90a). On a non-Hopper (sm_89) card the suite still runs; the Hopper-only tests skip cleanly.

## Reproducing the training runs

Configurations live in `nsa/train/config_*.yaml`. The five recipes shipped:

```bash
# NSA scaling points, all 32k context
python -m nsa.train.train --config nsa/train/config_100m.yaml         # NSA-100M, 1B tokens
python -m nsa.train.train --config nsa/train/config_150m.yaml         # NSA-150M, 1B tokens
python -m nsa.train.train --config nsa/train/config_300m.yaml         # NSA-300M, 500M tokens

# Dense baseline at the largest workable context (8k on A100 PCIe at this batch shape)
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
  triton/               compressed.py, selected.py, sliding.py, gating.py, forward.py, backward.py, fp8.py, moba.py
  cuda/                 selected_fwd.cu (Hopper WGMMA), selected_bwd.cu (in-progress), bindings.cpp, setup.py
  model/                llama_nsa.py, llama_dense.py, config.py
  train/                train.py, data.py, config_*.yaml (NSA-100M, 150M, 300M, dense, 64k, sanity)
  eval/                 perplexity.py, long_context_probe.py, longbench.py
  bench/                throughput.py, memory.py, memory_sweep.py, correctness.py, branch_breakdown.py, autotune.py, plots.py
tests/                  test_{compressed,selected,sliding,combined,fp8,training_step,cuda_selected,cuda_selected_bwd}_forward|backward.py
writeup/                post.md, figures/{01..08}.png
notes/                  refs.bib
scripts/                fetch_wandb.py, make_plots.py, multiseed_combine.py
```

## License

Apache-2.0. See LICENSE.
