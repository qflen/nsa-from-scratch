# 005 - Top-k tie-breaking: share indices instead of shuffling

## Context

The selected branch picks top_k blocks per query-block via `torch.topk` on a `[B, H, n_q_blocks, n_kv_blocks]` score tensor. When two blocks have exactly equal score (e.g. a row of zeros at init, or a saturated softmax in fp16), `torch.topk` on CUDA returns them in index order. The kernel and the reference both call `torch.topk`, but apply causal masking through different paths; floating-point determinism through different code paths means the two can disagree on which tied block "wins". Disagreement here means the kernel and reference gather different K/V rows, which produces a correctness-test mismatch even though both outputs are mathematically valid.

## Decision

Considered a random permutation of equal-prefix blocks before top-k (suggested in Yuan et al. to defuse position bias). Rejected: `block_indices` flows as a gather index, not a learned tensor, so position bias only affects data-routing, not the learning signal. Tests use a shared `block_indices` tensor (`tests/test_selected_forward.py::_shared_topk_indices`) fed to both paths.

## Consequences

The end-to-end NSA forward still calls `torch.topk` internally, so fresh random scores may pick different "tied" blocks across runs, but the outputs agree to the sub-1e-2 tolerance the writeup commits to. Tests isolate "same gather, same output?" from "same tie-break?". If training-time bias toward early blocks ever shows up, add a permutation inside `_resolve_block_indices` before top-k; the kernel does not change.
