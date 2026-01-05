"""End-to-end smoke: build a small LlamaNSA, run one fwd+bwd step,
check that gradients flow back to the embedding and no nans appear
after one optimizer step. Cheapest signal that the model is wired
correctly before paying for a real training run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")


def _tiny_cfg(attention: str):
    from nsa.model.config import TransformerConfig
    return TransformerConfig(
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=256,
        tied_embeddings=True,
        attention=attention,
        nsa_block_size_c=32,
        nsa_block_size_n=32,
        nsa_block_size_m=32,
        nsa_top_k=4,
        nsa_window_size=32,
    )


@pytest.mark.parametrize("attention", ["nsa", "dense"])
def test_training_step(attention):
    from nsa.model.llama_dense import LlamaDense
    from nsa.model.llama_nsa import LlamaNSA

    cfg = _tiny_cfg(attention)
    model = (LlamaNSA if attention == "nsa" else LlamaDense)(cfg).cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ids = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()

    nan_grads = []
    none_grads = []
    for n, p in model.named_parameters():
        if p.grad is None:
            none_grads.append(n)
        elif not torch.isfinite(p.grad).all():
            nan_grads.append(n)
    # Every learnable parameter must receive a gradient: a None grad on
    # q_proj / k_proj / v_proj means the kernel call is not autograd-aware
    # and the attention block is not actually training. This test exists
    # specifically to catch that regression.
    assert not none_grads, f"params got no grad (kernel autograd broken?): {none_grads}"
    assert not nan_grads, f"non-finite grads in: {nan_grads[:5]}"

    if attention == "nsa":
        # And: those gradients must actually be non-zero. A backward that
        # silently returns zeros would pass the None / finite checks above.
        zero_grads = []
        for layer_name in ["layers.0.attn.q_proj.weight", "layers.0.attn.k_proj.weight",
                           "layers.0.attn.v_proj.weight"]:
            p = dict(model.named_parameters())[layer_name]
            if p.grad.abs().max().item() < 1e-8:
                zero_grads.append(layer_name)
        assert not zero_grads, f"attention projections got zero grad: {zero_grads}"

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    assert grad_norm < 100.0, f"grad norm explodes: {grad_norm}"
    optim.step()

    # Sanity: a second forward must not produce nan.
    with torch.no_grad():
        out2 = model(ids)
        assert torch.isfinite(out2).all()
