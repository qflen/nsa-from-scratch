// pybind bindings for the Hopper NSA selected-branch CUDA forward and
// backward. Inputs are pre-padded by the Python wrapper; the backward
// also takes a host-precomputed D_vec = (dO * O).sum(-1).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>

namespace nsa_cuda {

extern "C" void launch_selected_attn_fwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const int* Idx,
    __nv_bfloat16* Out, float* Lse,
    int BH, int n_q_blocks, int Tq_p, int Tk_p, int D, int top_k,
    int offset, int causal, float sm_scale,
    cudaStream_t stream);

extern "C" void launch_selected_attn_bwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO,
    const float* LSE, const float* Dvec,
    const int* Idx,
    __nv_bfloat16* dQ, float* dK_f, float* dV_f,
    int BH, int n_q_blocks, int Tq_p, int Tk_p, int D, int top_k,
    int offset, int causal, float sm_scale,
    cudaStream_t stream);

}  // namespace nsa_cuda


std::vector<torch::Tensor>
selected_attention_fwd_cuda(
    torch::Tensor Q,                     // [B, H, Tq_p, D] bf16, contiguous
    torch::Tensor K,                     // [B, H, Tk_p, D] bf16, contiguous
    torch::Tensor V,                     // [B, H, Tk_p, D] bf16, contiguous
    torch::Tensor block_indices,         // [B, H, n_q_blocks, top_k] int32
    int64_t block_size_m,
    int64_t block_size_n,
    int64_t top_k,
    int64_t offset,
    bool causal,
    double sm_scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(block_indices.is_cuda(), "block_indices must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bf16 (this kernel)");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16, "K must be bf16");
    TORCH_CHECK(V.scalar_type() == at::kBFloat16, "V must be bf16");
    TORCH_CHECK(block_indices.scalar_type() == at::kInt, "block_indices must be int32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(block_indices.is_contiguous(), "block_indices must be contiguous");
    TORCH_CHECK(block_size_m == 64 && block_size_n == 64,
                "this CUDA kernel currently supports BLOCK_M = BLOCK_N = 64 only");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int Tq_p = Q.size(2);
    const int D = Q.size(3);
    const int Tk_p = K.size(2);
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(3) == D);
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == Tk_p && V.size(3) == D);
    TORCH_CHECK(D == 64 || D == 128, "this CUDA kernel currently supports HEAD_DIM in {64, 128}");
    TORCH_CHECK(Tq_p % block_size_m == 0, "Tq_p must be a multiple of BLOCK_M");
    TORCH_CHECK(Tk_p % block_size_n == 0, "Tk_p must be a multiple of BLOCK_N");
    const int n_q_blocks = Tq_p / block_size_m;
    TORCH_CHECK(block_indices.size(0) == B && block_indices.size(1) == H,
                "block_indices [B, H, ...] mismatch");
    TORCH_CHECK(block_indices.size(2) == n_q_blocks,
                "block_indices n_q_blocks mismatch");
    TORCH_CHECK(block_indices.size(3) == top_k,
                "block_indices top_k mismatch");

    auto opts_bf16 = torch::TensorOptions().dtype(at::kBFloat16).device(Q.device());
    auto opts_f32  = torch::TensorOptions().dtype(at::kFloat).device(Q.device());
    auto out = torch::empty({B, H, Tq_p, D}, opts_bf16);
    auto lse = torch::empty({B, H, Tq_p}, opts_f32);

    const int BH = B * H;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.device().index()).stream();

    nsa_cuda::launch_selected_attn_fwd(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        block_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(),
        BH, n_q_blocks, Tq_p, Tk_p, D,
        static_cast<int>(top_k), static_cast<int>(offset),
        causal ? 1 : 0, static_cast<float>(sm_scale),
        stream);

    return {out, lse};
}


std::vector<torch::Tensor>
selected_attention_bwd_cuda(
    torch::Tensor dO,                    // [B, H, Tq_p, D] bf16
    torch::Tensor Q,                     // [B, H, Tq_p, D] bf16
    torch::Tensor K,                     // [B, H, Tk_p, D] bf16
    torch::Tensor V,                     // [B, H, Tk_p, D] bf16
    torch::Tensor LSE,                   // [B, H, Tq_p] fp32
    torch::Tensor Dvec,                  // [B, H, Tq_p] fp32
    torch::Tensor block_indices,         // [B, H, n_q_blocks, top_k] int32
    int64_t block_size_m,
    int64_t block_size_n,
    int64_t top_k,
    int64_t offset,
    bool causal,
    double sm_scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(),
                "Q/K/V/dO must be CUDA tensors");
    TORCH_CHECK(LSE.is_cuda() && Dvec.is_cuda() && block_indices.is_cuda(),
                "LSE/Dvec/block_indices must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bf16");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16, "K must be bf16");
    TORCH_CHECK(V.scalar_type() == at::kBFloat16, "V must be bf16");
    TORCH_CHECK(dO.scalar_type() == at::kBFloat16, "dO must be bf16");
    TORCH_CHECK(LSE.scalar_type() == at::kFloat, "LSE must be fp32");
    TORCH_CHECK(Dvec.scalar_type() == at::kFloat, "Dvec must be fp32");
    TORCH_CHECK(block_indices.scalar_type() == at::kInt, "block_indices must be int32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q/K/V must be contiguous");
    TORCH_CHECK(dO.is_contiguous() && LSE.is_contiguous() && Dvec.is_contiguous(),
                "dO/LSE/Dvec must be contiguous");
    TORCH_CHECK(block_indices.is_contiguous(), "block_indices must be contiguous");
    TORCH_CHECK(block_size_m == 64 && block_size_n == 64,
                "this CUDA backward currently supports BLOCK_M = BLOCK_N = 64 only");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int Tq_p = Q.size(2);
    const int D = Q.size(3);
    const int Tk_p = K.size(2);
    TORCH_CHECK(K.size(0) == B && K.size(1) == H && K.size(3) == D);
    TORCH_CHECK(V.size(0) == B && V.size(1) == H && V.size(2) == Tk_p && V.size(3) == D);
    TORCH_CHECK(dO.size(0) == B && dO.size(1) == H && dO.size(2) == Tq_p && dO.size(3) == D);
    TORCH_CHECK(LSE.size(0) == B && LSE.size(1) == H && LSE.size(2) == Tq_p);
    TORCH_CHECK(Dvec.size(0) == B && Dvec.size(1) == H && Dvec.size(2) == Tq_p);
    TORCH_CHECK(D == 64 || D == 128, "this CUDA backward currently supports HEAD_DIM in {64, 128}");
    TORCH_CHECK(Tq_p % block_size_m == 0, "Tq_p must be a multiple of BLOCK_M");
    TORCH_CHECK(Tk_p % block_size_n == 0, "Tk_p must be a multiple of BLOCK_N");
    const int n_q_blocks = Tq_p / block_size_m;
    TORCH_CHECK(block_indices.size(0) == B && block_indices.size(1) == H,
                "block_indices [B, H, ...] mismatch");
    TORCH_CHECK(block_indices.size(2) == n_q_blocks,
                "block_indices n_q_blocks mismatch");
    TORCH_CHECK(block_indices.size(3) == top_k,
                "block_indices top_k mismatch");

    auto opts_bf16 = torch::TensorOptions().dtype(at::kBFloat16).device(Q.device());
    auto opts_f32  = torch::TensorOptions().dtype(at::kFloat).device(Q.device());
    auto dQ   = torch::empty({B, H, Tq_p, D}, opts_bf16);
    auto dK_f = torch::zeros({B, H, Tk_p, D}, opts_f32);
    auto dV_f = torch::zeros({B, H, Tk_p, D}, opts_f32);

    const int BH = B * H;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(Q.device().index()).stream();

    nsa_cuda::launch_selected_attn_bwd(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr()),
        LSE.data_ptr<float>(),
        Dvec.data_ptr<float>(),
        block_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(dQ.data_ptr()),
        dK_f.data_ptr<float>(),
        dV_f.data_ptr<float>(),
        BH, n_q_blocks, Tq_p, Tk_p, D,
        static_cast<int>(top_k), static_cast<int>(offset),
        causal ? 1 : 0, static_cast<float>(sm_scale),
        stream);

    return {dQ, dK_f, dV_f};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selected_attention_fwd_cuda", &selected_attention_fwd_cuda,
          "NSA selected-branch forward (Hopper WGMMA)",
          pybind11::arg("Q"),
          pybind11::arg("K"),
          pybind11::arg("V"),
          pybind11::arg("block_indices"),
          pybind11::arg("block_size_m"),
          pybind11::arg("block_size_n"),
          pybind11::arg("top_k"),
          pybind11::arg("offset"),
          pybind11::arg("causal"),
          pybind11::arg("sm_scale"));
    m.def("selected_attention_bwd_cuda", &selected_attention_bwd_cuda,
          "NSA selected-branch backward (Hopper WGMMA)",
          pybind11::arg("dO"),
          pybind11::arg("Q"),
          pybind11::arg("K"),
          pybind11::arg("V"),
          pybind11::arg("LSE"),
          pybind11::arg("Dvec"),
          pybind11::arg("block_indices"),
          pybind11::arg("block_size_m"),
          pybind11::arg("block_size_n"),
          pybind11::arg("top_k"),
          pybind11::arg("offset"),
          pybind11::arg("causal"),
          pybind11::arg("sm_scale"));
}
