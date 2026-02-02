// NSA selected-branch backward on Hopper (sm_90a). Mirrors
// nsa/triton/backward.py: single-pass FA-2 bwd over the same top-k
// gather pattern. Four WGMMAs per kk: dP = dO@V^T (TT), dV += P^T@dO
// (NN), dQ += dS@K (TN), dK += dS^T@Q (NN); sP and sdS in smem are
// read with both tnsp=0 and tnsp=1 across the inner loop.
//
// Pre-step D_vec = (dO * O).sum(-1) is host-side (Python wrapper).
// Constraints: BLOCK_M = BLOCK_N = 64, HEAD_DIM in {64, 128}.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace nsa_cuda {
namespace bwd {

// ---------------------------------------------------------------------------
// GMMA descriptor builder (LAYOUT_TYPE = INTERLEAVE, no swizzle).
// Same encoding as the forward kernel: start_address (smem byte offset
// >> 4), lbo (>> 4), sbo (>> 4) packed into a 64-bit field.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t
make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t desc = 0;
    uint64_t start = (smem_addr & 0x3FFFFu) >> 4;
    uint64_t lbo   = (uint64_t)(lbo_bytes >> 4) & 0x3FFFu;
    uint64_t sbo   = (uint64_t)(sbo_bytes >> 4) & 0x3FFFu;
    desc |= start;
    desc |= lbo << 16;
    desc |= sbo << 32;
    return desc;
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// WGMMA atom variants. tnspA / tnspB control whether the operand's
// K-of-MMA-axis is the inner (= 0, "K-major") or outer (= 1, "MN-major")
// axis of the smem layout. f32 += bf16 * bf16 SS form, with predicated
// scale-D so the kernel always accumulates (predicate p set from input scaleD).
//
// Naming: T = K-major (tnsp 0), N = MN-major (tnsp 1). Suffix _AB.
// ---------------------------------------------------------------------------

// m64n64k16: 32 fp32 fragment entries per thread.
__device__ __forceinline__ void
wgmma_m64n64k16_TT(uint64_t descA, uint64_t descB,
                   float (&d)[32], int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        :  "l"(descA), "l"(descB), "r"(scaleD));
}

__device__ __forceinline__ void
wgmma_m64n64k16_TN(uint64_t descA, uint64_t descB,
                   float (&d)[32], int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        :  "l"(descA), "l"(descB), "r"(scaleD));
}

__device__ __forceinline__ void
wgmma_m64n64k16_NN(uint64_t descA, uint64_t descB,
                   float (&d)[32], int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        :  "l"(descA), "l"(descB), "r"(scaleD));
}

// m64n128k16: 64 fp32 fragment entries per thread.
__device__ __forceinline__ void
wgmma_m64n128k16_TN(uint64_t descA, uint64_t descB,
                    float (&d)[64], int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 0, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        :  "l"(descA), "l"(descB), "r"(scaleD));
}

__device__ __forceinline__ void
wgmma_m64n128k16_NN(uint64_t descA, uint64_t descB,
                    float (&d)[64], int scaleD) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63}, "
        "%64, %65, p, 1, 1, 1, 1;\n"
        "}\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        :  "l"(descA), "l"(descB), "r"(scaleD));
}

// ---------------------------------------------------------------------------
// Layout helper: pack (r, c) into core-matrix-tiled BF16 element index.
// Identical to the forward kernel's core_idx.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned
core_idx(unsigned r, unsigned c, unsigned Kcols) {
    unsigned mb = r >> 3;
    unsigned kb = c >> 3;
    unsigned ri = r & 7;
    unsigned ci = c & 7;
    return ((mb * (Kcols >> 3) + kb) << 6) + (ri << 3) + ci;
}

template <typename Op>
__device__ __forceinline__ float warp_row_reduce4(float v, Op op) {
    float a = op(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return op(a, __shfl_xor_sync(0xffffffffu, a, 2));
}

struct AddOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };

// ---------------------------------------------------------------------------
// Tile loaders.
//   load_tile_contig_core: contiguous gmem rows (Q tile, dO tile).
//   load_tile_gather_core_K: per-row gmem pointers, K-major store (used
//     for K and V tiles in the backward; the V transpose used in the
//     forward kernel is unnecessary here).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void
load_tile_contig_core(__nv_bfloat16* smem,
                      const __nv_bfloat16* gmem_base,
                      unsigned BLOCK_M, unsigned K,
                      unsigned m_valid_rows,
                      unsigned tid) {
    unsigned chunks_per_row = K >> 3;
    unsigned total_chunks = BLOCK_M * chunks_per_row;
    for (unsigned c = tid; c < total_chunks; c += 128) {
        unsigned r = c / chunks_per_row;
        unsigned cc = c - r * chunks_per_row;
        unsigned smem_off = core_idx(r, cc << 3, K);
        uint4* dst = reinterpret_cast<uint4*>(smem + smem_off);
        if (r < m_valid_rows) {
            const uint4* src = reinterpret_cast<const uint4*>(
                gmem_base + r * K + (cc << 3));
            *dst = *src;
        } else {
            *dst = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

__device__ __forceinline__ void
load_tile_gather_core_K(__nv_bfloat16* smem,
                        const __nv_bfloat16* const* row_ptrs,
                        const bool* row_valid,
                        unsigned BLOCK_N, unsigned D,
                        unsigned tid) {
    unsigned chunks_per_row = D >> 3;
    unsigned total_chunks = BLOCK_N * chunks_per_row;
    for (unsigned c = tid; c < total_chunks; c += 128) {
        unsigned r = c / chunks_per_row;
        unsigned cc = c - r * chunks_per_row;
        unsigned smem_off = core_idx(r, cc << 3, D);
        uint4* dst = reinterpret_cast<uint4*>(smem + smem_off);
        if (row_valid[r]) {
            const uint4* src = reinterpret_cast<const uint4*>(row_ptrs[r] + (cc << 3));
            *dst = *src;
        } else {
            *dst = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// Store an (M=BLOCK_M, N=BLOCK_N) fp32 fragment into smem as bf16 in the
// K-major core-matrix layout (BLOCK_N contig). Used for both sP and sdS.
// Per-thread fragment index conventions match the WGMMA m64n64k16 output
// layout: 4 entries per N-block of 8 cols, indexing rows {r0, r0, r1, r1}
// and cols {2*tig, 2*tig+1, 2*tig, 2*tig+1}.
__device__ __forceinline__ void
store_frag_to_smem_kmajor(__nv_bfloat16* smem,
                          const float* frag_fp32,
                          unsigned BLOCK_M, unsigned BLOCK_N, unsigned tid) {
    unsigned warp_id = tid >> 5;
    unsigned lane = tid & 31;
    unsigned group_id = lane >> 2;
    unsigned tig = lane & 3;
    unsigned r0 = (warp_id << 4) + group_id;
    unsigned r1 = r0 + 8;
    unsigned n_blocks = BLOCK_N >> 3;
    for (unsigned ib = 0; ib < n_blocks; ++ib) {
        __nv_bfloat16 b00 = __float2bfloat16(frag_fp32[(ib << 2) + 0]);
        __nv_bfloat16 b01 = __float2bfloat16(frag_fp32[(ib << 2) + 1]);
        __nv_bfloat16 b10 = __float2bfloat16(frag_fp32[(ib << 2) + 2]);
        __nv_bfloat16 b11 = __float2bfloat16(frag_fp32[(ib << 2) + 3]);
        unsigned packed_r0 = (unsigned)(*reinterpret_cast<unsigned short*>(&b00))
                           | ((unsigned)(*reinterpret_cast<unsigned short*>(&b01)) << 16);
        unsigned packed_r1 = (unsigned)(*reinterpret_cast<unsigned short*>(&b10))
                           | ((unsigned)(*reinterpret_cast<unsigned short*>(&b11)) << 16);
        unsigned core_r0 = ((r0 >> 3) * (BLOCK_N >> 3) + ib) << 6;
        unsigned in_core_r0 = ((r0 & 7) << 3) + (tig << 1);
        unsigned core_r1 = ((r1 >> 3) * (BLOCK_N >> 3) + ib) << 6;
        unsigned in_core_r1 = ((r1 & 7) << 3) + (tig << 1);
        if (r0 < BLOCK_M)
            *reinterpret_cast<unsigned*>(smem + core_r0 + in_core_r0) = packed_r0;
        if (r1 < BLOCK_M)
            *reinterpret_cast<unsigned*>(smem + core_r1 + in_core_r1) = packed_r1;
    }
}

// Atomic-add a fragment of shape (M=BLOCK_M_FRAG, N=N_FRAG) onto a
// gmem fp32 buffer at gathered row pointers row_ptrs[r] (one per
// fragment row) plus the column offset implied by the fragment layout.
// The fragment uses the m64n{N_FRAG}k16 output convention: per thread,
// 4 entries per N-block of 8 cols at rows {r0, r0, r1, r1} and cols
// {2*tig, 2*tig+1, 2*tig, 2*tig+1} where r0 = warp*16 + group_id and
// r1 = r0 + 8.
//
// row_ptrs is indexed in the fragment's M-axis (= BLOCK_M_FRAG, in
// {64} for this kernel). row_valid masks out-of-range rows.
template <int BLOCK_M_FRAG, int N_FRAG>
__device__ __forceinline__ void
atomic_add_frag_rows(float* const* row_ptrs,
                     const bool* row_valid,
                     const float* frag_fp32,
                     unsigned tid) {
    unsigned warp_id = tid >> 5;
    unsigned lane = tid & 31;
    unsigned group_id = lane >> 2;
    unsigned tig = lane & 3;
    unsigned r0 = (warp_id << 4) + group_id;
    unsigned r1 = r0 + 8;
    constexpr int N_BLOCKS = N_FRAG / 8;
    #pragma unroll
    for (int ib = 0; ib < N_BLOCKS; ++ib) {
        int col0 = (ib << 3) + (tig << 1);
        int col1 = col0 + 1;
        float v00 = frag_fp32[(ib << 2) + 0];
        float v01 = frag_fp32[(ib << 2) + 1];
        float v10 = frag_fp32[(ib << 2) + 2];
        float v11 = frag_fp32[(ib << 2) + 3];
        if (r0 < BLOCK_M_FRAG && row_valid[r0]) {
            atomicAdd(row_ptrs[r0] + col0, v00);
            atomicAdd(row_ptrs[r0] + col1, v01);
        }
        if (r1 < BLOCK_M_FRAG && row_valid[r1]) {
            atomicAdd(row_ptrs[r1] + col0, v10);
            atomicAdd(row_ptrs[r1] + col1, v11);
        }
    }
}

// ---------------------------------------------------------------------------
// Main backward kernel. One CTA per (batch_head, q_block). 128 threads
// per CTA (one warpgroup). Template parameter HEAD_DIM in {64, 128}.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ __launch_bounds__(128, 1)
void selected_attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float*         __restrict__ LSE,
    const float*         __restrict__ Dvec,
    const int*           __restrict__ Idx,
    __nv_bfloat16*       __restrict__ dQ,
    float*               __restrict__ dK_f,
    float*               __restrict__ dV_f,
    int Tq_p, int Tk_p, int top_k, int offset, int causal,
    float sm_scale)
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int D = HEAD_DIM;
    static_assert(D == 64 || D == 128, "HEAD_DIM must be 64 or 128");

    const int pid_qb = blockIdx.x;
    const int pid_bh = blockIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;
    __nv_bfloat16* sdO = sQ  + BLOCK_M * D;
    __nv_bfloat16* sK  = sdO + BLOCK_M * D;
    __nv_bfloat16* sV  = sK  + BLOCK_N * D;
    __nv_bfloat16* sP  = sV  + BLOCK_N * D;
    __nv_bfloat16* sdS = sP  + BLOCK_M * BLOCK_N;
    // 1-tile padding so the WGMMA's M-major read (which extends to
    // sdS + 8192 in the last ks iteration) does not graze past the
    // end of claimed dynamic smem. Without this, reads at the boundary
    // throw "illegal memory access" on H100 NVL even though all
    // accessed bytes nominally fall inside [sdS, sdS + 8192).

    // Load Q and dO tiles (contiguous gmem rows).
    const int q_block_start = pid_qb * BLOCK_M;
    const int Q_rows_valid =
        (q_block_start + BLOCK_M <= Tq_p) ? BLOCK_M : (Tq_p - q_block_start);
    const __nv_bfloat16* Q_base  = Q  + (size_t)pid_bh * Tq_p * D + (size_t)q_block_start * D;
    const __nv_bfloat16* dO_base = dO + (size_t)pid_bh * Tq_p * D + (size_t)q_block_start * D;
    load_tile_contig_core(sQ,  Q_base,  BLOCK_M, D, Q_rows_valid, tid);
    load_tile_contig_core(sdO, dO_base, BLOCK_M, D, Q_rows_valid, tid);
    __syncthreads();

    // Per-thread row mapping for the M=BLOCK_M fragments (S, dS, dQ).
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tig = lane & 3;
    const int r0 = (warp_id << 4) + group_id;
    const int r1 = r0 + 8;
    const int q_pos_r0 = q_block_start + r0 + offset;
    const int q_pos_r1 = q_block_start + r1 + offset;
    const int q_block_max_pos = q_block_start + (BLOCK_M - 1) + offset;

    // Per-row LSE and D_vec values, broadcast inside the warp from the
    // tig=0 lane via the standard 4-thread reduction trick (here just a
    // load + shuffle to all tig lanes, since each value is one fp32).
    const float* lse_base = LSE  + (size_t)pid_bh * Tq_p + (size_t)q_block_start;
    const float* dv_base  = Dvec + (size_t)pid_bh * Tq_p + (size_t)q_block_start;
    float lse_r0 = (r0 < Q_rows_valid) ? lse_base[r0] : -1.0e30f;
    float lse_r1 = (r1 < Q_rows_valid) ? lse_base[r1] : -1.0e30f;
    if (lse_r0 == -INFINITY) lse_r0 = -1.0e30f;
    if (lse_r1 == -INFINITY) lse_r1 = -1.0e30f;
    const float Dvec_r0 = (r0 < Q_rows_valid) ? dv_base[r0] : 0.0f;
    const float Dvec_r1 = (r1 < Q_rows_valid) ? dv_base[r1] : 0.0f;

    // dQ accumulator (fp32, fragment shape = m64n{D}k... output).
    constexpr int FRAG_DQ = (D == 64) ? 32 : 64;
    float dQ_acc[FRAG_DQ];
    #pragma unroll
    for (int i = 0; i < FRAG_DQ; ++i) dQ_acc[i] = 0.0f;

    // Smem descriptors. LBO = 128 (1 core's worth of inner-axis), SBO =
    // 128 * (inner_extent / 8). The outer extent doesn't appear in the
    // descriptor: the WGMMA instruction's M / N / K fields fix the tile
    // sizes; LBO / SBO encode only how to step between adjacent cores.
    const uint32_t sQ_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
    const uint32_t sdO_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sdO));
    const uint32_t sK_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
    const uint32_t sV_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
    const uint32_t sP_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(sP));
    const uint32_t sdS_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sdS));

    // (BLOCK_M, D) D-contig storage: LBO = 128 (1 core along D), SBO = 128 * (D/8).
    const uint64_t descQ  = make_smem_desc(sQ_addr,  128u, 128u * (D / 8));
    const uint64_t descdO = make_smem_desc(sdO_addr, 128u, 128u * (D / 8));
    // (BLOCK_N, D) D-contig storage: same structure as Q.
    const uint64_t descK  = make_smem_desc(sK_addr,  128u, 128u * (D / 8));
    const uint64_t descV  = make_smem_desc(sV_addr,  128u, 128u * (D / 8));
    // (BLOCK_M, BLOCK_N) BLOCK_N-contig storage: LBO = 128, SBO = 128 * (BLOCK_N/8).
    const uint64_t descP  = make_smem_desc(sP_addr,  128u, 128u * (BLOCK_N / 8));
    const uint64_t descdS = make_smem_desc(sdS_addr, 128u, 128u * (BLOCK_N / 8));

    const __nv_bfloat16* K_bh = K + (size_t)pid_bh * Tk_p * D;
    const __nv_bfloat16* V_bh = V + (size_t)pid_bh * Tk_p * D;

    const int* idx_base =
        Idx + (size_t)pid_bh * gridDim.x * top_k + (size_t)pid_qb * top_k;

    __shared__ const __nv_bfloat16* sK_rowptrs[BLOCK_N];
    __shared__ const __nv_bfloat16* sV_rowptrs[BLOCK_N];
    __shared__ float* sdK_rowptrs[BLOCK_N];
    __shared__ float* sdV_rowptrs[BLOCK_N];
    __shared__ bool sRow_valid[BLOCK_N];
    __shared__ int s_block_idx;
    __shared__ int s_kv_start;
    __shared__ int s_skip;

    float* dK_bh = dK_f + (size_t)pid_bh * Tk_p * D;
    float* dV_bh = dV_f + (size_t)pid_bh * Tk_p * D;

    for (int kk = 0; kk < top_k; ++kk) {
        if (tid == 0) {
            s_block_idx = idx_base[kk];
            s_kv_start = s_block_idx * BLOCK_N;
            s_skip = (causal != 0) && (s_kv_start > q_block_max_pos);
        }
        __syncthreads();
        if (s_skip) continue;
        const int kv_start = s_kv_start;

        if (tid < BLOCK_N) {
            int kv_off = kv_start + tid;
            bool valid = (kv_off >= 0) && (kv_off < Tk_p);
            sRow_valid[tid] = valid;
            sK_rowptrs[tid] = valid ? (K_bh + (size_t)kv_off * D) : K_bh;
            sV_rowptrs[tid] = valid ? (V_bh + (size_t)kv_off * D) : V_bh;
            sdK_rowptrs[tid] = valid ? (dK_bh + (size_t)kv_off * D) : dK_bh;
            sdV_rowptrs[tid] = valid ? (dV_bh + (size_t)kv_off * D) : dV_bh;
        }
        __syncthreads();

        load_tile_gather_core_K(sK, sK_rowptrs, sRow_valid, BLOCK_N, D, tid);
        load_tile_gather_core_K(sV, sV_rowptrs, sRow_valid, BLOCK_N, D, tid);
        __syncthreads();

        // ---- Recompute S = Q @ K^T * sm_scale (fp32 fragment) ----
        float s_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) s_acc[i] = 0.0f;
        wgmma_fence();
        constexpr int K_STEPS_QK = D / 16;
        #pragma unroll
        for (int ks = 0; ks < K_STEPS_QK; ++ks) {
            uint64_t descQk = descQ + (uint64_t)((ks * 256u) >> 4);
            uint64_t descKk = descK + (uint64_t)((ks * 256u) >> 4);
            wgmma_m64n64k16_TT(descQk, descKk, s_acc, 1);
        }
        wgmma_commit();
        wgmma_wait<0>();

        // ---- WGMMA #1: dP = dO @ V^T ----
        float dP_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) dP_acc[i] = 0.0f;
        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < K_STEPS_QK; ++ks) {
            uint64_t descdOk = descdO + (uint64_t)((ks * 256u) >> 4);
            uint64_t descVk  = descV  + (uint64_t)((ks * 256u) >> 4);
            wgmma_m64n64k16_TT(descdOk, descVk, dP_acc, 1);
        }
        wgmma_commit();
        wgmma_wait<0>();

        // Apply sm_scale to S; mask both S and dP (causal, padding).
        // Compute P = exp(S - lse) and dS = P * (dP - Dvec) * sm_scale,
        // both as fp32 fragments. Zero where masked.
        float p_frag[32];
        float ds_frag[32];
        const bool row0_in_q = (r0 < Q_rows_valid);
        const bool row1_in_q = (r1 < Q_rows_valid);
        #pragma unroll
        for (int ib = 0; ib < BLOCK_N / 8; ++ib) {
            int col0 = (ib << 3) + (tig << 1);
            int kv0 = kv_start + col0;
            int kv1 = kv0 + 1;
            bool in_tk0 = (kv0 < Tk_p);
            bool in_tk1 = (kv1 < Tk_p);
            bool causal0_r0 = (causal == 0) || (kv0 <= q_pos_r0);
            bool causal1_r0 = (causal == 0) || (kv1 <= q_pos_r0);
            bool causal0_r1 = (causal == 0) || (kv0 <= q_pos_r1);
            bool causal1_r1 = (causal == 0) || (kv1 <= q_pos_r1);
            bool keep00 = row0_in_q && in_tk0 && causal0_r0;
            bool keep01 = row0_in_q && in_tk1 && causal1_r0;
            bool keep10 = row1_in_q && in_tk0 && causal0_r1;
            bool keep11 = row1_in_q && in_tk1 && causal1_r1;

            int d00 = (ib << 2) + 0;
            int d01 = (ib << 2) + 1;
            int d10 = (ib << 2) + 2;
            int d11 = (ib << 2) + 3;

            float s00 = s_acc[d00] * sm_scale;
            float s01 = s_acc[d01] * sm_scale;
            float s10 = s_acc[d10] * sm_scale;
            float s11 = s_acc[d11] * sm_scale;

            float p00 = keep00 ? __expf(s00 - lse_r0) : 0.0f;
            float p01 = keep01 ? __expf(s01 - lse_r0) : 0.0f;
            float p10 = keep10 ? __expf(s10 - lse_r1) : 0.0f;
            float p11 = keep11 ? __expf(s11 - lse_r1) : 0.0f;

            p_frag[d00] = p00;
            p_frag[d01] = p01;
            p_frag[d10] = p10;
            p_frag[d11] = p11;

            float ds00 = keep00 ? (p00 * (dP_acc[d00] - Dvec_r0) * sm_scale) : 0.0f;
            float ds01 = keep01 ? (p01 * (dP_acc[d01] - Dvec_r0) * sm_scale) : 0.0f;
            float ds10 = keep10 ? (p10 * (dP_acc[d10] - Dvec_r1) * sm_scale) : 0.0f;
            float ds11 = keep11 ? (p11 * (dP_acc[d11] - Dvec_r1) * sm_scale) : 0.0f;

            ds_frag[d00] = ds00;
            ds_frag[d01] = ds01;
            ds_frag[d10] = ds10;
            ds_frag[d11] = ds11;
        }

        // Stage P and dS in smem.
        store_frag_to_smem_kmajor(sP,  p_frag,  BLOCK_M, BLOCK_N, tid);
        store_frag_to_smem_kmajor(sdS, ds_frag, BLOCK_M, BLOCK_N, tid);
        __syncthreads();

        // Per-ks descriptor byte-advance constants.
        // Each ks covers 16 K-elements = 2 cores along the K-axis. The
        // K-axis is the operand's INNER axis when tnsp = 0 (then K-step =
        // 2*LBO = 256) and OUTER axis when tnsp = 1 (then K-step = 2*SBO).
        // For the smem layouts here:
        //   sP, sdS  : LBO = 128, SBO = 128 * (BLOCK_N/8) = 1024.
        //              tnsp=0 K-step = 256;   tnsp=1 K-step = 2048.
        //   sdO, sQ  : LBO = 128, SBO = 128 * (D/8).
        //              tnsp=0 K-step = 256;   tnsp=1 K-step = 2*128*(D/8) = 256*(D/8).
        //   sK, sV   : same as sdO / sQ.
        constexpr unsigned KSTEP_PdS_T   = 256u;                     // sP / sdS  K-major
        constexpr unsigned KSTEP_PdS_N   = 2048u;                    // sP / sdS  MN-major
        constexpr unsigned KSTEP_QKVdO_T = 256u;                     // sQ / sK / sV / sdO  K-major
        constexpr unsigned KSTEP_QKVdO_N = 256u * (D / 8);           // ... MN-major

        constexpr int K_STEPS_DV = BLOCK_M / 16;
        constexpr int K_STEPS_DQ = BLOCK_N / 16;

        // ---- WGMMA #2: dV += P^T @ dO. A = sP M-major, B = sdO MN-major.
        // K-of-MMA = BLOCK_M (= 64); K-steps = BLOCK_M / 16 = 4.
        if constexpr (D == 64) {
            float dV_frag[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) dV_frag[i] = 0.0f;
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DV; ++ks) {
                uint64_t descAk = descP  + (uint64_t)((ks * KSTEP_PdS_N)   >> 4);
                uint64_t descBk = descdO + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n64k16_NN(descAk, descBk, dV_frag, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            atomic_add_frag_rows<BLOCK_N, 64>(sdV_rowptrs, sRow_valid, dV_frag, tid);
        } else {
            float dV_frag[64];
            #pragma unroll
            for (int i = 0; i < 64; ++i) dV_frag[i] = 0.0f;
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DV; ++ks) {
                uint64_t descAk = descP  + (uint64_t)((ks * KSTEP_PdS_N)   >> 4);
                uint64_t descBk = descdO + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n128k16_NN(descAk, descBk, dV_frag, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            atomic_add_frag_rows<BLOCK_N, 128>(sdV_rowptrs, sRow_valid, dV_frag, tid);
        }

        // ---- WGMMA #3: dQ += dS @ K. A = sdS K-major, B = sK MN-major.
        // K-of-MMA = BLOCK_N (= 64); K-steps = 4.
        if constexpr (D == 64) {
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DQ; ++ks) {
                uint64_t descAk = descdS + (uint64_t)((ks * KSTEP_PdS_T)   >> 4);
                uint64_t descBk = descK  + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n64k16_TN(descAk, descBk, dQ_acc, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
        } else {
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DQ; ++ks) {
                uint64_t descAk = descdS + (uint64_t)((ks * KSTEP_PdS_T)   >> 4);
                uint64_t descBk = descK  + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n128k16_TN(descAk, descBk, dQ_acc, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
        }

        // ---- WGMMA #4: dK += dS^T @ Q. A = sdS M-major, B = sQ MN-major.
        // K-of-MMA = BLOCK_M (= 64); K-steps = 4.
        if constexpr (D == 64) {
            float dK_frag[32];
            #pragma unroll
            for (int i = 0; i < 32; ++i) dK_frag[i] = 0.0f;
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DV; ++ks) {
                uint64_t descAk = descdS + (uint64_t)((ks * KSTEP_PdS_N)   >> 4);
                uint64_t descBk = descQ  + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n64k16_NN(descAk, descBk, dK_frag, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            atomic_add_frag_rows<BLOCK_N, 64>(sdK_rowptrs, sRow_valid, dK_frag, tid);
        } else {
            float dK_frag[64];
            #pragma unroll
            for (int i = 0; i < 64; ++i) dK_frag[i] = 0.0f;
            wgmma_fence();
            #pragma unroll
            for (int ks = 0; ks < K_STEPS_DV; ++ks) {
                uint64_t descAk = descdS + (uint64_t)((ks * KSTEP_PdS_N)   >> 4);
                uint64_t descBk = descQ  + (uint64_t)((ks * KSTEP_QKVdO_N) >> 4);
                wgmma_m64n128k16_NN(descAk, descBk, dK_frag, 1);
            }
            wgmma_commit();
            wgmma_wait<0>();
            atomic_add_frag_rows<BLOCK_N, 128>(sdK_rowptrs, sRow_valid, dK_frag, tid);
        }

        __syncthreads();
    }

    // ---- finalize: store dQ_acc as bf16 into global dQ ----
    __nv_bfloat16* dQ_base =
        dQ + (size_t)pid_bh * Tq_p * D + (size_t)q_block_start * D;
    constexpr int N_BLOCKS_DQ = FRAG_DQ / 4;
    #pragma unroll
    for (int ib = 0; ib < N_BLOCKS_DQ; ++ib) {
        int col0 = (ib << 3) + (tig << 1);
        float a0 = dQ_acc[(ib << 2) + 0];
        float a1 = dQ_acc[(ib << 2) + 1];
        float b0 = dQ_acc[(ib << 2) + 2];
        float b1 = dQ_acc[(ib << 2) + 3];
        if (r0 < Q_rows_valid) {
            __nv_bfloat16 ba = __float2bfloat16(a0);
            __nv_bfloat16 bb = __float2bfloat16(a1);
            unsigned packed = (unsigned)(*reinterpret_cast<unsigned short*>(&ba))
                            | ((unsigned)(*reinterpret_cast<unsigned short*>(&bb)) << 16);
            *reinterpret_cast<unsigned*>(dQ_base + r0 * D + col0) = packed;
        }
        if (r1 < Q_rows_valid) {
            __nv_bfloat16 ba = __float2bfloat16(b0);
            __nv_bfloat16 bb = __float2bfloat16(b1);
            unsigned packed = (unsigned)(*reinterpret_cast<unsigned short*>(&ba))
                            | ((unsigned)(*reinterpret_cast<unsigned short*>(&bb)) << 16);
            *reinterpret_cast<unsigned*>(dQ_base + r1 * D + col0) = packed;
        }
    }
}

}  // namespace bwd

// ---------------------------------------------------------------------------
// Host launcher (in nsa_cuda::, exposed via bindings.cpp).
// ---------------------------------------------------------------------------
extern "C" void
launch_selected_attn_bwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO,
    const float* LSE, const float* Dvec,
    const int* Idx,
    __nv_bfloat16* dQ, float* dK_f, float* dV_f,
    int BH, int n_q_blocks, int Tq_p, int Tk_p, int D, int top_k,
    int offset, int causal, float sm_scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    dim3 grid(n_q_blocks, BH);
    dim3 block(128, 1, 1);

    // Trailing 1-tile padding after sdS so wgmma boundary reads stay
    // inside claimed dynamic smem (see kernel comment near sdS).
    constexpr int SMEM_TAIL_PAD = BLOCK_M * BLOCK_N;
    size_t smem_bytes_64 =
        (BLOCK_M * 64 + BLOCK_M * 64 + BLOCK_N * 64 + BLOCK_N * 64
         + BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_N + SMEM_TAIL_PAD) * sizeof(__nv_bfloat16);
    size_t smem_bytes_128 =
        (BLOCK_M * 128 + BLOCK_M * 128 + BLOCK_N * 128 + BLOCK_N * 128
         + BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_N + SMEM_TAIL_PAD) * sizeof(__nv_bfloat16);

    if (D == 64) {
        cudaFuncSetAttribute(bwd::selected_attn_bwd_kernel<64>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes_64);
        bwd::selected_attn_bwd_kernel<64><<<grid, block, smem_bytes_64, stream>>>(
            Q, K, V, dO, LSE, Dvec, Idx, dQ, dK_f, dV_f,
            Tq_p, Tk_p, top_k, offset, causal, sm_scale);
    } else if (D == 128) {
        cudaFuncSetAttribute(bwd::selected_attn_bwd_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes_128);
        bwd::selected_attn_bwd_kernel<128><<<grid, block, smem_bytes_128, stream>>>(
            Q, K, V, dO, LSE, Dvec, Idx, dQ, dK_f, dV_f,
            Tq_p, Tk_p, top_k, offset, causal, sm_scale);
    }
}

}  // namespace nsa_cuda
