// NSA selected-branch forward on Hopper (sm_90a). FA-2 style streaming
// softmax with WGMMA SS atoms (bf16 inputs, f32 accumulator) for both
// Q@K^T and P@V. K-major smem layout (interleaved, no swizzle); V is
// transposed at load time so PV is also K-major. Causal masking uses
// original token positions; OFFSET = Tk - Tq.
//
// Constraints: BLOCK_M = BLOCK_N = 64, HEAD_DIM in {64, 128}.
// One CTA per (batch_head, q_block); 128 threads = one warpgroup.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace nsa_cuda {

// GMMA descriptor (INTERLEAVE, no swizzle). start_address bits [0,14),
// lbo [16,30), sbo [32,46), all in units of 16 bytes; base + layout_type
// zero for INTERLEAVE.
__device__ __forceinline__ uint64_t
make_smem_desc(uint32_t smem_addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {
    uint64_t desc = 0;
    uint64_t start = (smem_addr & 0x3FFFFu) >> 4;          // 14 bits
    uint64_t lbo   = (uint64_t)(lbo_bytes >> 4) & 0x3FFFu; // 14 bits
    uint64_t sbo   = (uint64_t)(sbo_bytes >> 4) & 0x3FFFu; // 14 bits
    desc |= start;
    desc |= lbo << 16;
    desc |= sbo << 32;
    return desc;
}

// ---------------------------------------------------------------------------
// WGMMA control + atoms via inline PTX. We use the SS form (both operands
// from smem) for both QKt and PV; tnspA = tnspB = 0 (Major::K), scaleA =
// scaleB = +1, scaleD predicated by the C-source flag.
// ---------------------------------------------------------------------------
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

// m64n64k16 f32 += bf16*bf16, both operands smem, both K-major.
__device__ __forceinline__ void
wgmma_m64n64k16_bf16_ss(uint64_t desc_a, uint64_t desc_b,
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
        :  "l"(desc_a), "l"(desc_b), "r"(scaleD));
}

// m64n128k16 f32 += bf16*bf16, both operands smem, both K-major.
__device__ __forceinline__ void
wgmma_m64n128k16_bf16_ss(uint64_t desc_a, uint64_t desc_b,
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
        "%64, %65, p, 1, 1, 0, 0;\n"
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
        :  "l"(desc_a), "l"(desc_b), "r"(scaleD));
}

// ---------------------------------------------------------------------------
// Layout helper: pack (r, c) into core-matrix-tiled BF16 element index.
//   Total elements = M_rows * K_cols (no padding).
//   smem[r, c] -> index ((r/8)*(K/8) + c/8) * 64 + (r%8)*8 + (c%8).
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned
core_idx(unsigned r, unsigned c, unsigned Kcols) {
    unsigned mb = r >> 3;
    unsigned kb = c >> 3;
    unsigned ri = r & 7;
    unsigned ci = c & 7;
    return ((mb * (Kcols >> 3) + kb) << 6) + (ri << 3) + ci;
}

// ---------------------------------------------------------------------------
// WGMMA accumulator <-> per-thread (row, col) mapping.
//
// In a 128-thread warpgroup, with warp_id = tid/32, lane = tid%32,
//   group_id = lane/4 (in [0,8)) -> within-warp row offset (group_id, group_id+8)
//   tid_in_grp = lane%4 (in [0,4))
//
// Per fragment N-block (8 cols at a time), the 4 fp32 entries of one
// thread are at:
//   d[4*i + 0] -> (warp*16 + group_id,     8*i + 2*tig + 0)
//   d[4*i + 1] -> (warp*16 + group_id,     8*i + 2*tig + 1)
//   d[4*i + 2] -> (warp*16 + group_id + 8, 8*i + 2*tig + 0)
//   d[4*i + 3] -> (warp*16 + group_id + 8, 8*i + 2*tig + 1)
//
// Each row of M is held by exactly 4 threads (the 4 tig values).
// ---------------------------------------------------------------------------

template <typename Op>
__device__ __forceinline__ float warp_row_reduce4(float v, Op op) {
    float a = op(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return op(a, __shfl_xor_sync(0xffffffffu, a, 2));
}

struct MaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct AddOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };

// ---------------------------------------------------------------------------
// Tile loaders.
//
// load_tile_contig_core: contiguous gmem rows (e.g. Q tile). The 16-byte
// dest in smem is naturally aligned (one core-row).
//
// load_tile_gather_core_K: per-row gmem pointers into K (gather load),
// store as (rows=BLOCK_N, cols=D) in core-matrix layout. K=D inside.
//
// load_tile_gather_core_V_T: per-row gmem pointers into V, but transpose
// at write time: gmem V[bn, d] -> smem element at core_idx(d, bn, BLOCK_N).
// We do this with one BF16 store per element (slow but safe; total 64*64
// BF16 = 4096 stores per CTA per top_k iter, at 32 stores/thread for 128
// threads per warpgroup).
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

__device__ __forceinline__ void
load_tile_gather_core_V_T(__nv_bfloat16* smem,
                          const __nv_bfloat16* const* row_ptrs,
                          const bool* row_valid,
                          unsigned BLOCK_N, unsigned D,
                          unsigned tid) {
    // V gmem layout: row r (bn=r), D contiguous values along cols.
    // Smem layout (transposed): (D rows, BLOCK_N cols), core-matrix tiled.
    //   element at gmem (bn, d) -> smem core_idx(d, bn, BLOCK_N).
    //
    // Strategy: each thread reads 8 contiguous BF16 from gmem (d=8j..8j+7
    // for some bn) and writes them to 8 different smem locations (each
    // with the same bn, d running over 8 values). Each write is one
    // unsigned short (2 bytes).
    unsigned chunks_per_row = D >> 3;            // 8 BF16 per chunk
    unsigned total_chunks = BLOCK_N * chunks_per_row;
    for (unsigned c = tid; c < total_chunks; c += 128) {
        unsigned bn = c / chunks_per_row;
        unsigned dj = c - bn * chunks_per_row;
        unsigned d_base = dj << 3;
        // Read 8 BF16 from V[bn, d_base .. d_base+7].
        __nv_bfloat16 vals[8];
        if (row_valid[bn]) {
            const uint4* src = reinterpret_cast<const uint4*>(row_ptrs[bn] + d_base);
            uint4 packed = *src;
            // Cast bytes back into 8 BF16. uint4 = 4 x 32-bit lanes,
            // each holding 2 BF16.
            unsigned short* s = reinterpret_cast<unsigned short*>(&packed);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                vals[u] = *reinterpret_cast<__nv_bfloat16*>(&s[u]);
            }
        } else {
            #pragma unroll
            for (int u = 0; u < 8; ++u) vals[u] = __float2bfloat16(0.0f);
        }
        // Scatter to smem with d running 0..7.
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            unsigned d = d_base + u;
            unsigned smem_off = core_idx(d, bn, BLOCK_N);
            smem[smem_off] = vals[u];
        }
    }
}

// Store the P tile (BLOCK_M x BLOCK_N, fp32 in fragment regs) to smem
// (bf16) in K-major canonical layout. Each thread writes 4 * (BLOCK_N/8)
// BF16 values; we pack adjacent col pairs into one 32-bit store.
__device__ __forceinline__ void
store_p_to_smem(__nv_bfloat16* smem,
                const float* p_fp32,
                unsigned BLOCK_M, unsigned BLOCK_N, unsigned tid) {
    unsigned warp_id = tid >> 5;
    unsigned lane = tid & 31;
    unsigned group_id = lane >> 2;
    unsigned tig = lane & 3;
    unsigned r0 = (warp_id << 4) + group_id;
    unsigned r1 = r0 + 8;
    unsigned n_blocks = BLOCK_N >> 3;
    for (unsigned ib = 0; ib < n_blocks; ++ib) {
        __nv_bfloat16 b00 = __float2bfloat16(p_fp32[(ib << 2) + 0]);
        __nv_bfloat16 b01 = __float2bfloat16(p_fp32[(ib << 2) + 1]);
        __nv_bfloat16 b10 = __float2bfloat16(p_fp32[(ib << 2) + 2]);
        __nv_bfloat16 b11 = __float2bfloat16(p_fp32[(ib << 2) + 3]);

        unsigned packed_r0 = (unsigned)(*reinterpret_cast<unsigned short*>(&b00))
                           | ((unsigned)(*reinterpret_cast<unsigned short*>(&b01)) << 16);
        unsigned packed_r1 = (unsigned)(*reinterpret_cast<unsigned short*>(&b10))
                           | ((unsigned)(*reinterpret_cast<unsigned short*>(&b11)) << 16);

        // core_idx(r, col0, BLOCK_N), col0 = ib*8 + tig*2.
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

// ---------------------------------------------------------------------------
// Main kernel. One CTA per (batch_head, q_block). 128 threads per CTA.
// Template parameter HEAD_DIM in {64, 128}.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ __launch_bounds__(128, 1)
void selected_attn_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int*           __restrict__ Idx,
    __nv_bfloat16*       __restrict__ Out,
    float*               __restrict__ Lse,
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
    __nv_bfloat16* sQ = smem;                          // [BLOCK_M x D]
    __nv_bfloat16* sK = sQ + BLOCK_M * D;              // [BLOCK_N x D]
    __nv_bfloat16* sV = sK + BLOCK_N * D;              // [D x BLOCK_N]   (transposed)
    __nv_bfloat16* sP = sV + D * BLOCK_N;              // [BLOCK_M x BLOCK_N]

    // Load Q tile.
    const int q_block_start = pid_qb * BLOCK_M;
    const int Q_rows_valid =
        (q_block_start + BLOCK_M <= Tq_p) ? BLOCK_M : (Tq_p - q_block_start);
    const __nv_bfloat16* Q_base = Q + (size_t)pid_bh * Tq_p * D + (size_t)q_block_start * D;
    load_tile_contig_core(sQ, Q_base, BLOCK_M, D, Q_rows_valid, tid);
    __syncthreads();

    // Per-thread state.
    constexpr int FRAG_QK = 32;
    constexpr int FRAG_PV = (D == 64) ? 32 : 64;

    float acc[FRAG_PV];
    #pragma unroll
    for (int i = 0; i < FRAG_PV; ++i) acc[i] = 0.0f;

    float m_r0 = -1.0e30f, m_r1 = -1.0e30f;
    float l_r0 = 0.0f,     l_r1 = 0.0f;

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int tig = lane & 3;
    const int r0 = (warp_id << 4) + group_id;
    const int r1 = r0 + 8;
    const int q_pos_r0 = q_block_start + r0 + offset;
    const int q_pos_r1 = q_block_start + r1 + offset;
    const int q_block_max_pos = q_block_start + (BLOCK_M - 1) + offset;

    // Smem descriptors.
    const uint32_t sQ_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sQ));
    const uint32_t sK_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sK));
    const uint32_t sV_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sV));
    const uint32_t sP_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sP));

    // Q: (BLOCK_M, D), K-major. SBO_bytes = 128 * (D/8).
    const uint64_t descQ = make_smem_desc(sQ_addr, 128u, 128u * (D / 8));
    // K: (BLOCK_N, D), K-major (D contiguous). SBO_bytes = 128 * (D/8).
    const uint64_t descK = make_smem_desc(sK_addr, 128u, 128u * (D / 8));
    // V (transposed): (D rows, BLOCK_N cols), K-major (BLOCK_N contiguous).
    // For WGMMA #2 (P @ V), B is (K=BLOCK_N, N=D). With our transposed
    // smem, mode 0 (N=D) is the outer row, mode 1 (K=BLOCK_N) is contiguous
    // -> K-major. SBO_bytes = 128 * (BLOCK_N / 8) for N step.
    const uint64_t descV = make_smem_desc(sV_addr, 128u, 128u * (BLOCK_N / 8));
    // P: (BLOCK_M, BLOCK_N), K-major. SBO_bytes = 128 * (BLOCK_N/8).
    const uint64_t descP = make_smem_desc(sP_addr, 128u, 128u * (BLOCK_N / 8));

    const __nv_bfloat16* K_bh = K + (size_t)pid_bh * Tk_p * D;
    const __nv_bfloat16* V_bh = V + (size_t)pid_bh * Tk_p * D;

    const int* idx_base =
        Idx + (size_t)pid_bh * gridDim.x * top_k + (size_t)pid_qb * top_k;

    __shared__ const __nv_bfloat16* sK_rowptrs[BLOCK_N];
    __shared__ const __nv_bfloat16* sV_rowptrs[BLOCK_N];
    __shared__ bool sRow_valid[BLOCK_N];
    __shared__ int s_block_idx;
    __shared__ int s_kv_start;
    __shared__ int s_skip;

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
        }
        __syncthreads();

        load_tile_gather_core_K(sK, sK_rowptrs, sRow_valid, BLOCK_N, D, tid);
        load_tile_gather_core_V_T(sV, sV_rowptrs, sRow_valid, BLOCK_N, D, tid);
        __syncthreads();

        // ---- WGMMA #1: S = Q @ K^T, M=BLOCK_M, N=BLOCK_N, K=D ----
        // Each k-step covers 16 K-elements; advance both A and B
        // start_addresses by 256 bytes per k-step (= 2 cores along K).
        float s_acc[FRAG_QK];
        #pragma unroll
        for (int i = 0; i < FRAG_QK; ++i) s_acc[i] = 0.0f;

        wgmma_fence();
        constexpr int K_STEPS_QK = D / 16;
        #pragma unroll
        for (int ks = 0; ks < K_STEPS_QK; ++ks) {
            uint64_t descQk = descQ + (uint64_t)((ks * 256u) >> 4);
            uint64_t descKk = descK + (uint64_t)((ks * 256u) >> 4);
            wgmma_m64n64k16_bf16_ss(descQk, descKk, s_acc, 1);
        }
        wgmma_commit();
        wgmma_wait<0>();

        // Apply scale, masks, online softmax update.
        float m_new_r0 = m_r0;
        float m_new_r1 = m_r1;
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
            float v00 = s_acc[d00] * sm_scale;
            float v01 = s_acc[d01] * sm_scale;
            float v10 = s_acc[d10] * sm_scale;
            float v11 = s_acc[d11] * sm_scale;
            s_acc[d00] = keep00 ? v00 : -1.0e30f;
            s_acc[d01] = keep01 ? v01 : -1.0e30f;
            s_acc[d10] = keep10 ? v10 : -1.0e30f;
            s_acc[d11] = keep11 ? v11 : -1.0e30f;

            m_new_r0 = fmaxf(m_new_r0, fmaxf(s_acc[d00], s_acc[d01]));
            m_new_r1 = fmaxf(m_new_r1, fmaxf(s_acc[d10], s_acc[d11]));
        }
        m_new_r0 = warp_row_reduce4(m_new_r0, MaxOp{});
        m_new_r1 = warp_row_reduce4(m_new_r1, MaxOp{});

        float alpha_r0 = __expf(m_r0 - m_new_r0);
        float alpha_r1 = __expf(m_r1 - m_new_r1);

        float p_sum_r0 = 0.0f, p_sum_r1 = 0.0f;
        float p_frag[FRAG_QK];
        #pragma unroll
        for (int ib = 0; ib < BLOCK_N / 8; ++ib) {
            int d00 = (ib << 2) + 0;
            int d01 = (ib << 2) + 1;
            int d10 = (ib << 2) + 2;
            int d11 = (ib << 2) + 3;
            float p00 = __expf(s_acc[d00] - m_new_r0);
            float p01 = __expf(s_acc[d01] - m_new_r0);
            float p10 = __expf(s_acc[d10] - m_new_r1);
            float p11 = __expf(s_acc[d11] - m_new_r1);
            p_frag[d00] = p00; p_frag[d01] = p01;
            p_frag[d10] = p10; p_frag[d11] = p11;
            p_sum_r0 += p00 + p01;
            p_sum_r1 += p10 + p11;
        }
        p_sum_r0 = warp_row_reduce4(p_sum_r0, AddOp{});
        p_sum_r1 = warp_row_reduce4(p_sum_r1, AddOp{});

        l_r0 = l_r0 * alpha_r0 + p_sum_r0;
        l_r1 = l_r1 * alpha_r1 + p_sum_r1;
        m_r0 = m_new_r0;
        m_r1 = m_new_r1;

        // Scale acc rows by alpha for next PV accumulate.
        #pragma unroll
        for (int ib = 0; ib < FRAG_PV / 4; ++ib) {
            acc[(ib << 2) + 0] *= alpha_r0;
            acc[(ib << 2) + 1] *= alpha_r0;
            acc[(ib << 2) + 2] *= alpha_r1;
            acc[(ib << 2) + 3] *= alpha_r1;
        }

        // Store P (bf16) into smem.
        store_p_to_smem(sP, p_frag, BLOCK_M, BLOCK_N, tid);
        __syncthreads();

        // ---- WGMMA #2: O += P @ V, M=BLOCK_M, N=D, K=BLOCK_N ----
        constexpr int K_STEPS_PV = BLOCK_N / 16;
        wgmma_fence();
        #pragma unroll
        for (int ks = 0; ks < K_STEPS_PV; ++ks) {
            uint64_t descPk = descP + (uint64_t)((ks * 256u) >> 4);
            uint64_t descVk = descV + (uint64_t)((ks * 256u) >> 4);
            if constexpr (D == 64) {
                wgmma_m64n64k16_bf16_ss(descPk, descVk, acc, 1);
            } else {
                wgmma_m64n128k16_bf16_ss(descPk, descVk, acc, 1);
            }
        }
        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // ---- finalize ----
    bool valid_r0 = (l_r0 > 0.0f) && (r0 < Q_rows_valid);
    bool valid_r1 = (l_r1 > 0.0f) && (r1 < Q_rows_valid);
    float inv_l_r0 = valid_r0 ? (1.0f / l_r0) : 0.0f;
    float inv_l_r1 = valid_r1 ? (1.0f / l_r1) : 0.0f;

    __nv_bfloat16* Out_base =
        Out + (size_t)pid_bh * Tq_p * D + (size_t)q_block_start * D;
    constexpr int N_BLOCKS_PV = FRAG_PV / 4;        // = D/8
    #pragma unroll
    for (int ib = 0; ib < N_BLOCKS_PV; ++ib) {
        int col0 = (ib << 3) + (tig << 1);
        float a0 = acc[(ib << 2) + 0] * inv_l_r0;
        float a1 = acc[(ib << 2) + 1] * inv_l_r0;
        float b0 = acc[(ib << 2) + 2] * inv_l_r1;
        float b1 = acc[(ib << 2) + 3] * inv_l_r1;

        if (r0 < Q_rows_valid) {
            __nv_bfloat16 ba = __float2bfloat16(valid_r0 ? a0 : 0.0f);
            __nv_bfloat16 bb = __float2bfloat16(valid_r0 ? a1 : 0.0f);
            unsigned packed = (unsigned)(*reinterpret_cast<unsigned short*>(&ba))
                            | ((unsigned)(*reinterpret_cast<unsigned short*>(&bb)) << 16);
            *reinterpret_cast<unsigned*>(Out_base + r0 * D + col0) = packed;
        }
        if (r1 < Q_rows_valid) {
            __nv_bfloat16 ba = __float2bfloat16(valid_r1 ? b0 : 0.0f);
            __nv_bfloat16 bb = __float2bfloat16(valid_r1 ? b1 : 0.0f);
            unsigned packed = (unsigned)(*reinterpret_cast<unsigned short*>(&ba))
                            | ((unsigned)(*reinterpret_cast<unsigned short*>(&bb)) << 16);
            *reinterpret_cast<unsigned*>(Out_base + r1 * D + col0) = packed;
        }
    }

    if (tig == 0) {
        float* Lse_base = Lse + (size_t)pid_bh * Tq_p + (size_t)q_block_start;
        if (r0 < Q_rows_valid) {
            Lse_base[r0] = valid_r0 ? (m_r0 + __logf(l_r0)) : -INFINITY;
        }
        if (r1 < Q_rows_valid) {
            Lse_base[r1] = valid_r1 ? (m_r1 + __logf(l_r1)) : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------
extern "C" void
launch_selected_attn_fwd(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const int* Idx,
    __nv_bfloat16* Out, float* Lse,
    int BH, int n_q_blocks, int Tq_p, int Tk_p, int D, int top_k,
    int offset, int causal, float sm_scale,
    cudaStream_t stream)
{
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    dim3 grid(n_q_blocks, BH);
    dim3 block(128, 1, 1);

    size_t smem_bytes_64 = (BLOCK_M * 64 + BLOCK_N * 64 + 64 * BLOCK_N + BLOCK_M * BLOCK_N)
                         * sizeof(__nv_bfloat16);
    size_t smem_bytes_128 = (BLOCK_M * 128 + BLOCK_N * 128 + 128 * BLOCK_N + BLOCK_M * BLOCK_N)
                          * sizeof(__nv_bfloat16);

    if (D == 64) {
        cudaFuncSetAttribute(selected_attn_fwd_kernel<64>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes_64);
        selected_attn_fwd_kernel<64><<<grid, block, smem_bytes_64, stream>>>(
            Q, K, V, Idx, Out, Lse, Tq_p, Tk_p, top_k, offset, causal, sm_scale);
    } else if (D == 128) {
        cudaFuncSetAttribute(selected_attn_fwd_kernel<128>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes_128);
        selected_attn_fwd_kernel<128><<<grid, block, smem_bytes_128, stream>>>(
            Q, K, V, Idx, Out, Lse, Tq_p, Tk_p, top_k, offset, causal, sm_scale);
    }
}

}  // namespace nsa_cuda
