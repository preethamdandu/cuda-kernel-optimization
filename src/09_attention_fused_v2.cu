#include "attn.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

// Fused SDPA v2. Same online-softmax math as v1, different thread mapping.
//
// v1 gave one Q row to one thread in a 32-thread block, so seq=1024 launched
// 32 blocks of 1 warp: 32 warps for the whole GPU, and acc[64] + s[32] per
// thread spilled to local memory.
//
// v2 gives one Q row to one warp in a 128-thread block (4 rows per block).
// seq=1024 launches 256 blocks of 4 warps = 1024 warps. head_dim 64 splits
// across 32 lanes as 2 accumulator floats each, so nothing spills.
//
// Within a K/V tile, lane j owns K row j: it computes the whole 64-long dot
// itself out of shared memory, which keeps s_j in a register and needs no
// shuffle. The softmax reductions across the 32 K rows are warp shuffles.

constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpsPerBlock * 32;
constexpr int kBc = 32;
constexpr int kHead = 64;
// 65 floats per row: (j * 65 + d) % 32 == (j + d) % 32, so the lane-varying
// reads of tile_k[lane][d] land on 32 distinct banks.
constexpr int kRowStride = kHead + 1;

__global__ void __launch_bounds__(kBlockThreads)
    fusedAttentionV2Kernel(const float* __restrict__ q,
                           const float* __restrict__ k,
                           const float* __restrict__ v,
                           float* __restrict__ out,
                           int seq,
                           float scale) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int q_row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
  const bool active = q_row < seq;

  __shared__ float tile_k[kBc][kRowStride];
  __shared__ float tile_v[kBc][kRowStride];
  __shared__ float tile_q[kWarpsPerBlock][kHead];
  __shared__ float tile_p[kWarpsPerBlock][kBc];

  for (int idx = tid; idx < kWarpsPerBlock * kHead; idx += kBlockThreads) {
    const int r = idx / kHead;
    const int c = idx % kHead;
    const int g_row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + r;
    tile_q[r][c] = (g_row < seq) ? q[g_row * kHead + c] : 0.0f;
  }

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int kv = 0; kv < seq; kv += kBc) {
    const int remaining = seq - kv;
    const int tile = remaining < kBc ? remaining : kBc;

    __syncthreads();
    for (int idx = tid; idx < tile * kHead; idx += kBlockThreads) {
      const int r = idx / kHead;
      const int c = idx % kHead;
      tile_k[r][c] = k[(kv + r) * kHead + c];
      tile_v[r][c] = v[(kv + r) * kHead + c];
    }
    __syncthreads();

    // Lane j owns K row j. tile_q[warp][d] is a broadcast read.
    float s_j = -INFINITY;
    if (lane < tile) {
      float dot = 0.0f;
#pragma unroll
      for (int d = 0; d < kHead; ++d) {
        dot += tile_q[warp][d] * tile_k[lane][d];
      }
      s_j = scale * dot;
    }

    float tile_max = s_j;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
    }

    // expf, not __expf: v1 uses expf, and the point of this kernel is to
    // isolate the thread mapping, not to also change the math.
    const float m_new = fmaxf(m_i, tile_max);
    const float alpha = expf(m_i - m_new);
    const float p_j = (lane < tile) ? expf(s_j - m_new) : 0.0f;

    float l_add = p_j;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      l_add += __shfl_xor_sync(0xffffffffu, l_add, offset);
    }

    tile_p[warp][lane] = p_j;
    __syncwarp();

    float pv0 = 0.0f;
    float pv1 = 0.0f;
    for (int j = 0; j < tile; ++j) {
      const float p = tile_p[warp][j];
      pv0 += p * tile_v[j][lane];
      pv1 += p * tile_v[j][lane + 32];
    }

    acc0 = acc0 * alpha + pv0;
    acc1 = acc1 * alpha + pv1;
    l_i = l_i * alpha + l_add;
    m_i = m_new;
  }

  if (active) {
    const float inv = 1.0f / l_i;
    out[q_row * kHead + lane] = acc0 * inv;
    out[q_row * kHead + lane + 32] = acc1 * inv;
  }
}

int gridBlocks(int seq) { return (seq + kWarpsPerBlock - 1) / kWarpsPerBlock; }

}  // namespace

void launchFusedAttentionV2(const float* q,
                            const float* k,
                            const float* v,
                            float* out,
                            int seq,
                            int head_dim) {
  if (head_dim != kHead) {
    throw std::runtime_error("fused attention v2 requires head_dim=" + std::to_string(kHead) +
                             ", got " + std::to_string(head_dim));
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  fusedAttentionV2Kernel<<<gridBlocks(seq), kBlockThreads>>>(q, k, v, out, seq, scale);
}

std::vector<KernelOccupancy> describeFusedAttentionV2(int seq) {
  return {describeKernel(
      "fused v2 (warp per Q row)", fusedAttentionV2Kernel, kBlockThreads, gridBlocks(seq))};
}
