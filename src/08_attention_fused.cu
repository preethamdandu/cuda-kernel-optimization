#include "attn.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBr = 32;
constexpr int kBc = 32;
constexpr int kHead = 64;

__global__ void fusedAttentionKernel(const float* q,
                                     const float* k,
                                     const float* v,
                                     float* out,
                                     int seq,
                                     float scale) {
  const int q_row = static_cast<int>(blockIdx.x) * kBr + static_cast<int>(threadIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

  __shared__ float tile_k[kBc][kHead + 1];
  __shared__ float tile_v[kBc][kHead + 1];

  float acc[kHead];
#pragma unroll
  for (int p = 0; p < kHead; ++p) {
    acc[p] = 0.0f;
  }
  float m_i = -1.0e30f;
  float l_i = 0.0f;

  const float* q_ptr = (q_row < seq) ? (q + q_row * kHead) : nullptr;

  for (int kv = 0; kv < seq; kv += kBc) {
    const int remaining = seq - kv;
    const int tile = remaining < kBc ? remaining : kBc;

    for (int idx = tid; idx < tile * kHead; idx += kBr) {
      const int r = idx / kHead;
      const int c = idx % kHead;
      tile_k[r][c] = k[(kv + r) * kHead + c];
      tile_v[r][c] = v[(kv + r) * kHead + c];
    }
    __syncthreads();

    if (q_ptr != nullptr) {
      float s[kBc];
      float tile_max = -1.0e30f;
      for (int j = 0; j < tile; ++j) {
        float dot = 0.0f;
#pragma unroll
        for (int p = 0; p < kHead; ++p) {
          dot += q_ptr[p] * tile_k[j][p];
        }
        s[j] = scale * dot;
        tile_max = fmaxf(tile_max, s[j]);
      }

      const float m_new = fmaxf(m_i, tile_max);
      const float alpha = expf(m_i - m_new);
      float l_add = 0.0f;
      for (int j = 0; j < tile; ++j) {
        s[j] = expf(s[j] - m_new);
        l_add += s[j];
      }

#pragma unroll
      for (int p = 0; p < kHead; ++p) {
        float pv = 0.0f;
        for (int j = 0; j < tile; ++j) {
          pv += s[j] * tile_v[j][p];
        }
        acc[p] = acc[p] * alpha + pv;
      }
      l_i = l_i * alpha + l_add;
      m_i = m_new;
    }

    __syncthreads();
  }

  if (q_ptr != nullptr) {
    const float inv = 1.0f / l_i;
#pragma unroll
    for (int p = 0; p < kHead; ++p) {
      out[q_row * kHead + p] = acc[p] * inv;
    }
  }
}

}  // namespace

void launchFusedAttention(const float* q,
                          const float* k,
                          const float* v,
                          float* out,
                          int seq,
                          int head_dim) {
  if (head_dim != kHead) {
    throw std::runtime_error("fused attention requires head_dim=" + std::to_string(kHead) +
                             ", got " + std::to_string(head_dim));
  }
  if (seq > 1024) {
    throw std::runtime_error("fused attention supports seq_len <= 1024");
  }

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const int blocks = (seq + kBr - 1) / kBr;
  fusedAttentionKernel<<<blocks, kBr>>>(q, k, v, out, seq, scale);
}

std::vector<KernelOccupancy> describeFusedAttention(int seq) {
  return {describeKernel(
      "fused v1 (thread per Q row)", fusedAttentionKernel, kBr, (seq + kBr - 1) / kBr)};
}
