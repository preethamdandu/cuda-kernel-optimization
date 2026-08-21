#include "attn.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace {

// Unfused SDPA: three kernels, S and P live in global memory.
//   scores = scale * Q K^T     (seq x seq)
//   P      = softmax(scores)   (row-wise)
//   out    = P V               (seq x d)
// That materialization is the FlashAttention / TRT-LLM motivation: seq^2
// intermediates blow up memory and bandwidth before the matmuls even matter.

__global__ void qkKernel(const float* q,
                         const float* k,
                         float* scores,
                         int seq,
                         int head_dim,
                         float scale) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= seq || col >= seq) {
    return;
  }

  float acc = 0.0f;
  for (int p = 0; p < head_dim; ++p) {
    acc += q[row * head_dim + p] * k[col * head_dim + p];
  }
  scores[row * seq + col] = acc * scale;
}

__global__ void softmaxKernel(float* scores, int seq) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq) {
    return;
  }

  float* s = scores + row * seq;
  float row_max = -1.0e30f;
  for (int j = 0; j < seq; ++j) {
    row_max = fmaxf(row_max, s[j]);
  }
  float sum = 0.0f;
  for (int j = 0; j < seq; ++j) {
    const float e = expf(s[j] - row_max);
    s[j] = e;
    sum += e;
  }
  const float inv = 1.0f / sum;
  for (int j = 0; j < seq; ++j) {
    s[j] *= inv;
  }
}

__global__ void pvKernel(const float* p,
                         const float* v,
                         float* out,
                         int seq,
                         int head_dim) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= seq || col >= head_dim) {
    return;
  }

  float acc = 0.0f;
  for (int j = 0; j < seq; ++j) {
    acc += p[row * seq + j] * v[j * head_dim + col];
  }
  out[row * head_dim + col] = acc;
}

}  // namespace

void launchUnfusedAttention(const float* q,
                            const float* k,
                            const float* v,
                            float* scores,
                            float* out,
                            int seq,
                            int head_dim) {
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  dim3 block(32, 32);
  dim3 qk_grid((seq + 31) / 32, (seq + 31) / 32);
  qkKernel<<<qk_grid, block>>>(q, k, scores, seq, head_dim, scale);

  const int softmax_block = 128;
  softmaxKernel<<<(seq + softmax_block - 1) / softmax_block, softmax_block>>>(scores, seq);

  dim3 pv_grid((head_dim + 31) / 32, (seq + 31) / 32);
  pvKernel<<<pv_grid, block>>>(scores, v, out, seq, head_dim);
}

std::vector<KernelOccupancy> describeUnfusedAttention(int seq, int head_dim) {
  const int tiles = (seq + 31) / 32;
  return {
      describeKernel("unfused QK^T", qkKernel, 1024, tiles * tiles),
      describeKernel("unfused softmax", softmaxKernel, 128, (seq + 127) / 128),
      describeKernel("unfused PV", pvKernel, 1024, ((head_dim + 31) / 32) * tiles),
  };
}
