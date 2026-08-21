#include "stage_registry.cuh"

#include <cuda_runtime.h>

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;

namespace {
// Stage 4: Register-blocked SGEMM (thread tiling).
// A 64x64 C tile is owned by a 16x16 thread block. Each thread accumulates a
// 4x4 output patch in registers. K is walked in 16-wide panels. Arithmetic
// intensity jumps because each shared value feeds a 4x4 outer product.
// Layout credit: same idea as Simon Boehm's CUDA matmul writeup.

__global__ void registerBlockedSgemmKernel(const float* a,
                                           const float* b,
                                           float* c,
                                           int m,
                                           int n,
                                           int k) {
  __shared__ float tile_a[kBM][kBK + 1];
  __shared__ float tile_b[kBK][kBN + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int linear = ty * blockDim.x + tx;

  float results[kTM][kTN];
#pragma unroll
  for (int i = 0; i < kTM; ++i) {
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      results[i][j] = 0.0f;
    }
  }

  const int row0 = blockIdx.y * kBM + ty * kTM;
  const int col0 = blockIdx.x * kBN + tx * kTN;
  const int panels = (k + kBK - 1) / kBK;

  for (int p = 0; p < panels; ++p) {
    const int k0 = p * kBK;

    // 256 threads, 64*16 = 1024 elements: 4 scalar loads per thread.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = linear + i * 256;
      const int a_row = idx / kBK;
      const int a_col = idx % kBK;
      const int g_row = blockIdx.y * kBM + a_row;
      const int g_col = k0 + a_col;
      tile_a[a_row][a_col] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = linear + i * 256;
      const int b_row = idx / kBN;
      const int b_col = idx % kBN;
      const int g_row = k0 + b_row;
      const int g_col = blockIdx.x * kBN + b_col;
      tile_b[b_row][b_col] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kBK; ++inner) {
      float a_reg[kTM];
      float b_reg[kTN];
#pragma unroll
      for (int i = 0; i < kTM; ++i) {
        a_reg[i] = tile_a[ty * kTM + i][inner];
      }
#pragma unroll
      for (int j = 0; j < kTN; ++j) {
        b_reg[j] = tile_b[inner][tx * kTN + j];
      }
#pragma unroll
      for (int i = 0; i < kTM; ++i) {
#pragma unroll
        for (int j = 0; j < kTN; ++j) {
          results[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kTM; ++i) {
#pragma unroll
    for (int j = 0; j < kTN; ++j) {
      const int row = row0 + i;
      const int col = col0 + j;
      if (row < m && col < n) {
        c[row * n + col] = results[i][j];
      }
    }
  }
}

}  // namespace

void launchRegisterBlockedSgemm(const float* a,
                                const float* b,
                                float* c,
                                int m,
                                int n,
                                int k) {
  dim3 block(kBN / kTN, kBM / kTM);
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  registerBlockedSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeRegisterBlockedSgemm(int n) {
  const int tiles = (n + kBN - 1) / kBN;
  return {describeKernel(
      "register blocked", registerBlockedSgemmKernel, (kBN / kTN) * (kBM / kTM), tiles * tiles)};
}
