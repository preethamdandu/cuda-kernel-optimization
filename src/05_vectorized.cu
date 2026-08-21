#include "stage_registry.cuh"

#include <cuda_runtime.h>

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kTM = 4;
constexpr int kTN = 4;

namespace {
// Stage 5: Same 64x64 / 4x4 register blocking as Stage 4, but A and B tiles
// move global -> shared as float4. Cuts load instructions 4x on the K-panel
// copies. K and N are multiples of 4 in this benchmark (1024/2048/4096).

__device__ __forceinline__ float loadBound(const float* ptr, int row, int col, int rows, int cols, int ld) {
  return (row < rows && col < cols) ? ptr[row * ld + col] : 0.0f;
}

__global__ void vectorizedSgemmKernel(const float* a,
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

    // 256 threads, 256 float4s: one vector load of A and one of B per thread.
    {
      const int a_row = linear / 4;
      const int a_col = (linear % 4) * 4;
      const int g_row = blockIdx.y * kBM + a_row;
      const int g_col = k0 + a_col;
      if (g_row < m && g_col + 3 < k) {
        const float4 v = *reinterpret_cast<const float4*>(&a[g_row * k + g_col]);
        tile_a[a_row][a_col + 0] = v.x;
        tile_a[a_row][a_col + 1] = v.y;
        tile_a[a_row][a_col + 2] = v.z;
        tile_a[a_row][a_col + 3] = v.w;
      } else {
        tile_a[a_row][a_col + 0] = loadBound(a, g_row, g_col + 0, m, k, k);
        tile_a[a_row][a_col + 1] = loadBound(a, g_row, g_col + 1, m, k, k);
        tile_a[a_row][a_col + 2] = loadBound(a, g_row, g_col + 2, m, k, k);
        tile_a[a_row][a_col + 3] = loadBound(a, g_row, g_col + 3, m, k, k);
      }
    }

    {
      const int b_row = linear / 16;
      const int b_col = (linear % 16) * 4;
      const int g_row = k0 + b_row;
      const int g_col = blockIdx.x * kBN + b_col;
      if (g_row < k && g_col + 3 < n) {
        const float4 v = *reinterpret_cast<const float4*>(&b[g_row * n + g_col]);
        tile_b[b_row][b_col + 0] = v.x;
        tile_b[b_row][b_col + 1] = v.y;
        tile_b[b_row][b_col + 2] = v.z;
        tile_b[b_row][b_col + 3] = v.w;
      } else {
        tile_b[b_row][b_col + 0] = loadBound(b, g_row, g_col + 0, k, n, n);
        tile_b[b_row][b_col + 1] = loadBound(b, g_row, g_col + 1, k, n, n);
        tile_b[b_row][b_col + 2] = loadBound(b, g_row, g_col + 2, k, n, n);
        tile_b[b_row][b_col + 3] = loadBound(b, g_row, g_col + 3, k, n, n);
      }
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

void launchVectorizedSgemm(const float* a,
                           const float* b,
                           float* c,
                           int m,
                           int n,
                           int k) {
  dim3 block(kBN / kTN, kBM / kTM);
  dim3 grid((n + kBN - 1) / kBN, (m + kBM - 1) / kBM);
  vectorizedSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeVectorizedSgemm(int n) {
  const int tiles = (n + kBN - 1) / kBN;
  return {describeKernel(
      "vectorized", vectorizedSgemmKernel, (kBN / kTN) * (kBM / kTM), tiles * tiles)};
}
