#include "stage_registry.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>
#include <string>

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kWmma = 16;

namespace {

using nvcuda::wmma::accumulator;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::row_major;
using nvcuda::wmma::store_matrix_sync;

// Stage 6: Tensor Core WMMA. FP16 A/B, FP32 accumulate. 64x64 block tile,
// 16 warps each with a 16x16x16 fragment. K walks in 16-wide shared panels.

__global__ void wmmaSgemmKernel(const __half* a,
                                const __half* b,
                                float* c,
                                int m,
                                int n,
                                int k) {
  const int warp_id = static_cast<int>(threadIdx.y);
  const int warp_m = warp_id / 4;
  const int warp_n = warp_id % 4;
  const int block_row = static_cast<int>(blockIdx.y) * kBM;
  const int block_col = static_cast<int>(blockIdx.x) * kBN;

  fragment<matrix_a, kWmma, kWmma, kWmma, __half, row_major> a_frag;
  fragment<matrix_b, kWmma, kWmma, kWmma, __half, row_major> b_frag;
  fragment<accumulator, kWmma, kWmma, kWmma, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  __shared__ __half tile_a[kBM][kBK];
  __shared__ __half tile_b[kBK][kBN];

  const int linear = static_cast<int>(threadIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const int panels = k / kBK;

  for (int p = 0; p < panels; ++p) {
    const int k0 = p * kBK;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = linear + i * 512;
      const int r = idx / kBK;
      const int col = idx % kBK;
      const int g_row = block_row + r;
      const int g_col = k0 + col;
      tile_a[r][col] =
          (g_row < m && g_col < k) ? a[g_row * k + g_col] : __float2half(0.0f);
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = linear + i * 512;
      const int r = idx / kBN;
      const int col = idx % kBN;
      const int g_row = k0 + r;
      const int g_col = block_col + col;
      tile_b[r][col] =
          (g_row < k && g_col < n) ? b[g_row * n + g_col] : __float2half(0.0f);
    }

    __syncthreads();

    load_matrix_sync(a_frag, &tile_a[warp_m * kWmma][0], kBK);
    load_matrix_sync(b_frag, &tile_b[0][warp_n * kWmma], kBN);
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    __syncthreads();
  }

  const int c_row = block_row + warp_m * kWmma;
  const int c_col = block_col + warp_n * kWmma;
  if (c_row < m && c_col < n) {
    store_matrix_sync(c + c_row * n + c_col, c_frag, n, mem_row_major);
  }
}

}  // namespace

void launchWmmaSgemm(const __half* a,
                     const __half* b,
                     float* c,
                     int m,
                     int n,
                     int k) {
  if (m % kBM != 0 || n % kBN != 0 || k % kBK != 0) {
    throw std::runtime_error(
        "WMMA kernel requires M and N multiples of 64 and K a multiple of 16, got " +
        std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k));
  }

  dim3 block(32, 16);
  dim3 grid(n / kBN, m / kBM);
  wmmaSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeWmmaSgemm(int n) {
  const int tiles = (n + kBN - 1) / kBN;
  return {describeKernel("wmma", wmmaSgemmKernel, 32 * 16, tiles * tiles)};
}
