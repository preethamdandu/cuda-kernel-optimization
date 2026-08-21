#include "stage_registry.cuh"

#include <cuda_runtime.h>

constexpr int kTile = 32;

namespace {
// Stage 3: Shared-memory tiled SGEMM.
// Each 32x32 thread block owns a 32x32 C tile. It walks K in 32-wide chunks,
// loading an A tile and a B tile into shared memory so each global load is
// reused 32 times. Output mapping matches Stage 2 (threadIdx.x -> column).
// Shared tiles are padded by 1 column to avoid bank conflicts on the inner
// product. Two __syncthreads(): one after the loads, one after the MAC so
// the next K-tile does not overwrite values still in use.

__global__ void tiledSgemmKernel(const float* a,
                                 const float* b,
                                 float* c,
                                 int m,
                                 int n,
                                 int k) {
  __shared__ float tile_a[kTile][kTile + 1];
  __shared__ float tile_b[kTile][kTile + 1];

  const int col = blockIdx.x * kTile + threadIdx.x;
  const int row = blockIdx.y * kTile + threadIdx.y;

  float acc = 0.0f;
  const int tiles = (k + kTile - 1) / kTile;

  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * kTile + threadIdx.x;
    const int b_row = t * kTile + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kTile; ++inner) {
      acc += tile_a[threadIdx.y][inner] * tile_b[inner][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

}  // namespace

void launchTiledSgemm(const float* a,
                      const float* b,
                      float* c,
                      int m,
                      int n,
                      int k) {
  dim3 block(kTile, kTile);
  dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  tiledSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeTiledSgemm(int n) {
  const int tiles = (n + kTile - 1) / kTile;
  return {describeKernel("tiled", tiledSgemmKernel, kTile * kTile, tiles * tiles)};
}
