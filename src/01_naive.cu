#include "stage_registry.cuh"

#include <cuda_runtime.h>

namespace {

// Stage 1: Naive uncoalesced SGEMM.
// threadIdx.x drives the ROW, so consecutive lanes in a warp read 32 different
// rows of A (stride K) and write 32 different rows of C (stride N).
// Expected bottleneck: uncoalesced global loads/stores; ~32 sectors per A load
// and C store. The B load broadcasts (col is constant across the warp).
__global__ void naiveSgemmKernel(const float* a,
                                 const float* b,
                                 float* c,
                                 int m,
                                 int n,
                                 int k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    acc += a[row * k + inner] * b[inner * n + col];
  }

  c[row * n + col] = acc;
}

}  // namespace

void launchNaiveSgemm(const float* a,
                      const float* b,
                      float* c,
                      int m,
                      int n,
                      int k) {
  dim3 block(32, 32);
  dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
  naiveSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeNaiveSgemm(int n) {
  const int tiles = (n + 31) / 32;
  return {describeKernel("naive", naiveSgemmKernel, 1024, tiles * tiles)};
}
