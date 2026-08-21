#include "stage_registry.cuh"

#include <cuda_runtime.h>

namespace {

// Stage 2: Coalesced SGEMM.
// Same arithmetic as Stage 1. The only mapping change: threadIdx.x drives the
// COLUMN, so consecutive lanes in a warp read 32 contiguous columns of B and
// write 32 contiguous columns of C (128-byte transactions). The A load
// broadcasts (row is constant across the warp).
// Both stages use dim3 block(32, 32). For square M=N the grid expressions
// evaluate to the same dim3; they would differ for non-square.
__global__ void coalescedSgemmKernel(const float* a,
                                     const float* b,
                                     float* c,
                                     int m,
                                     int n,
                                     int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

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

void launchCoalescedSgemm(const float* a,
                          const float* b,
                          float* c,
                          int m,
                          int n,
                          int k) {
  dim3 block(32, 32);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  coalescedSgemmKernel<<<grid, block>>>(a, b, c, m, n, k);
}

std::vector<KernelOccupancy> describeCoalescedSgemm(int n) {
  const int tiles = (n + 31) / 32;
  return {describeKernel("coalesced", coalescedSgemmKernel, 1024, tiles * tiles)};
}
