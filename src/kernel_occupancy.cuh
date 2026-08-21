#pragma once

// Register counts, spill bytes, and occupancy come from the compiler and the
// runtime API rather than Nsight Compute. Both Colab and RunPod deny the
// hardware counters ncu needs, but none of this needs them.

#include <cuda_runtime.h>

#include <cstddef>

struct KernelOccupancy {
  const char* name = "";
  int registers_per_thread = 0;
  int static_shared_bytes = 0;
  int local_bytes_per_thread = 0;
  int block_threads = 0;
  int max_active_blocks_per_sm = 0;
  double occupancy = 0.0;
  int grid_blocks = 0;
  long long total_warps = 0;
};

template <typename KernelPtr>
inline KernelOccupancy describeKernel(const char* name,
                                      KernelPtr kernel,
                                      int block_threads,
                                      int grid_blocks,
                                      std::size_t dynamic_shared_bytes = 0) {
  KernelOccupancy info;
  info.name = name;
  info.block_threads = block_threads;
  info.grid_blocks = grid_blocks;
  info.total_warps = static_cast<long long>(grid_blocks) * ((block_threads + 31) / 32);

  const void* entry = reinterpret_cast<const void*>(kernel);

  cudaFuncAttributes attrs{};
  if (cudaFuncGetAttributes(&attrs, entry) == cudaSuccess) {
    info.registers_per_thread = attrs.numRegs;
    info.static_shared_bytes = static_cast<int>(attrs.sharedSizeBytes);
    info.local_bytes_per_thread = static_cast<int>(attrs.localSizeBytes);
  }

  int max_blocks = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_blocks, entry, block_threads, dynamic_shared_bytes) == cudaSuccess) {
    info.max_active_blocks_per_sm = max_blocks;

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess &&
        props.maxThreadsPerMultiProcessor > 0) {
      info.occupancy = static_cast<double>(max_blocks) * block_threads /
                       static_cast<double>(props.maxThreadsPerMultiProcessor);
    }
  }

  return info;
}
