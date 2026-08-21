#include "../src/attn.cuh"
#include "error_stats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 10;
constexpr int kHeadDim = 64;
constexpr float kRelTolerance = 1.0e-3f;

void checkCuda(cudaError_t status, const std::string& label) {
  if (status != cudaSuccess) {
    throw std::runtime_error(label + ": " + cudaGetErrorString(status));
  }
}

struct DeviceBuffer {
  float* ptr = nullptr;
  std::size_t count = 0;

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t element_count) : count(element_count) {
    if (count == 0) {
      return;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float)), "cudaMalloc");
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
    other.ptr = nullptr;
    other.count = 0;
  }

  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }
};

struct CudaEventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CudaEventPair() {
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");
  }

  ~CudaEventPair() {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
  }
};

std::vector<float> makeRandom(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(count);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

std::vector<int> parseSeqs(int argc, char** argv) {
  std::vector<int> seqs = {256, 512, 1024};
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--seqs") {
      seqs.clear();
      for (int j = i + 1; j < argc; ++j) {
        if (std::string(argv[j]).rfind("--", 0) == 0) {
          break;
        }
        seqs.push_back(std::stoi(argv[j]));
        i = j;
      }
    }
  }
  return seqs;
}

template <typename Launch>
float timeLaunch(const CudaEventPair& events, Launch&& launch) {
  for (int i = 0; i < kWarmupRuns; ++i) {
    launch();
  }
  checkCuda(cudaDeviceSynchronize(), "warmup sync");

  float total_ms = 0.0f;
  for (int i = 0; i < kTimedRuns; ++i) {
    checkCuda(cudaEventRecord(events.start), "record start");
    launch();
    checkCuda(cudaEventRecord(events.stop), "record stop");
    checkCuda(cudaEventSynchronize(events.stop), "event sync");
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, events.start, events.stop), "elapsed");
    total_ms += ms;
  }
  return total_ms / static_cast<float>(kTimedRuns);
}

void printDevice() {
  cudaDeviceProp props{};
  checkCuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  const double peak_dram_gbs = 2.0 * static_cast<double>(props.memoryClockRate) * 1.0e3 *
                               (static_cast<double>(props.memoryBusWidth) / 8.0) / 1.0e9;
  std::cout << "Device: " << props.name << "\n";
  std::cout << "SMs: " << props.multiProcessorCount << "\n";
  std::cout << "Max threads per SM: " << props.maxThreadsPerMultiProcessor << "\n";
  std::cout << "L2 cache: " << props.l2CacheSize / (1024 * 1024) << " MiB\n";
  std::cout << "Peak DRAM: " << std::fixed << std::setprecision(2) << peak_dram_gbs << " GB/s\n";
  std::cout << "head_dim: " << kHeadDim << " (single head, batch=1)\n";
  std::cout << "FLOPs counted as 4*seq*seq*d (QK^T + PV); softmax excluded\n";
  std::cout << "Unfused times QK + softmax + PV as one event pair\n\n";
}

void printOccupancy(int seq) {
  std::vector<KernelOccupancy> all = describeUnfusedAttention(seq, kHeadDim);
  for (const auto& entry : describeFusedAttention(seq)) {
    all.push_back(entry);
  }
  for (const auto& entry : describeFusedAttentionV2(seq)) {
    all.push_back(entry);
  }

  std::cout << "Launch geometry and occupancy at seq=" << seq
            << " (cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor,\n"
               "no Nsight counters needed):\n\n";
  std::cout << std::left << std::setw(30) << "Kernel" << std::right << std::setw(8) << "Regs"
            << std::setw(10) << "Smem B" << std::setw(10) << "Local B" << std::setw(9) << "Thr/blk"
            << std::setw(9) << "Blocks" << std::setw(12) << "Warps" << std::setw(12) << "Blk/SM"
            << std::setw(12) << "Occupancy" << "\n";
  for (const auto& info : all) {
    std::cout << std::left << std::setw(30) << info.name << std::right << std::setw(8)
              << info.registers_per_thread << std::setw(10) << info.static_shared_bytes
              << std::setw(10) << info.local_bytes_per_thread << std::setw(9) << info.block_threads
              << std::setw(9) << info.grid_blocks << std::setw(12) << info.total_warps
              << std::setw(12) << info.max_active_blocks_per_sm << std::setw(11) << std::fixed
              << std::setprecision(1) << info.occupancy * 100.0 << "%" << "\n";
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<int> seqs = parseSeqs(argc, argv);
    printDevice();
    if (!seqs.empty()) {
      printOccupancy(seqs.back());
    }

    std::cout << std::left << std::setw(7) << "seq" << std::right << std::setw(12) << "unfused_ms"
              << std::setw(11) << "v1_ms" << std::setw(11) << "v2_ms" << std::setw(10) << "v1_spd"
              << std::setw(10) << "v2_spd" << std::setw(11) << "v2/v1" << std::setw(13)
              << "unfused_GF/s" << std::setw(11) << "v2_GF/s" << std::setw(9) << "S_MiB"
              << std::setw(12) << "v2_maxabs" << std::setw(11) << "v2_fail" << "\n";

    CudaEventPair events;

    for (int seq : seqs) {
      if (seq <= 0) {
        throw std::runtime_error("seq must be positive");
      }
      if (seq > 1024) {
        throw std::runtime_error("fused v1 supports seq_len <= 1024");
      }

      const std::size_t qkv = static_cast<std::size_t>(seq) * static_cast<std::size_t>(kHeadDim);
      const std::size_t scores_n = static_cast<std::size_t>(seq) * static_cast<std::size_t>(seq);

      const auto q_h = makeRandom(qkv, 42);
      const auto k_h = makeRandom(qkv, 43);
      const auto v_h = makeRandom(qkv, 44);

      DeviceBuffer q(qkv), k(qkv), v(qkv), scores(scores_n);
      DeviceBuffer out_unfused(qkv), out_v1(qkv), out_v2(qkv);
      checkCuda(cudaMemcpy(q.ptr, q_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D Q");
      checkCuda(cudaMemcpy(k.ptr, k_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D K");
      checkCuda(cudaMemcpy(v.ptr, v_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D V");

      const float unfused_ms = timeLaunch(events, [&]() {
        launchUnfusedAttention(q.ptr, k.ptr, v.ptr, scores.ptr, out_unfused.ptr, seq, kHeadDim);
      });
      checkCuda(cudaGetLastError(), "unfused launch");

      const float v1_ms = timeLaunch(events, [&]() {
        launchFusedAttention(q.ptr, k.ptr, v.ptr, out_v1.ptr, seq, kHeadDim);
      });
      checkCuda(cudaGetLastError(), "fused v1 launch");

      const float v2_ms = timeLaunch(events, [&]() {
        launchFusedAttentionV2(q.ptr, k.ptr, v.ptr, out_v2.ptr, seq, kHeadDim);
      });
      checkCuda(cudaGetLastError(), "fused v2 launch");
      checkCuda(cudaDeviceSynchronize(), "final sync");

      std::vector<float> unfused_h(qkv), v1_h(qkv), v2_h(qkv);
      checkCuda(cudaMemcpy(unfused_h.data(), out_unfused.ptr, qkv * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H unfused");
      checkCuda(cudaMemcpy(v1_h.data(), out_v1.ptr, qkv * sizeof(float), cudaMemcpyDeviceToHost),
                "D2H v1");
      checkCuda(cudaMemcpy(v2_h.data(), out_v2.ptr, qkv * sizeof(float), cudaMemcpyDeviceToHost),
                "D2H v2");

      const bench::ErrorStats v1_err = bench::computeErrorStats(v1_h, unfused_h, kRelTolerance);
      const bench::ErrorStats v2_err = bench::computeErrorStats(v2_h, unfused_h, kRelTolerance);

      const double flops = 4.0 * static_cast<double>(seq) * static_cast<double>(seq) *
                           static_cast<double>(kHeadDim);
      const double scores_mib = static_cast<double>(scores_n * sizeof(float)) / (1024.0 * 1024.0);

      std::cout << std::left << std::setw(7) << seq << std::right << std::fixed
                << std::setprecision(4) << std::setw(12) << unfused_ms << std::setw(11) << v1_ms
                << std::setw(11) << v2_ms << std::setprecision(2) << std::setw(10)
                << unfused_ms / v1_ms << std::setw(10) << unfused_ms / v2_ms << std::setw(11)
                << v1_ms / v2_ms << std::setprecision(1) << std::setw(13)
                << flops / (static_cast<double>(unfused_ms) * 1.0e6) << std::setw(11)
                << flops / (static_cast<double>(v2_ms) * 1.0e6) << std::setprecision(2)
                << std::setw(9) << scores_mib << std::scientific << std::setprecision(2)
                << std::setw(12) << v2_err.max_abs_diff << std::setw(11) << v2_err.fail_elems
                << "\n";

      if (v1_err.max_rel_diff > kRelTolerance) {
        throw std::runtime_error("fused v1 rel error " + std::to_string(v1_err.max_rel_diff) +
                                 " exceeds " + std::to_string(kRelTolerance));
      }
      if (v2_err.max_rel_diff > kRelTolerance) {
        throw std::runtime_error("fused v2 rel error " + std::to_string(v2_err.max_rel_diff) +
                                 " exceeds " + std::to_string(kRelTolerance));
      }
    }

    std::cout << "\nv1_spd / v2_spd are unfused/fused, so >1 means fusion won.\n";
    std::cout << "PASS: both fused kernels match unfused within rel " << std::scientific
              << std::setprecision(1) << kRelTolerance << " (host-side compare).\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
