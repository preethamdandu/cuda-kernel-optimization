#include "../src/attn.cuh"

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
constexpr float kRelDenomFloor = 1.0e-12f;

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

struct ErrorStats {
  float max_abs_diff = 0.0f;
  float max_abs_ref = 0.0f;
  float max_rel_diff = 0.0f;
  std::size_t mismatch_count = 0;
};

ErrorStats compareHost(const std::vector<float>& out, const std::vector<float>& ref) {
  ErrorStats stats;
  if (out.size() != ref.size()) {
    throw std::runtime_error("output and reference sizes differ");
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float diff = std::fabs(out[i] - ref[i]);
    const float abs_ref = std::fabs(ref[i]);
    if (diff > stats.max_abs_diff) {
      stats.max_abs_diff = diff;
    }
    if (abs_ref > stats.max_abs_ref) {
      stats.max_abs_ref = abs_ref;
    }
  }
  const float denom = std::max(stats.max_abs_ref, kRelDenomFloor);
  stats.max_rel_diff = stats.max_abs_diff / denom;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float diff = std::fabs(out[i] - ref[i]);
    if (diff > kRelTolerance * denom) {
      ++stats.mismatch_count;
    }
  }
  return stats;
}

void printDevice() {
  cudaDeviceProp props{};
  checkCuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  const double peak_dram_gbs = 2.0 * static_cast<double>(props.memoryClockRate) * 1.0e3 *
                               (static_cast<double>(props.memoryBusWidth) / 8.0) / 1.0e9;
  std::cout << "Device: " << props.name << "\n";
  std::cout << "SMs: " << props.multiProcessorCount << "\n";
  std::cout << "Peak DRAM: " << std::fixed << std::setprecision(2) << peak_dram_gbs << " GB/s\n";
  std::cout << "head_dim: " << kHeadDim << " (single head, batch=1)\n";
  std::cout << "FLOPs counted as 4*seq*seq*d (QK^T + PV); softmax excluded\n";
  std::cout << "Unfused times QK + softmax + PV as one event pair\n\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<int> seqs = parseSeqs(argc, argv);
    printDevice();

    std::cout << std::left << std::setw(8) << "seq" << std::right << std::setw(12)
              << "unfused_ms" << std::setw(12) << "fused_ms" << std::setw(10) << "speedup"
              << std::setw(14) << "unfused_GF/s" << std::setw(14) << "fused_GF/s" << std::setw(12)
              << "S_MiB" << std::setw(12) << "max_abs" << std::setw(12) << "max_rel" << std::setw(12)
              << "mismatch" << "\n";

    CudaEventPair events;

    for (int seq : seqs) {
      if (seq <= 0 || seq > 1024) {
        throw std::runtime_error("seq must be in (0, 1024]");
      }

      const std::size_t qkv = static_cast<std::size_t>(seq) * static_cast<std::size_t>(kHeadDim);
      const std::size_t scores_n = static_cast<std::size_t>(seq) * static_cast<std::size_t>(seq);

      const auto q_h = makeRandom(qkv, 42);
      const auto k_h = makeRandom(qkv, 43);
      const auto v_h = makeRandom(qkv, 44);

      DeviceBuffer q(qkv), k(qkv), v(qkv), scores(scores_n), out_unfused(qkv), out_fused(qkv);
      checkCuda(cudaMemcpy(q.ptr, q_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D Q");
      checkCuda(cudaMemcpy(k.ptr, k_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D K");
      checkCuda(cudaMemcpy(v.ptr, v_h.data(), qkv * sizeof(float), cudaMemcpyHostToDevice), "H2D V");

      const float unfused_ms = timeLaunch(events, [&]() {
        launchUnfusedAttention(q.ptr, k.ptr, v.ptr, scores.ptr, out_unfused.ptr, seq, kHeadDim);
      });
      checkCuda(cudaGetLastError(), "unfused launch");

      const float fused_ms = timeLaunch(events, [&]() {
        launchFusedAttention(q.ptr, k.ptr, v.ptr, out_fused.ptr, seq, kHeadDim);
      });
      checkCuda(cudaGetLastError(), "fused launch");
      checkCuda(cudaDeviceSynchronize(), "final sync");

      std::vector<float> unfused_h(qkv), fused_h(qkv);
      checkCuda(cudaMemcpy(unfused_h.data(), out_unfused.ptr, qkv * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H unfused");
      checkCuda(cudaMemcpy(fused_h.data(), out_fused.ptr, qkv * sizeof(float), cudaMemcpyDeviceToHost),
                "D2H fused");

      const ErrorStats err = compareHost(fused_h, unfused_h);
      const double flops = 4.0 * static_cast<double>(seq) * static_cast<double>(seq) *
                           static_cast<double>(kHeadDim);
      const double unfused_gflops = flops / (static_cast<double>(unfused_ms) * 1.0e6);
      const double fused_gflops = flops / (static_cast<double>(fused_ms) * 1.0e6);
      const double scores_mib = static_cast<double>(scores_n * sizeof(float)) / (1024.0 * 1024.0);
      const float speedup = unfused_ms / fused_ms;

      std::cout << std::left << std::setw(8) << seq << std::right << std::fixed
                << std::setprecision(4) << std::setw(12) << unfused_ms << std::setw(12) << fused_ms
                << std::setprecision(2) << std::setw(10) << speedup << std::setprecision(1)
                << std::setw(14) << unfused_gflops << std::setw(14) << fused_gflops
                << std::setprecision(3) << std::setw(12) << scores_mib << std::scientific
                << std::setprecision(2) << std::setw(12) << err.max_abs_diff << std::setw(12)
                << err.max_rel_diff << std::setw(12) << err.mismatch_count << "\n";

      if (err.max_rel_diff > kRelTolerance) {
        throw std::runtime_error("fused vs unfused relative error " +
                                 std::to_string(err.max_rel_diff) + " exceeds " +
                                 std::to_string(kRelTolerance));
      }
    }

    std::cout << "\nPASS: fused matches unfused within rel " << kRelTolerance
              << " (host-side compare).\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
