#include "../src/stage_registry.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWarmupRuns = 3;
constexpr int kTimedRuns = 10;
constexpr float kRelDenomFloor = 1.0e-12f;

void checkCuda(cudaError_t status, const std::string& label) {
  if (status != cudaSuccess) {
    throw std::runtime_error(label + ": " + cudaGetErrorString(status));
  }
}

void checkCublas(cublasStatus_t status, const std::string& label) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(label + " failed with cuBLAS status " + std::to_string(status));
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
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float));
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), count(other.count) {
    other.ptr = nullptr;
    other.count = 0;
  }

  ~DeviceBuffer() { reset(); }

  void reset() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
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

struct BenchmarkResult {
  float milliseconds = 0.0f;
  double gflops = 0.0;
  double requested_gbs = 0.0;
  float max_abs_diff = 0.0f;
  float max_abs_ref = 0.0f;
  float max_rel_diff = 0.0f;
  std::size_t mismatch_count = 0;
};

std::vector<int> parseSizes(int argc, char** argv) {
  std::vector<int> sizes = {1024, 2048, 4096};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--sizes") {
      sizes.clear();
      for (int j = i + 1; j < argc; ++j) {
        const std::string next = argv[j];
        if (next.rfind("--", 0) == 0) {
          break;
        }
        sizes.push_back(std::stoi(next));
        i = j;
      }
    }
  }
  return sizes;
}

std::string parseStageName(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--stage" && i + 1 < argc) {
      return argv[i + 1];
    }
  }
  return "naive";
}

StageDefinition getStageOrThrow(const std::string& stage_name) {
  const auto stages = getRegisteredStages();
  for (const auto& stage : stages) {
    if (stage.name == stage_name) {
      return stage;
    }
  }

  std::ostringstream message;
  message << "Unknown stage '" << stage_name << "'. Available stages:";
  for (const auto& stage : stages) {
    message << " " << stage.name;
  }
  throw std::runtime_error(message.str());
}

std::vector<float> makeRandomMatrix(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> values(count);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

struct ErrorStats {
  float max_abs_diff = 0.0f;
  float max_abs_ref = 0.0f;
  float max_rel_diff = 0.0f;
  std::size_t mismatch_count = 0;
  std::size_t first_mismatch = static_cast<std::size_t>(-1);
};

ErrorStats computeErrorStats(const std::vector<float>& out, const std::vector<float>& ref) {
  // Do not use std::max here. nvcc -O3 has been observed to collapse
  // `max_diff = std::max(max_diff, fabs(a-b))` into a constant 0 at N=4096
  // while a separate max|ref| loop still sees real values.
  ErrorStats stats;
  const std::size_t count = out.size();
  if (ref.size() != count) {
    throw std::runtime_error("output and reference sizes differ");
  }
  // nvcc -O3 at N=4096 has twice collapsed this reduction to "all equal"
  // (std::max, then a non-volatile pointer loop with 0 mismatches while
  // max|ref| was 110). Force the loads; do not let the compiler prove aliasing.
  volatile const float* out_ptr = out.data();
  volatile const float* ref_ptr = ref.data();
  float max_diff = 0.0f;
  float max_ref = 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    const float ref_val = ref_ptr[i];
    const float out_val = out_ptr[i];
    const float diff = std::fabs(out_val - ref_val);
    const float abs_ref = std::fabs(ref_val);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (abs_ref > max_ref) {
      max_ref = abs_ref;
    }
    if (diff > 0.0f) {
      ++stats.mismatch_count;
      if (stats.first_mismatch == static_cast<std::size_t>(-1)) {
        stats.first_mismatch = i;
      }
    }
  }
  stats.max_abs_diff = max_diff;
  stats.max_abs_ref = max_ref;
  stats.max_rel_diff = max_diff / std::max(max_ref, kRelDenomFloor);
  return stats;
}

void launchStage(const StageDefinition& stage,
                 const float* a,
                 const float* b,
                 float* c,
                 int m,
                 int n,
                 int k,
                 const char* label) {
  stage.launcher(a, b, c, m, n, k);
  checkCuda(cudaGetLastError(), label);
}

void runCublasSgemm(cublasHandle_t handle,
                    const StageDefinition& stage,
                    const float* a,
                    const float* b,
                    float* c,
                    int m,
                    int n,
                    int k,
                    const char* label) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Stage 6 will switch to FP16 here (cublasGemmEx, CUDA_R_16F inputs,
  // CUDA_R_32F accumulate). Every current stage is FP32, so we always take
  // the cublasSgemm branch below.
  if (stage.use_tensor_core_baseline) {
    // Placeholder: Tensor Core FP16 baseline is not implemented yet.
  }

  // cuBLAS uses column-major order. Swapping A/B computes the same row-major result.
  checkCublas(
      cublasSgemm(handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  n,
                  m,
                  k,
                  &alpha,
                  b,
                  n,
                  a,
                  k,
                  &beta,
                  c,
                  n),
      label);
}

double sgemmFlops(int m, int n, int k) {
  return 2.0 * static_cast<double>(m) * n * k;
}

double requestedBytes(int m, int n, int k) {
  return (2.0 * static_cast<double>(m) * n * k + 1.0 * static_cast<double>(m) * n) * 4.0;
}

BenchmarkResult runCustomKernel(const StageDefinition& stage, int m, int n, int k) {
  const std::size_t a_count = static_cast<std::size_t>(m) * k;
  const std::size_t b_count = static_cast<std::size_t>(k) * n;
  const std::size_t c_count = static_cast<std::size_t>(m) * n;

  const std::vector<float> host_a = makeRandomMatrix(a_count, 42);
  const std::vector<float> host_b = makeRandomMatrix(b_count, 43);
  std::vector<float> host_c(c_count, 0.0f);
  std::vector<float> host_reference(c_count, 0.0f);

  DeviceBuffer dev_a(a_count);
  DeviceBuffer dev_b(b_count);
  DeviceBuffer dev_c(c_count);
  DeviceBuffer dev_reference(c_count);

  checkCuda(cudaMemcpy(dev_a.ptr, host_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice),
            "Copy A to device");
  checkCuda(cudaMemcpy(dev_b.ptr, host_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice),
            "Copy B to device");
  checkCuda(cudaMemset(dev_c.ptr, 0, c_count * sizeof(float)), "Zero custom output");
  checkCuda(cudaMemset(dev_reference.ptr, 0, c_count * sizeof(float)), "Zero cuBLAS output");

  cublasHandle_t handle = nullptr;
  checkCublas(cublasCreate(&handle), "cublasCreate");

  runCublasSgemm(handle, stage, dev_a.ptr, dev_b.ptr, dev_reference.ptr, m, n, k, "cublasSgemm");

  launchStage(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Kernel launch");
  checkCuda(cudaDeviceSynchronize(), "Kernel sync");

  checkCuda(cudaMemcpy(host_c.data(), dev_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost),
            "Copy custom output to host");
  checkCuda(
      cudaMemcpy(host_reference.data(), dev_reference.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost),
      "Copy cuBLAS output to host");

  BenchmarkResult result;
  const ErrorStats error = computeErrorStats(host_c, host_reference);
  result.max_abs_diff = error.max_abs_diff;
  result.max_abs_ref = error.max_abs_ref;
  result.max_rel_diff = error.max_rel_diff;
  result.mismatch_count = error.mismatch_count;

  CudaEventPair timer;
  for (int run = 0; run < kWarmupRuns; ++run) {
    launchStage(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Warmup kernel launch");
  }
  checkCuda(cudaDeviceSynchronize(), "Warmup sync");

  checkCuda(cudaEventRecord(timer.start), "Record start event");
  for (int run = 0; run < kTimedRuns; ++run) {
    launchStage(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Timed kernel launch");
  }
  checkCuda(cudaEventRecord(timer.stop), "Record stop event");
  checkCuda(cudaEventSynchronize(timer.stop), "Synchronize stop event");
  checkCuda(cudaEventElapsedTime(&result.milliseconds, timer.start, timer.stop), "Elapsed time");

  result.milliseconds /= static_cast<float>(kTimedRuns);
  result.gflops = sgemmFlops(m, n, k) / (result.milliseconds * 1.0e6);
  result.requested_gbs = requestedBytes(m, n, k) / (result.milliseconds * 1.0e6);

  checkCublas(cublasDestroy(handle), "cublasDestroy");
  return result;
}

double runCublasBaseline(const StageDefinition& stage, int m, int n, int k) {
  const std::size_t a_count = static_cast<std::size_t>(m) * k;
  const std::size_t b_count = static_cast<std::size_t>(k) * n;
  const std::size_t c_count = static_cast<std::size_t>(m) * n;

  const std::vector<float> host_a = makeRandomMatrix(a_count, 42);
  const std::vector<float> host_b = makeRandomMatrix(b_count, 43);

  DeviceBuffer dev_a(a_count);
  DeviceBuffer dev_b(b_count);
  DeviceBuffer dev_c(c_count);

  checkCuda(cudaMemcpy(dev_a.ptr, host_a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice),
            "Copy A to device");
  checkCuda(cudaMemcpy(dev_b.ptr, host_b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice),
            "Copy B to device");
  checkCuda(cudaMemset(dev_c.ptr, 0, c_count * sizeof(float)), "Zero cuBLAS baseline output");

  cublasHandle_t handle = nullptr;
  checkCublas(cublasCreate(&handle), "cublasCreate");

  CudaEventPair timer;

  for (int run = 0; run < kWarmupRuns; ++run) {
    runCublasSgemm(handle, stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "cublasSgemm warmup");
  }
  checkCuda(cudaDeviceSynchronize(), "cuBLAS warmup sync");

  float milliseconds = 0.0f;
  checkCuda(cudaEventRecord(timer.start), "Record cuBLAS start event");
  for (int run = 0; run < kTimedRuns; ++run) {
    runCublasSgemm(handle, stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "cublasSgemm timed");
  }
  checkCuda(cudaEventRecord(timer.stop), "Record cuBLAS stop event");
  checkCuda(cudaEventSynchronize(timer.stop), "Synchronize cuBLAS stop event");
  checkCuda(cudaEventElapsedTime(&milliseconds, timer.start, timer.stop), "cuBLAS elapsed time");

  checkCublas(cublasDestroy(handle), "cublasDestroy");

  milliseconds /= static_cast<float>(kTimedRuns);
  return sgemmFlops(m, n, k) / (milliseconds * 1.0e6);
}

void printDeviceHeader() {
  cudaDeviceProp props{};
  checkCuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");

  const double peak_dram_gbs =
      2.0 * static_cast<double>(props.memoryClockRate) * 1.0e3 *
      (static_cast<double>(props.memoryBusWidth) / 8.0) / 1.0e9;

  std::cout << "Device: " << props.name << "\n";
  std::cout << "SMs: " << props.multiProcessorCount << "\n";
  std::cout << "Clock: " << props.clockRate << " kHz\n";
  std::cout << "Memory clock: " << props.memoryClockRate << " kHz\n";
  std::cout << "Memory bus width: " << props.memoryBusWidth << " bits\n";
  std::cout << "Peak DRAM: " << std::fixed << std::setprecision(2) << peak_dram_gbs << " GB/s\n\n";
}

void printUsage() {
  std::cout << "Usage: ./bench [--stage naive|coalesced|tiled|register|vectorized] "
               "[--sizes 1024 2048 4096]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help") {
        printUsage();
        return 0;
      }
    }

    printDeviceHeader();

    const std::string stage_name = parseStageName(argc, argv);
    const StageDefinition stage = getStageOrThrow(stage_name);
    const std::vector<int> sizes = parseSizes(argc, argv);

    std::cout << "Stage: " << stage.name << "\n";
    std::cout << stage.description << "\n\n";

    std::cout << std::left << std::setw(10) << "M=N=K"
              << std::setw(14) << "Kernel GF/s"
              << std::setw(14) << "cuBLAS GF/s"
              << std::setw(12) << "% cuBLAS"
              << std::setw(12) << "Avg ms"
              << std::setw(14) << "Req GB/s"
              << std::setw(14) << "Max abs err"
              << std::setw(14) << "Max |ref|"
              << std::setw(14) << "Max rel err"
              << std::setw(12) << "Mismatches"
              << "\n";

    for (int size : sizes) {
      const BenchmarkResult custom = runCustomKernel(stage, size, size, size);
      const double cublas_gflops = runCublasBaseline(stage, size, size, size);
      const double pct_of_cublas = (custom.gflops / cublas_gflops) * 100.0;

      std::cout << std::left << std::setw(10) << size
                << std::setw(14) << std::fixed << std::setprecision(2) << custom.gflops
                << std::setw(14) << std::fixed << std::setprecision(2) << cublas_gflops
                << std::setw(12) << std::fixed << std::setprecision(2) << pct_of_cublas
                << std::setw(12) << std::fixed << std::setprecision(4) << custom.milliseconds
                << std::setw(14) << std::fixed << std::setprecision(2) << custom.requested_gbs
                << std::setw(14) << std::scientific << std::setprecision(3) << custom.max_abs_diff
                << std::setw(14) << std::scientific << std::setprecision(3) << custom.max_abs_ref
                << std::setw(14) << std::scientific << std::setprecision(3) << custom.max_rel_diff
                << std::setw(12) << custom.mismatch_count
                << "\n";

      if (custom.max_abs_diff == 0.0f && custom.max_abs_ref > 1.0f) {
        std::cerr << "Warning: exact-zero abs error at size " << size
                  << " with max|ref|=" << custom.max_abs_ref
                  << " is not expected for sequential FP32 vs cuBLAS.\n";
      }

      if (custom.max_rel_diff > stage.rel_tolerance) {
        std::cerr << "Correctness check failed for size " << size
                  << ": max rel diff " << custom.max_rel_diff
                  << " exceeded " << stage.rel_tolerance << ".\n";
        return 1;
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Benchmark error: " << ex.what() << "\n";
    return 1;
  }
}
