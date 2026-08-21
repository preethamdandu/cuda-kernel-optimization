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

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    reset();
    ptr = other.ptr;
    count = other.count;
    other.ptr = nullptr;
    other.count = 0;
    return *this;
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
    cudaError_t start_status = cudaEventCreate(&start);
    cudaError_t stop_status = cudaEventCreate(&stop);
    if (start_status != cudaSuccess || stop_status != cudaSuccess) {
      if (start != nullptr) {
        cudaEventDestroy(start);
        start = nullptr;
      }
      if (stop != nullptr) {
        cudaEventDestroy(stop);
        stop = nullptr;
      }
      throw std::runtime_error(std::string("cudaEventCreate failed: ") +
                               cudaGetErrorString(start_status != cudaSuccess ? start_status : stop_status));
    }
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

struct ErrorStats {
  float max_abs_diff = 0.0f;
  float max_abs_ref = 0.0f;

  float max_rel_diff() const {
    if (max_abs_ref > 0.0f) {
      return max_abs_diff / max_abs_ref;
    }
    return max_abs_diff;
  }
};

struct BenchmarkResult {
  float milliseconds = 0.0f;
  double gflops = 0.0;
  double requested_gbs = 0.0;
  ErrorStats error;
};

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

void launchAndCheck(const StageDefinition& stage,
                    const float* a,
                    const float* b,
                    float* c,
                    int m,
                    int n,
                    int k,
                    const std::string& label) {
  stage.launcher(a, b, c, m, n, k);
  checkCuda(cudaGetLastError(), label);
}

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
    if (std::string(argv[i]) == "--stage") {
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
        throw std::runtime_error("--stage requires a name (naive or coalesced)");
      }
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

ErrorStats computeErrorStats(const std::vector<float>& out, const std::vector<float>& ref) {
  ErrorStats stats;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    stats.max_abs_diff = std::max(stats.max_abs_diff, std::fabs(out[i] - ref[i]));
    stats.max_abs_ref = std::max(stats.max_abs_ref, std::fabs(ref[i]));
  }
  return stats;
}

double requestedBytes(int m, int n, int k) {
  // No reuse: every K-iteration is a real A and B load, plus one C store.
  return (2.0 * static_cast<double>(m) * n * k + static_cast<double>(m) * n) * sizeof(float);
}

double gflopsFromMs(int m, int n, int k, float milliseconds) {
  const double flops = 2.0 * static_cast<double>(m) * n * k;
  return flops / (milliseconds * 1.0e6);
}

double requestedGbsFromMs(int m, int n, int k, float milliseconds) {
  return requestedBytes(m, n, k) / (milliseconds * 1.0e6);
}

struct DeviceInfo {
  std::string name;
  int sm_count = 0;
  int compute_major = 0;
  int compute_minor = 0;
  float sm_clock_mhz = 0.0f;
  float mem_clock_mhz = 0.0f;
  int memory_bus_width_bits = 0;
  double peak_dram_gbs = 0.0;
};

DeviceInfo queryDevice() {
  int device = 0;
  checkCuda(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp props{};
  checkCuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");

  DeviceInfo info;
  info.name = props.name;
  info.sm_count = props.multiProcessorCount;
  info.compute_major = props.major;
  info.compute_minor = props.minor;
  info.sm_clock_mhz = static_cast<float>(props.clockRate) / 1000.0f;
  info.mem_clock_mhz = static_cast<float>(props.memoryClockRate) / 1000.0f;
  info.memory_bus_width_bits = props.memoryBusWidth;
  // DDR: two transfers per clock. memoryClockRate is kHz; bus width is bits.
  info.peak_dram_gbs = 2.0 * static_cast<double>(props.memoryClockRate) * 1.0e3 *
                       (static_cast<double>(props.memoryBusWidth) / 8.0) / 1.0e9;
  return info;
}

void printDeviceHeader(const DeviceInfo& info, const StageDefinition& stage) {
  std::cout << "Device: " << info.name
            << "  SMs=" << info.sm_count
            << "  sm_" << info.compute_major << info.compute_minor
            << "  SM clock=" << std::fixed << std::setprecision(0) << info.sm_clock_mhz << " MHz"
            << "  Mem clock=" << info.mem_clock_mhz << " MHz"
            << "  Bus=" << info.memory_bus_width_bits << " bits\n";
  std::cout << "Peak DRAM bandwidth (theoretical): " << std::setprecision(1) << info.peak_dram_gbs
            << " GB/s\n";
  std::cout << "Req GB/s is requested traffic, not DRAM bandwidth; cache hits can make it exceed peak.\n";
  std::cout << "Baseline: " << (stage.use_tensor_core_baseline ? "cuBLAS FP16 (Tensor Core)" : "cuBLAS FP32")
            << "  rel_tolerance=" << std::scientific << std::setprecision(1) << stage.rel_tolerance << "\n";
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

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS uses column-major order. Swapping A/B computes the same row-major result.
  checkCublas(
      cublasSgemm(handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  n,
                  m,
                  k,
                  &alpha,
                  dev_b.ptr,
                  n,
                  dev_a.ptr,
                  k,
                  &beta,
                  dev_reference.ptr,
                  n),
      "cublasSgemm");

  launchAndCheck(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Kernel launch (correctness)");
  checkCuda(cudaDeviceSynchronize(), "Kernel sync");

  checkCuda(cudaMemcpy(host_c.data(), dev_c.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost),
            "Copy custom output to host");
  checkCuda(
      cudaMemcpy(host_reference.data(), dev_reference.ptr, c_count * sizeof(float), cudaMemcpyDeviceToHost),
      "Copy cuBLAS output to host");

  BenchmarkResult result;
  result.error = computeErrorStats(host_c, host_reference);

  CudaEventPair timer;
  for (int run = 0; run < kWarmupRuns; ++run) {
    launchAndCheck(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Kernel launch (warmup)");
  }
  checkCuda(cudaDeviceSynchronize(), "Warmup sync");

  checkCuda(cudaEventRecord(timer.start), "Record start event");
  for (int run = 0; run < kTimedRuns; ++run) {
    launchAndCheck(stage, dev_a.ptr, dev_b.ptr, dev_c.ptr, m, n, k, "Kernel launch (timed)");
  }
  checkCuda(cudaEventRecord(timer.stop), "Record stop event");
  checkCuda(cudaEventSynchronize(timer.stop), "Synchronize stop event");
  checkCuda(cudaEventElapsedTime(&result.milliseconds, timer.start, timer.stop), "Elapsed time");

  result.milliseconds /= static_cast<float>(kTimedRuns);
  result.gflops = gflopsFromMs(m, n, k, result.milliseconds);
  result.requested_gbs = requestedGbsFromMs(m, n, k, result.milliseconds);

  checkCublas(cublasDestroy(handle), "cublasDestroy");
  return result;
}

double runCublasBaseline(int m, int n, int k) {
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

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CudaEventPair timer;

  for (int run = 0; run < kWarmupRuns; ++run) {
    checkCublas(cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            dev_b.ptr,
                            n,
                            dev_a.ptr,
                            k,
                            &beta,
                            dev_c.ptr,
                            n),
                "cublasSgemm warmup");
  }
  checkCuda(cudaDeviceSynchronize(), "cuBLAS warmup sync");

  float milliseconds = 0.0f;
  checkCuda(cudaEventRecord(timer.start), "Record cuBLAS start event");
  for (int run = 0; run < kTimedRuns; ++run) {
    checkCublas(cublasSgemm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            dev_b.ptr,
                            n,
                            dev_a.ptr,
                            k,
                            &beta,
                            dev_c.ptr,
                            n),
                "cublasSgemm timed");
  }
  checkCuda(cudaEventRecord(timer.stop), "Record cuBLAS stop event");
  checkCuda(cudaEventSynchronize(timer.stop), "Synchronize cuBLAS stop event");
  checkCuda(cudaEventElapsedTime(&milliseconds, timer.start, timer.stop), "cuBLAS elapsed time");

  checkCublas(cublasDestroy(handle), "cublasDestroy");

  milliseconds /= static_cast<float>(kTimedRuns);
  return gflopsFromMs(m, n, k, milliseconds);
}

void printUsage() {
  std::cout << "Usage: ./bench [--stage naive|coalesced] [--sizes 1024 2048 4096]\n";
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

    const std::string stage_name = parseStageName(argc, argv);
    const StageDefinition stage = getStageOrThrow(stage_name);
    const std::vector<int> sizes = parseSizes(argc, argv);
    const DeviceInfo device = queryDevice();

    printDeviceHeader(device, stage);
    std::cout << "\nStage: " << stage.name << "\n";
    std::cout << stage.description << "\n\n";

    std::cout << std::left << std::setw(10) << "M=N=K"
              << std::setw(14) << "Kernel GF/s"
              << std::setw(14) << "cuBLAS GF/s"
              << std::setw(12) << "% cuBLAS"
              << std::setw(14) << "Req GB/s"
              << std::setw(12) << "Peak DRAM"
              << std::setw(12) << "Avg ms"
              << std::setw(14) << "Max abs err"
              << std::setw(14) << "Max rel err"
              << "\n";

    for (int size : sizes) {
      const BenchmarkResult custom = runCustomKernel(stage, size, size, size);
      const double cublas_gflops = runCublasBaseline(size, size, size);
      const double pct_of_cublas = (custom.gflops / cublas_gflops) * 100.0;
      const float max_rel = custom.error.max_rel_diff();

      std::cout << std::left << std::setw(10) << size
                << std::setw(14) << std::fixed << std::setprecision(2) << custom.gflops
                << std::setw(14) << std::fixed << std::setprecision(2) << cublas_gflops
                << std::setw(12) << std::fixed << std::setprecision(2) << pct_of_cublas
                << std::setw(14) << std::fixed << std::setprecision(1) << custom.requested_gbs
                << std::setw(12) << std::fixed << std::setprecision(1) << device.peak_dram_gbs
                << std::setw(12) << std::fixed << std::setprecision(4) << custom.milliseconds
                << std::setw(14) << std::scientific << std::setprecision(3) << custom.error.max_abs_diff
                << std::setw(14) << std::scientific << std::setprecision(3) << max_rel
                << "\n";

      if (max_rel > stage.rel_tolerance) {
        std::cerr << "Correctness check failed for size " << size
                  << ": max rel diff " << std::scientific << max_rel
                  << " exceeded tolerance " << stage.rel_tolerance
                  << " (max abs " << custom.error.max_abs_diff
                  << ", max |ref| " << custom.error.max_abs_ref << ").\n";
        return 1;
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Benchmark error: " << ex.what() << "\n";
    return 1;
  }
}
