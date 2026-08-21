#pragma once

// Host-only. No CUDA includes, so tests/test_error_stats.cpp can build this
// with a plain host compiler on a machine with no GPU.

#include <cmath>
#include <cstddef>
#include <vector>

namespace bench {

constexpr float kRelDenomFloor = 1.0e-12f;

struct ErrorStats {
  float max_abs_diff = 0.0f;
  float max_abs_ref = 0.0f;
  float max_rel_diff = 0.0f;
  // Elements that differ from the reference at all, at any magnitude. Two FP32
  // kernels summing the same products in a different order differ here while
  // still being correct, so a large count is not a failure.
  std::size_t diff_elems = 0;
  // Elements that exceed the stage tolerance. This is the one that gates.
  std::size_t fail_elems = 0;
  std::size_t first_diff = static_cast<std::size_t>(-1);
};

inline ErrorStats computeErrorStats(const float* out,
                                    const float* ref,
                                    std::size_t count,
                                    float rel_tolerance) {
  ErrorStats stats;

  float max_diff = 0.0f;
  float max_ref = 0.0f;
  for (std::size_t i = 0; i < count; ++i) {
    const float diff = std::fabs(out[i] - ref[i]);
    const float abs_ref = std::fabs(ref[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
    if (abs_ref > max_ref) {
      max_ref = abs_ref;
    }
    if (diff > 0.0f) {
      ++stats.diff_elems;
      if (stats.first_diff == static_cast<std::size_t>(-1)) {
        stats.first_diff = i;
      }
    }
  }

  const float denom = max_ref > kRelDenomFloor ? max_ref : kRelDenomFloor;

  // Second pass: the per-element threshold is not known until max_ref is.
  const float threshold = rel_tolerance * denom;
  for (std::size_t i = 0; i < count; ++i) {
    if (std::fabs(out[i] - ref[i]) > threshold) {
      ++stats.fail_elems;
    }
  }

  stats.max_abs_diff = max_diff;
  stats.max_abs_ref = max_ref;
  stats.max_rel_diff = max_diff / denom;
  return stats;
}

inline ErrorStats computeErrorStats(const std::vector<float>& out,
                                    const std::vector<float>& ref,
                                    float rel_tolerance) {
  const std::size_t count = out.size() < ref.size() ? out.size() : ref.size();
  return computeErrorStats(out.data(), ref.data(), count, rel_tolerance);
}

}  // namespace bench
