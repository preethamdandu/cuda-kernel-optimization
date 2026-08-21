// Host-only unit test for the correctness checker. No GPU, no nvcc.
//
//   clang++ -O3 -std=c++17 tests/test_error_stats.cpp -o test_error_stats
//   ./test_error_stats
//
// This exists because the benchmark printed exact-zero error at one matrix
// size on each GPU and the first hypothesis was that the compare loop itself
// was being optimized away. These cases pin that down at the same element
// counts the benchmark uses (1024^2 through 4096^2).

#include "../benchmark/error_stats.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) {
    std::printf("  ok    %s\n", what.c_str());
  } else {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  }
}

std::vector<float> randomVector(std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(count);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

void testIdenticalBuffersReportZero() {
  std::printf("identical buffers report zero\n");
  const auto ref = randomVector(4096, 7);
  const auto out = ref;

  const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
  check(stats.max_abs_diff == 0.0f, "max_abs_diff is 0");
  check(stats.max_rel_diff == 0.0f, "max_rel_diff is 0");
  check(stats.diff_elems == 0, "diff_elems is 0");
  check(stats.fail_elems == 0, "fail_elems is 0");
  check(stats.max_abs_ref > 0.0f, "max_abs_ref still reports real data");
}

void testSinglePerturbedElementIsCaught() {
  std::printf("one perturbed element in a 4096^2 buffer is caught\n");
  // 16,777,216 elements: the exact size where the benchmark printed 0 on T4.
  const std::size_t count = 4096ull * 4096ull;
  std::vector<float> ref(count, 1.0f);
  std::vector<float> out(count, 1.0f);
  out[count - 1] = 1.5f;

  const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
  check(stats.diff_elems == 1, "diff_elems is exactly 1");
  check(stats.fail_elems == 1, "fail_elems is exactly 1");
  check(std::fabs(stats.max_abs_diff - 0.5f) < 1.0e-6f, "max_abs_diff is 0.5");
  check(stats.first_diff == count - 1, "first_diff points at the last element");
}

void testPerturbationAtEveryBenchmarkSize() {
  std::printf("a single perturbation is caught at every benchmark size\n");
  for (int n : {1024, 2048, 4096}) {
    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<float> ref(count, 2.0f);
    std::vector<float> out(count, 2.0f);
    out[count / 2] = 2.0f + 1.0e-3f;

    const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
    check(stats.diff_elems == 1, "N=" + std::to_string(n) + " diff_elems is 1");
    check(stats.fail_elems == 1, "N=" + std::to_string(n) + " fail_elems is 1");
  }
}

void testRelativeErrorUsesMaxRef() {
  std::printf("max_rel_diff divides by max|ref|\n");
  std::vector<float> ref = {100.0f, -200.0f, 50.0f};
  std::vector<float> out = {100.0f, -200.0f, 52.0f};

  const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
  check(std::fabs(stats.max_abs_ref - 200.0f) < 1.0e-6f, "max_abs_ref is 200");
  check(std::fabs(stats.max_abs_diff - 2.0f) < 1.0e-6f, "max_abs_diff is 2");
  check(std::fabs(stats.max_rel_diff - 0.01f) < 1.0e-6f, "max_rel_diff is 2/200");
}

void testDiffElemsAndFailElemsDiffer() {
  std::printf("reordered-summation noise counts as diff but not as failure\n");
  const std::size_t count = 1000;
  std::vector<float> ref(count, 100.0f);
  std::vector<float> out(count, 100.0f);
  // Every element off by 1e-5 relative: real bits differ, tolerance is fine.
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = 100.0f + 1.0e-3f;
  }

  const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
  check(stats.diff_elems == count, "every element differs");
  check(stats.fail_elems == 0, "no element exceeds the tolerance");
  check(stats.max_rel_diff < 1.0e-4f, "max_rel_diff is inside the gate");
}

void testToleranceBoundaryFails() {
  std::printf("an element past the tolerance is reported as a failure\n");
  std::vector<float> ref(16, 10.0f);
  std::vector<float> out(16, 10.0f);
  out[3] = 10.0f + 1.0e-2f;  // 1e-3 relative against max|ref| = 10

  const bench::ErrorStats stats = bench::computeErrorStats(out, ref, 1.0e-4f);
  check(stats.fail_elems == 1, "fail_elems is 1");
  check(stats.max_rel_diff > 1.0e-4f, "max_rel_diff trips the gate");
}

}  // namespace

int main() {
  testIdenticalBuffersReportZero();
  testSinglePerturbedElementIsCaught();
  testPerturbationAtEveryBenchmarkSize();
  testRelativeErrorUsesMaxRef();
  testDiffElemsAndFailElemsDiffer();
  testToleranceBoundaryFails();

  if (g_failures == 0) {
    std::printf("\nPASS: the compare loop detects differences at every benchmark size.\n");
    return 0;
  }
  std::printf("\nFAIL: %d check(s) failed.\n", g_failures);
  return 1;
}
