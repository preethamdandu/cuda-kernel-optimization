// Host simulation of the fused attention v2 recurrence. No GPU, no nvcc.
//
//   clang++ -O3 -std=c++17 tests/test_fused_attention_math.cpp -o test_attn_math
//   ./test_attn_math
//
// src/09_attention_fused_v2.cu cannot be compiled without nvcc, so this mirrors
// its exact loop structure -- warp per Q row, lane j owning K row j, per-tile
// max followed by one rescale -- and checks it against a plain double-precision
// softmax(QK^T/sqrt(d))V. The tail cases (seq not a multiple of the 32-wide K
// tile or the 4-row block) are the ones worth pinning down.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kHead = 64;
constexpr int kBc = 32;
constexpr int kWarpsPerBlock = 4;

int g_failures = 0;

void check(bool condition, const std::string& what) {
  if (condition) {
    std::printf("  ok    %s\n", what.c_str());
  } else {
    std::printf("  FAIL  %s\n", what.c_str());
    ++g_failures;
  }
}

std::vector<float> randomMatrix(int rows, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> values(static_cast<std::size_t>(rows) * kHead);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

// Straightforward reference: materialize the row, softmax it, multiply by V.
std::vector<double> referenceAttention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       int seq) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHead));
  std::vector<double> out(static_cast<std::size_t>(seq) * kHead, 0.0);

  for (int i = 0; i < seq; ++i) {
    std::vector<double> s(seq);
    double row_max = -1.0e300;
    for (int j = 0; j < seq; ++j) {
      double dot = 0.0;
      for (int d = 0; d < kHead; ++d) {
        dot += static_cast<double>(q[i * kHead + d]) * static_cast<double>(k[j * kHead + d]);
      }
      s[j] = dot * scale;
      row_max = std::fmax(row_max, s[j]);
    }
    double sum = 0.0;
    for (int j = 0; j < seq; ++j) {
      s[j] = std::exp(s[j] - row_max);
      sum += s[j];
    }
    for (int j = 0; j < seq; ++j) {
      const double p = s[j] / sum;
      for (int d = 0; d < kHead; ++d) {
        out[static_cast<std::size_t>(i) * kHead + d] += p * static_cast<double>(v[j * kHead + d]);
      }
    }
  }
  return out;
}

// Mirrors fusedAttentionV2Kernel: blocks of kWarpsPerBlock warps, one warp per
// Q row, 32 lanes each owning one K row of the tile and two output columns.
std::vector<float> simulateFusedV2(const std::vector<float>& q,
                                   const std::vector<float>& k,
                                   const std::vector<float>& v,
                                   int seq) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHead));
  std::vector<float> out(static_cast<std::size_t>(seq) * kHead, 0.0f);
  const int blocks = (seq + kWarpsPerBlock - 1) / kWarpsPerBlock;

  for (int block = 0; block < blocks; ++block) {
    // Shared, block-wide.
    std::vector<float> tile_q(kWarpsPerBlock * kHead, 0.0f);
    for (int r = 0; r < kWarpsPerBlock; ++r) {
      const int g_row = block * kWarpsPerBlock + r;
      for (int c = 0; c < kHead; ++c) {
        tile_q[r * kHead + c] = (g_row < seq) ? q[g_row * kHead + c] : 0.0f;
      }
    }

    for (int warp = 0; warp < kWarpsPerBlock; ++warp) {
      const int q_row = block * kWarpsPerBlock + warp;
      const bool active = q_row < seq;

      float m_i = -INFINITY;
      float l_i = 0.0f;

      for (int kv = 0; kv < seq; kv += kBc) {
        const int remaining = seq - kv;
        const int tile = remaining < kBc ? remaining : kBc;

        // Lane j owns K row j.
        std::vector<float> s(32, -INFINITY);
        for (int lane = 0; lane < tile; ++lane) {
          float dot = 0.0f;
          for (int d = 0; d < kHead; ++d) {
            dot += tile_q[warp * kHead + d] * k[(kv + lane) * kHead + d];
          }
          s[lane] = scale * dot;
        }

        float tile_max = -INFINITY;
        for (int lane = 0; lane < 32; ++lane) {
          tile_max = std::fmax(tile_max, s[lane]);
        }

        const float m_new = std::fmax(m_i, tile_max);
        const float alpha = std::exp(m_i - m_new);

        std::vector<float> p(32, 0.0f);
        float l_add = 0.0f;
        for (int lane = 0; lane < 32; ++lane) {
          p[lane] = (lane < tile) ? std::exp(s[lane] - m_new) : 0.0f;
          l_add += p[lane];
        }

        // Each lane owns output columns `lane` and `lane + 32`. On the first
        // tile alpha is exp(-inf - m_new) = 0, so the accumulator starts clean.
        if (active) {
          const std::size_t base = static_cast<std::size_t>(q_row) * kHead;
          for (int lane = 0; lane < 32; ++lane) {
            float pv0 = 0.0f;
            float pv1 = 0.0f;
            for (int j = 0; j < tile; ++j) {
              pv0 += p[j] * v[(kv + j) * kHead + lane];
              pv1 += p[j] * v[(kv + j) * kHead + lane + 32];
            }
            out[base + lane] = out[base + lane] * alpha + pv0;
            out[base + lane + 32] = out[base + lane + 32] * alpha + pv1;
          }
        }

        l_i = l_i * alpha + l_add;
        m_i = m_new;
      }

      if (active) {
        const float inv = 1.0f / l_i;
        for (int d = 0; d < kHead; ++d) {
          out[static_cast<std::size_t>(q_row) * kHead + d] *= inv;
        }
      }
    }
  }
  return out;
}

void testAgainstReference(int seq) {
  const auto q = randomMatrix(seq, 42);
  const auto k = randomMatrix(seq, 43);
  const auto v = randomMatrix(seq, 44);

  const std::vector<double> ref = referenceAttention(q, k, v, seq);
  const std::vector<float> sim = simulateFusedV2(q, k, v, seq);

  double max_abs = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(sim[i]) - ref[i]));
  }
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "seq=%4d  max abs vs FP64 reference = %.3e", seq, max_abs);
  check(max_abs < 1.0e-5, buffer);
}

void testOnlineSoftmaxSurvivesLargeLogits() {
  // A late tile with much larger scores forces the alpha rescale path. Without
  // it the running sum from earlier tiles would dominate the result.
  const int seq = 64;
  auto q = randomMatrix(seq, 7);
  auto k = randomMatrix(seq, 8);
  const auto v = randomMatrix(seq, 9);
  for (int d = 0; d < kHead; ++d) {
    k[(seq - 1) * kHead + d] = 20.0f;
    q[0 * kHead + d] = 20.0f;
  }

  const std::vector<double> ref = referenceAttention(q, k, v, seq);
  const std::vector<float> sim = simulateFusedV2(q, k, v, seq);

  double max_abs = 0.0;
  bool finite = true;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    finite = finite && std::isfinite(sim[i]);
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(sim[i]) - ref[i]));
  }
  check(finite, "no NaN or inf with a 20x logit spike in the last tile");
  char buffer[128];
  std::snprintf(buffer, sizeof(buffer), "rescale path max abs = %.3e", max_abs);
  check(max_abs < 1.0e-5, buffer);
}

}  // namespace

int main() {
  std::printf("fused v2 recurrence vs FP64 reference\n");
  // 32 and 64 are exact tile multiples; 1, 3, 33, 100 and 129 exercise the
  // K-tile tail and the 4-rows-per-block tail.
  for (int seq : {1, 3, 32, 33, 64, 100, 128, 129, 256}) {
    testAgainstReference(seq);
  }

  std::printf("online softmax rescaling\n");
  testOnlineSoftmaxSurvivesLargeLogits();

  if (g_failures == 0) {
    std::printf("\nPASS: the v2 tiling and online softmax match the reference.\n");
    return 0;
  }
  std::printf("\nFAIL: %d check(s) failed.\n", g_failures);
  return 1;
}
