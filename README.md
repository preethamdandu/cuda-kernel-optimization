# CUDA Kernel Optimization Suite

SGEMM optimization ladder from an honest uncoalesced baseline through coalescing, tiling, register blocking, vectorized loads, and Tensor Cores — then the same ideas applied to fused attention.

**GPU:** Tesla T4, 40 SMs, sm_75. Peak DRAM 320.06 GB/s (from `cudaGetDeviceProperties`: mem clock 5001 MHz, 256-bit bus). Colab, 2026-08-21. Headline numbers are N=4096.

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Req GB/s | Max abs err | Max rel err | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Naive (uncoalesced) | FP32 | 61.73 | 1.48 | 247 | 1.53e-04† | 1.99e-06† | `threadIdx.x` → row |
| 2 | Coalesced | FP32 | 580.13 | 13.89 | 2321 | 1.53e-04† | 1.99e-06† | `threadIdx.x` → col |
| 3 | Shared-memory tiled | FP32 | 725.61 | 16.80 | 2903 | 1.53e-04† | 1.99e-06† | 32×32 tiles; only 1.25× Stage 2 |
| 4 | Register blocked | FP32 | 2202.43 | 51.52 | 8811 | 1.53e-04† | 1.99e-06† | 64×64 block, 4×4 per thread |
| 5 | Vectorized loads | FP32 | 2364.88 | 55.60 | 9461 | 1.53e-04† | 1.99e-06† | float4; +7% over Stage 4 |
| 6 | WMMA / Tensor Cores | FP16→FP32 | 4192 | 10.1‡ | 8386 | unverified | unverified | vs cuBLAS **FP16**, not FP32 |

† Error quoted from N=2048. At N=4096 the T4 run matched cuBLAS bit for bit — see [bitwise matches](#about-the-bitwise-matches), which is a real result, not a broken checker.  
‡ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 Tensor Cores (~41.5 TFLOPS at 4096), **not** vs FP32 cuBLAS (~4.2 TFLOPS). Do not put 10% and 55% in the same sentence.

Week 4 (same T4): fused attention wins only at seq=1024 (**1.47×** unfused); it **loses** at 256/512. Triton SGEMM is **80.8%** of `torch.matmul` at 4096 vs Stage 5 CUDA **55.6%** of cuBLAS. Details below.

Same kernels on an **RTX 4090** (RunPod, sm_89, separate session): [RTX 4090 section](#rtx-4090-runpod-2026-08-21). Do not mix T4 and 4090 rows.

Two conclusions in this README were wrong and got corrected by later measurements. Both corrections are in [notes/what_failed.md](notes/what_failed.md): the exact-zero error was **not** a compiler bug, and the fused attention regression was **not** an L2 story.

The only difference between these two kernels is which thread index drives the row. (For square M=N=K — the sizes in this table — the two grid expressions evaluate to identical `dim3` values; they would differ for non-square.)

Both kernels use `dim3 block(32, 32)`. Coalesced / naive at 4096 is **9.4×** (580.13 / 61.73). That is a bit above the 5–8× rule of thumb because Stage 1 is a *true* uncoalesced mapping, not the accidentally-coalesced kernel that used to sit in `01_naive.cu`.

## Stages 3–5 (Tesla T4, N=4096)

- **Stage 3 `tiled`:** 725.61 GFLOP/s, **16.8%** of cuBLAS. Only **1.25×** coalesced, not the 2–3× textbook jump. At N=1024 it *lost* (365 vs 623) — working set already in L2, `__syncthreads()` is extra cost. Logged in [notes/what_failed.md](notes/what_failed.md). Left as the teaching kernel; did not tune it to chase Stage 2.
- **Stage 4 `register`:** 2202 GFLOP/s, **51.5%** of cuBLAS. **3.0×** tiled, **3.8×** coalesced. This is the real reuse jump: each thread holds a 4×4 C patch, each shared value feeds 16 FMAs.
- **Stage 5 `vectorized`:** 2365 GFLOP/s, **55.6%** of cuBLAS. **+7%** over register (below the 10–30% band, still a real win).

Tiled, register, and vectorized report exactly the same 2048 error as Stages 1–2 (`1.53e-04` / `1.99e-06`, 4,000,760 differing elements, none past tolerance). Identical to the digit, because all five accumulate `k` in the same order — that is the proof the ladder never changed the math, only the memory access pattern.

## Stage 6 WMMA (Tesla T4)

`--stage wmma`: FP16 A/B, FP32 accumulate, 16×16×16 fragments, 64×64 block. Conversion is untimed. Baseline is **cuBLAS FP16 Tensor Cores**.

| N | Kernel GF/s | cuBLAS FP16 GF/s | % cuBLAS FP16 | Avg ms |
|---:|---:|---:|---:|---:|
| 1024 | 2383 | 20126 | 11.8 | 0.90 |
| 2048 | 2585 | 30760 | 8.4 | 6.65 |
| 4096 | 4192 | 41512 | 10.1 | 32.79 |

cuBLAS 41.5 TFLOPS at 4096 is ~64% of T4 FP16 Tensor Core peak (~65 TFLOPS). That baseline is healthy. The WMMA kernel is **4.2 TFLOPS, ~10% of that cuBLAS**, about **1.8×** Stage 5's FP32 2.37 TFLOPS. This is a teaching WMMA (no async copy, no double-buffer, 64×64 tile), not 60–80% of cuBLAS FP16. Do not inflate it.

On T4 this kernel matched `cublasGemmEx` bit for bit at all three sizes. On the 4090 the same kernel differs from cuBLAS in 1.0M / 4.2M / 16.7M elements at max rel 2.1e-06 / 3.2e-06 / 7.1e-06, well inside the 1e-2 gate. Both pass; see [bitwise matches](#about-the-bitwise-matches) for why the T4 zeros are not a checker bug.

Req GB/s in the harness still uses the no-reuse formula `(2MNK+MN)×4`. That number is meaningful vs peak for Stages 1–2. From Stage 3 on it is an upper bound, not DRAM traffic. Stage 6 uses 2-byte A/B in that formula.

### Per-size (same T4 session)

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Req GB/s | Peak DRAM | Max abs | Max rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1024 | 60.94 | 6271.39 | 0.97 | 243.87 | 320.06 | 6.87e-05 | 1.27e-06 |
| Naive | 2048 | 61.85 | 6661.33 | 0.93 | 247.46 | 320.06 | 1.53e-04 | 1.99e-06 |
| Naive | 4096 | 61.73 | 4170.76 | 1.48 | 246.95 | 320.06 | 0* | 0* |
| Coalesced | 1024 | 623.19 | 5262.86 | 11.84 | 2493.97 | 320.06 | 6.87e-05 | 1.27e-06 |
| Coalesced | 2048 | 580.51 | 6271.55 | 9.26 | 2322.62 | 320.06 | 1.53e-04 | 1.99e-06 |
| Coalesced | 4096 | 580.13 | 4175.41 | 13.89 | 2320.82 | 320.06 | 0* | 0* |
| Tiled | 1024 | 364.64 | 3049.07 | 11.96 | 1459.29 | 320.06 | 6.87e-05 | 1.27e-06 |
| Tiled | 2048 | 730.03 | 6396.33 | 11.41 | 2920.83 | 320.06 | 1.53e-04 | 1.99e-06 |
| Tiled | 4096 | 725.61 | 4320.24 | 16.80 | 2902.81 | 320.06 | 0* | 0* |
| Register | 1024 | 2622.10 | 5850.92 | 44.82 | 10493.50 | 320.06 | 6.87e-05 | 1.27e-06 |
| Register | 2048 | 2363.78 | 7182.81 | 32.91 | 9457.42 | 320.06 | 1.53e-04 | 1.99e-06 |
| Register | 4096 | 2202.43 | 4275.13 | 51.52 | 8810.81 | 320.06 | 0* | 0* |
| Vectorized | 1024 | 2914.15 | 6255.84 | 46.58 | 11662.30 | 320.06 | 6.87e-05 | 1.27e-06 |
| Vectorized | 2048 | 2432.24 | 6522.35 | 37.29 | 9731.35 | 320.06 | 1.53e-04 | 1.99e-06 |
| Vectorized | 4096 | 2364.88 | 4253.37 | 55.60 | 9460.67 | 320.06 | 0* | 0* |

\* Bitwise match against cuBLAS at this size. Explained below.

cuBLAS GF/s at 1024 moved 6271 → 5262 between the two process runs. That is boost/clock noise, not a kernel bug. Prefer the 4096 % of cuBLAS (same ~4170 GF/s both times).

## About the bitwise matches

Some cells report exactly `0` error. That is a real result, and working out why took two wrong turns (logged in [notes/what_failed.md](notes/what_failed.md)).

Every FP32 stage here accumulates `acc += a[k] * b[k]` in strictly increasing `k` with a single accumulator. Whenever cuBLAS's heuristic picks a kernel with that same order and no split-K, both sides round identically and the difference is exactly zero. Three observations rule out a broken comparison:

- The affected size **moves with the GPU**: N=4096 on the T4, N=1024 on the 4090. A compiler bug does not follow the hardware; a cuBLAS kernel-selection heuristic does.
- On the 4090, WMMA reports millions of differing elements from the same compare loop in the same binary.
- [tests/test_error_stats.cpp](tests/test_error_stats.cpp) compiles that loop with `clang++ -O3` on a machine with no GPU and catches a single perturbed element in a 4096² buffer.

It also explains why Stages 1–5 always report *identical* error to each other: same FMA order, same rounding, different memory access patterns. That is the correctness argument that the ladder never changed the math.

Two columns come out of this. **Diff elems** counts elements that differ at all, which reassociation makes large and harmless. **Fail elems** counts elements past the stage tolerance, and that is the one that gates.

## Why coalescing matters here

A CUDA warp is 32 consecutive `threadIdx.x` values (same `threadIdx.y` in a 32×32 block). Global memory coalesces when those 32 threads hit a contiguous 128-byte span (4 × 32-byte sectors).

- **Naive (uncoalesced):** `threadIdx.x` is the row, `threadIdx.y` is the column. Adjacent warp threads write `C[row+i][col]` — N floats apart — so the C store (and the A load) issue ~32 sectors per request instead of 4.
- **Coalesced:** `threadIdx.x` is the column, `threadIdx.y` is the row. Adjacent warp threads write `C[row][col+i]` — consecutive addresses — so the C store (and the B load) collapse into one 128-byte transaction.

Arithmetic is identical: one thread, one `C[i,j]`, K FMAs from global memory.

**Req GB/s** is requested traffic `(2·M·N·K + M·N) × 4` bytes over kernel time — not DRAM bandwidth. Both stages request the same byte count, so the stage-to-stage GB/s ratio equals the GFLOP/s ratio. The useful comparison is against the 320 GB/s DRAM peak:

- Naive sits at ~247 GB/s **below** peak. The kernel is latency-bound on uncoalesced 32-sector transactions, so wall-clock stretches and bytes/time look small.
- Coalesced sits at ~2320 GB/s, about **7×** peak. That is L1/L2 answering reuse the kernel does not express (each A row is reread across columns; each B column is reread across rows). Not a measurement bug.

## Profiling without Nsight counters

`ncu` was run on Colab T4 and on RunPod RTX 4090 **as root** with Nsight Compute 2025.1.1. Both printed:

`ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`

No `.ncu-rep` files were produced, so **sectors per request is not measured anywhere in this repo**. Root inside a container is not host privilege; this needs `NVreg_RestrictProfiling=0` on the host or a privileged VM.

What does work without those counters, and is wired into the harness:

- `-Xptxas -v` at compile time: registers per thread, static shared memory, and **spill store/load bytes** per kernel. This is what showed the fused attention v1 spill.
- `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at runtime: blocks per SM and theoretical occupancy. Both `bench` and `bench_attn` print an occupancy table before their results.
- `nsys` uses CUPTI tracing rather than the restricted counters and is attempted in [scripts/run_all.sh](scripts/run_all.sh).

If a privileged host ever becomes available, the number to collect is kernel-average sectors/request, expected around 16.5 for naive against 2.5 for coalesced. Until then the coalescing claim rests on wall-clock: **9.4×** on T4, **8.2×** on 4090.

## Project layout

```text
.
├── README.md
├── benchmark/
│   ├── bench.cu
│   └── bench_attn.cu
├── benchmark/
│   └── error_stats.hpp        # host-only checker, shared by both harnesses
├── notebooks/
│   ├── colab_week1.ipynb
│   ├── colab_week2.ipynb
│   ├── colab_week4.ipynb
│   └── colab_week5.ipynb      # v2 attention, occupancy, fair Triton
├── notes/
│   └── what_failed.md
├── profiles/
├── scripts/
│   └── run_all.sh             # one script for T4 and 4090 so they cannot drift
├── src/
│   ├── 01_naive.cu
│   ├── 02_coalesced.cu
│   ├── 03_tiled.cu
│   ├── 04_register_blocked.cu
│   ├── 05_vectorized.cu
│   ├── 06_wmma_tensorcore.cu
│   ├── 07_attention_unfused.cu
│   ├── 08_attention_fused.cu       # v1: thread per Q row
│   ├── 09_attention_fused_v2.cu    # v2: warp per Q row
│   ├── attn.cuh
│   ├── kernel_occupancy.cuh
│   ├── stage_registry.cu
│   └── stage_registry.cuh
├── tests/
│   ├── test_error_stats.cpp            # no GPU, no nvcc
│   └── test_fused_attention_math.cpp   # no GPU, no nvcc
└── triton/
    └── matmul.py
```

The two files in `tests/` build with plain `clang++` on a laptop:

```bash
c++ -O3 -std=c++17 tests/test_error_stats.cpp -o test_error_stats && ./test_error_stats
c++ -O3 -std=c++17 tests/test_fused_attention_math.cpp -o test_attn_math && ./test_attn_math
```

The first proves the correctness checker catches a single wrong element in a 4096² buffer at `-O3`. The second simulates the fused v2 recurrence — warp per Q row, per-tile max, single rescale — against an FP64 reference at nine sequence lengths including the tile and block tails.

## Week 4 — Fused vs unfused attention (Tesla T4)

Single-head SDPA, `head_dim=64`, batch=1. Unfused is three kernels with a `seq×seq` score matrix in global memory. Fused is one kernel with online softmax (FlashAttention recurrence) and 32×64 K/V tiles in shared memory. Sequence length capped at 1024.

FLOPs counted as `4·seq·seq·d` (QKᵀ + PV). Softmax is extra work, not in the GFLOP/s numerator. Host-side fused vs unfused compare (not the GEMM device checker).

| seq | Unfused ms | Fused ms | Speedup | Unfused GF/s | Fused GF/s | S MiB | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.555 | 0.829 | **0.67×** | 30.2 | 20.2 | 0.25 | 1.19e-07 | 9.59e-07 |
| 512 | 1.523 | 1.639 | **0.93×** | 44.1 | 41.0 | 1.00 | 1.42e-07 | 1.48e-06 |
| 1024 | 4.894 | 3.328 | **1.47×** | 54.9 | 80.7 | 4.00 | 1.15e-07 | 1.97e-06 |

Nothing past tolerance at any size, abs error ~1e-7. That is a real host copy, not the GEMM device checker.

**The explanation originally given here was wrong.** It read: T4 L2 is 4 MiB, the score matrix is 0.25 / 1.00 / 4.00 MiB, so fusion only pays once the working set exceeds L2. That effect exists, but it is not why this kernel was slow, and the correction is the interesting part. See the next section.

## The fused attention bug: 32 warps on a 128-SM GPU

The 4090 numbers are what exposed it. Fused was **0.38× / 0.37× / 0.45×** there — *worse* than on the T4. A cache-capacity story predicts the opposite, so the cache story was wrong.

`launchFusedAttention` was:

```cpp
fusedAttentionKernel<<<(seq + 31) / 32, 32>>>(...);  // one thread per Q row
```

At seq=1024 that is **32 blocks of one warp — 32 warps for the whole GPU**. On a 128-SM 4090, 96 SMs had no work at all and the rest ran a single warp with nothing to hide latency behind. The unfused path meanwhile launches 1024 blocks of 1024 threads. The benchmark was never measuring fusion against materialization; it was measuring a well-parallelized three-kernel path against a kernel using a fraction of a percent of the machine.

Each thread also carried `acc[64]` and `s[32]`, and the `s[]` loops are bounded by a runtime tile length, so that array cannot live in registers. `-Xptxas -v` reports the spill directly.

### v2: one warp per Q row

[`src/09_attention_fused_v2.cu`](src/09_attention_fused_v2.cu), same online-softmax math, different thread mapping:

| | v1 | v2 |
|---|---|---|
| Q rows per | thread | warp |
| Block | 32 threads | 128 threads (4 warps, 4 Q rows) |
| Blocks at seq=1024 | 32 | 256 |
| **Warps at seq=1024** | **32** | **1024** |
| Per-thread accumulator | `acc[64]` + `s[32]` | 2 floats |

head_dim 64 across 32 lanes gives each lane two accumulator floats, which is what removes the spill. Inside a tile, lane `j` owns K row `j` and computes the whole 64-long dot from shared memory, so `s_j` stays in a register and the dot needs no shuffle at all; only the two softmax reductions across the 32 K rows use `__shfl_xor_sync`. `tile_k` and `tile_v` are padded to 65 floats per row so that `(j*65 + d) % 32 == (j + d) % 32` and the lane-varying reads hit 32 distinct banks.

v1 stays in the binary. `./bench_attn` runs unfused, v1, and v2 side by side and prints the occupancy table for all three, so the before/after is measured rather than described.

```bash
bash scripts/run_all.sh          # builds both binaries with -Xptxas -v, runs everything
```

### CUDA vs Triton (SGEMM, same T4 session)

Triton tiled matmul (`triton/matmul.py`, 64×64×32, `tl.dot`) vs `torch.matmul` (cuBLAS). Same 2N³ GFLOP formula as the C++ harness.

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.675 | 0.523 | 3181 | 4105 | 77.5 | 7.63e-05 |
| 2048 | 4.743 | 3.888 | 3622 | 4419 | 82.0 | 1.60e-04 |
| 4096 | 39.93 | 32.25 | 3442 | 4262 | 80.8 | 0* |

\* Bitwise match at N=4096, same cause as the C++ harness. Quote the 2048 error.

torch.matmul 4262 GF/s matches the C++ cuBLAS row (~4253 on vectorized). Triton is **~81% of cuBLAS** at 4096 vs Stage 5 CUDA **55.6%**. About 80 lines of Python beat the five-stage FP32 ladder (3442 vs 2365 GF/s). That is the productivity-versus-control point: Triton won on this GPU, and the CUDA ladder is how you see why. `tl.dot` is genuinely FP32 here because the T4 has no TF32 path — do not mix this 81% with Stage 6's 10% of FP16 cuBLAS.

## RTX 4090 (RunPod, 2026-08-21)

Same binaries, compiled `-arch=sm_89`. **GPU:** NVIDIA GeForce RTX 4090, 128 SMs, sm_89. Peak DRAM 1008.10 GB/s (mem clock 10501 MHz, 384-bit bus). Headline numbers are N=4096. **Do not put these in the T4 table.**

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Req GB/s | Max abs err | Max rel err | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Naive (uncoalesced) | FP32 | 683.23 | 1.16 | 2733 | 3.36e-04† | 3.03e-06† | `threadIdx.x` → row |
| 2 | Coalesced | FP32 | 5602.65 | 9.54 | 22413 | 3.36e-04† | 3.03e-06† | **8.2×** naive |
| 3 | Shared-memory tiled | FP32 | 4517.12 | 7.69 | 18071 | 3.36e-04† | 3.03e-06† | **lost** to Stage 2 at every size |
| 4 | Register blocked | FP32 | 30162.64 | 51.33 | 120665 | 3.36e-04† | 3.03e-06† | **5.4×** coalesced |
| 5 | Vectorized loads | FP32 | 31573.96 | 54.55 | 126311 | 3.36e-04† | 3.03e-06† | **+4.7%** over Stage 4 |
| 6 | WMMA / Tensor Cores | FP16→FP32 | 46303 | 30.0‡ | 92628 | 7.86e-04† | 7.10e-06† | vs cuBLAS **FP16**, not FP32 |

† FP32/WMMA error quoted from N=4096. N=1024 FP32 matched cuBLAS bit for bit — inverted from the T4, where 4096 was the matching size, which is [the evidence that this tracks cuBLAS kernel selection](#about-the-bitwise-matches) rather than a compiler or checker bug. WMMA differs from cuBLAS at all three sizes here.  
‡ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 (~154 TFLOPS at 4096), **not** vs FP32 cuBLAS (~58.8 TFLOPS). Do not put 30% and 55% in the same sentence.

cuBLAS FP32 ~58.8 TFLOPS is a healthy 4090 baseline. Coalesced/naive is **8.2×** (5603 / 683). Tiling lost everywhere (4517 vs 5603 at 4096) — stronger miss than T4. Register blocking is still the real jump. Vectorized *lost* at N=1024 vs register (27486 vs 30002) and won by 4.7% at 4096. WMMA **46.3 TFLOPS, 30% of FP16 cuBLAS** vs T4's 10% — same teaching kernel, Ada Tensor Cores are the matching baseline.

### Per-size (same 4090 session)

`Diff elems` counts elements differing from cuBLAS at all; none of these rows had an element past tolerance.

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Avg ms | Max abs | Max rel | Diff elems |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1024 | 622.21 | 48321 | 1.29 | 3.451 | 0* | 0* | 0* |
| Naive | 2048 | 682.61 | 55116 | 1.24 | 25.17 | 1.49e-04 | 1.94e-06 | 4043834 |
| Naive | 4096 | 683.23 | 58826 | 1.16 | 201.16 | 3.36e-04 | 3.03e-06 | 16327639 |
| Coalesced | 1024 | 5583.70 | 48771 | 11.45 | 0.385 | 0* | 0* | 0* |
| Coalesced | 2048 | 5620.32 | 55098 | 10.20 | 3.057 | 1.49e-04 | 1.94e-06 | 4043834 |
| Coalesced | 4096 | 5602.65 | 58736 | 9.54 | 24.53 | 3.36e-04 | 3.03e-06 | 16327639 |
| Tiled | 1024 | 4803.41 | 48197 | 9.97 | 0.447 | 0* | 0* | 0* |
| Tiled | 2048 | 4836.05 | 55005 | 8.79 | 3.553 | 1.49e-04 | 1.94e-06 | 4043834 |
| Tiled | 4096 | 4517.12 | 58749 | 7.69 | 30.43 | 3.36e-04 | 3.03e-06 | 16327639 |
| Register | 1024 | 30002.17 | 48658 | 61.66 | 0.072 | 0* | 0* | 0* |
| Register | 2048 | 31353.42 | 55573 | 56.42 | 0.548 | 1.49e-04 | 1.94e-06 | 4043834 |
| Register | 4096 | 30162.64 | 58757 | 51.33 | 4.557 | 3.36e-04 | 3.03e-06 | 16327639 |
| Vectorized | 1024 | 27485.61 | 48100 | 57.14 | 0.078 | 0* | 0* | 0* |
| Vectorized | 2048 | 31993.17 | 57358 | 55.78 | 0.537 | 1.49e-04 | 1.94e-06 | 4043834 |
| Vectorized | 4096 | 31573.96 | 57885 | 54.55 | 4.353 | 3.36e-04 | 3.03e-06 | 16327639 |
| WMMA | 1024 | 41364 | 103819 | 39.84 | 0.052 | 1.14e-04 | 2.12e-06 | 1023911 |
| WMMA | 2048 | 45209 | 152105 | 29.72 | 0.380 | 2.44e-04 | 3.18e-06 | 4150276 |
| WMMA | 4096 | 46303 | 154185 | 30.03 | 2.968 | 7.86e-04 | 7.10e-06 | 16713183 |

\* Bitwise match against cuBLAS at N=1024 on this binary.

### Attention (4090) — v1, before the rewrite

| seq | Unfused ms | Fused v1 ms | Speedup | Unfused GF/s | Fused v1 GF/s | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.128 | 0.339 | **0.38×** | 130.9 | 49.6 | 1.19e-07 |
| 512 | 0.251 | 0.671 | **0.37×** | 267.1 | 100.1 | 1.42e-07 |
| 1024 | 0.607 | 1.340 | **0.45×** | 442.1 | 200.3 | 1.15e-07 |

Fused v1 lost at every sequence length, and lost *harder* here than on the T4. That is the observation that killed the L2 explanation and led to [the occupancy bug](#the-fused-attention-bug-32-warps-on-a-128-sm-gpu): on a 128-SM GPU, a kernel launching 32 warps has further to fall. v2 numbers go here after the next 4090 session; v1 rows stay for the comparison.

Host-side compare: nothing past tolerance, same ~1e-7 abs as T4.

### Triton (4090) — TF32, not FP32

This run was **not** a fair FP32 comparison, and is kept as the example of how the mistake looks:

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.037 | 0.046 | 58429 | 47211 | 123.8 | 3.95e-02 |
| 2048 | 0.245 | 0.311 | 70134 | 55225 | 127.0 | 5.66e-02 |
| 4096 | 1.902 | 2.347 | 72244 | 58568 | 123.4 | 9.12e-02 |

Do **not** quote "Triton beats cuBLAS." The tell is `max_abs` ~1e-2 against ~1e-4 for the CUDA FP32 stages: on Ada, `tl.dot` on float32 inputs lowers to **TF32 Tensor Cores** by default, which keeps 10 mantissa bits. `triton/matmul.py` now passes `input_precision="ieee"` and sets `torch.backends.cuda.matmul.allow_tf32 = False`, and gates on 1e-4. The TF32 row is still available behind `--tf32`, labelled. Both get re-run on the next 4090 session.

## Harness rules

- Correctness vs cuBLAS SGEMM (row-major via swapped A/B). Relative gate: `max_abs_diff / max(max_abs_ref, tiny) ≤ 1e-4` for FP32, 1e-2 for WMMA.
- **Diff elems** is any differing element; **Fail elems** is elements past that gate. Only the second one means anything.
- `cudaEvent` timing: 3 warmup + 10 timed; mean excludes warmup. cuBLAS uses the same harness.
- GFLOP/s = `2·M·N·K / (ms · 1e6)` with the product in `double`.
- Seed A with 42 and B with 43 so square A and B differ, but runs stay deterministic.
- `getRegisteredStages()` lives only in `stage_registry.cu`. Attention is a separate binary (`bench_attn`), not a GEMM `--stage`.
- Every table here is pasted from a run on the named GPU. T4 and 4090 never share a table.

## Run on Colab

Current notebook: [`notebooks/colab_week5.ipynb`](notebooks/colab_week5.ipynb) — attention v1 vs v2, occupancy, IEEE-FP32 Triton. **Runtime → Change runtime type → Hardware accelerator = T4 GPU**, then **Runtime → Run all**. A CPU runtime has no `nvidia-smi` and no `nvcc`; that is the `command not found` / `sm_` error. Changing the accelerator reconnects you to a new VM, so clone again after the switch.

Earlier notebooks: [week 1](notebooks/colab_week1.ipynb), [week 2](notebooks/colab_week2.ipynb), [week 4](notebooks/colab_week4.ipynb).

Then:

```bash
REPO_URL="https://github.com/preethamdandu/cuda-kernel-optimization.git"
DIR="cuda-kernel-optimization"
if [ -d "$DIR/.git" ]; then
  cd "$DIR" && git pull --ff-only
else
  git clone "$REPO_URL" "$DIR"
fi
```

```bash
cd cuda-kernel-optimization
bash scripts/run_all.sh          # add --quick to skip naive at 4096
```

If the GPU name in the harness header is not the one you expect, do not edit the results tables — send the output back instead.

## Build by hand (when `nvcc` is available)

`scripts/run_all.sh` is the same thing with the host tests, both binaries, Triton, and the profiling attempts wired together. The individual commands:

```bash
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')

# -Xptxas -v prints registers, shared memory and spill bytes per kernel.
nvcc -O3 -arch=$ARCH -lineinfo -Xptxas -v \
  benchmark/bench.cu \
  src/01_naive.cu src/02_coalesced.cu src/03_tiled.cu \
  src/04_register_blocked.cu src/05_vectorized.cu src/06_wmma_tensorcore.cu \
  src/stage_registry.cu \
  -lcublas -o bench

nvcc -O3 -arch=$ARCH -lineinfo -Xptxas -v \
  benchmark/bench_attn.cu \
  src/07_attention_unfused.cu src/08_attention_fused.cu src/09_attention_fused_v2.cu \
  -o bench_attn

./bench --stage naive --sizes 1024 2048 4096
./bench_attn --seqs 256 512 1024
python3 triton/matmul.py --sizes 1024 2048 4096          # IEEE FP32
python3 triton/matmul.py --sizes 1024 2048 4096 --tf32   # TF32, labelled
```

Both binaries print a launch-geometry and occupancy table before their results. Unknown `--stage` names print the registered list and exit.
