# CUDA Kernel Optimization Suite

SGEMM optimization ladder from an honest uncoalesced baseline through coalescing, tiling, register blocking, vectorized loads, and Tensor Cores — then the same ideas applied to fused attention.

**GPU:** Tesla T4, 40 SMs, sm_75. Peak DRAM 320.06 GB/s (from `cudaGetDeviceProperties`: mem clock 5001 MHz, 256-bit bus). Colab, 2026-08-21, single session via [`scripts/run_all.sh`](scripts/run_all.sh). Headline numbers are N=4096.

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Sectors/req† | Achieved occ.† | Max abs err‡ | Max rel err‡ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Naive (uncoalesced) | FP32 | 61.77 | 1.79 | **16.5** | 98.1% | 1.53e-04 | 1.99e-06 |
| 2 | Coalesced | FP32 | 530.21 | 15.57 | **2.5** | 98.5% | 1.53e-04 | 1.99e-06 |
| 3 | Shared-memory tiled | FP32 | 656.99 | 19.18 | 4.0 | 100.0% | 1.53e-04 | 1.99e-06 |
| 4 | Register blocked | FP32 | 1813.60 | 54.16 | 3.95 | 66.6% | 1.53e-04 | 1.99e-06 |
| 5 | Vectorized loads | FP32 | 2046.31 | 61.48 | 15.67 | 81.6% | 1.53e-04 | 1.99e-06 |
| 6 | WMMA / Tensor Cores | FP16→FP32 | 3489.60 | 9.58§ | — | — | 0 (bitwise) | 0 (bitwise) |

† Measured with Nsight Compute at N=1024. See [measured memory behaviour](#measured-memory-behaviour-nsight-compute-t4). Sectors/request is only meaningful against the access width: 4 is perfect for a `float` load, 16 for a `float4`, so Stage 5's 15.67 is optimal, not a regression.  
‡ Error quoted from N=2048. At N=4096 every FP32 stage matched cuBLAS bit for bit — see [bitwise matches](#about-the-bitwise-matches), which is a real result, not a broken checker. No stage had a single element past tolerance at any size.  
§ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 Tensor Cores (36.4 TFLOPS at 4096), **not** vs FP32 cuBLAS (3.3 TFLOPS). Do not put 9.6% and 61% in the same sentence.

The two headline results:

- **Coalescing is 8.6×** (530.21 / 61.77 at 4096), and the counters show why: **16.5 → 2.5 sectors per request**, which is 35.4 GB of L1 traffic collapsing to 5.4 GB for identical arithmetic.
- **Fused attention v2 is 8.5–9.8× faster than unfused** at every sequence length, after a rewrite that took the kernel from 32 warps to 1024. The [previous version lost](#the-fused-attention-bug-a-grid-too-small-to-fill-the-machine) at two of three lengths.

Same kernels on an **RTX 4090** (RunPod, sm_89, separate session): [RTX 4090 section](#rtx-4090-runpod-2026-08-21). Do not mix T4 and 4090 rows.

Three conclusions in this README were wrong and were corrected by later measurements: the exact-zero error was not a compiler bug, the fused attention regression was not an L2 story, and the fused kernel was not spilling registers. Each correction is logged in [notes/what_failed.md](notes/what_failed.md) with what the wrong reasoning was and what refuted it.

## Stages 1–2: the same arithmetic, 8.6× apart

The only difference is which thread index drives the row. Both use `dim3 block(32, 32)`; for the square M=N=K sizes here the two grid expressions produce identical `dim3` values. One thread computes one `C[i,j]` with K FMAs either way.

8.6× is above the usual 5–8× rule of thumb because Stage 1 is a *true* uncoalesced mapping, not the accidentally-coalesced kernel that used to sit in `01_naive.cu`.

## Stages 3–5 (Tesla T4, N=4096)

- **Stage 3 `tiled`:** 656.99 GFLOP/s, **19.2%** of cuBLAS. Only **1.24×** coalesced, not the 2–3× textbook jump. The counters show the reuse is real — global load requests drop **32×** — but 32×32 one-thread-per-output tiling does not raise arithmetic intensity enough to convert it. Left as the teaching kernel; not tuned to chase Stage 2. An earlier session had this stage *losing* to coalesced at N=1024; that did not reproduce, and [notes/what_failed.md](notes/what_failed.md) records why N=1024 measurements on Colab are not trustworthy.
- **Stage 4 `register`:** 1813.60 GFLOP/s, **54.2%** of cuBLAS. **2.8×** tiled. This is the real reuse jump: each thread holds a 4×4 C patch, so each shared value feeds 16 FMAs.
- **Stage 5 `vectorized`:** 2046.31 GFLOP/s, **61.5%** of cuBLAS, **+12.8%** over register. The counters say this is an instruction-issue win, not a traffic win: same bytes as Stage 4, a quarter of the requests.

Tiled, register, and vectorized report exactly the same 2048 error as Stages 1–2 (`1.53e-04` / `1.99e-06`, 4,000,760 differing elements, none past tolerance). Identical to the digit, because all five accumulate `k` in the same order — that is the proof the ladder never changed the math, only the memory access pattern.

## Stage 6 WMMA (Tesla T4)

`--stage wmma`: FP16 A/B, FP32 accumulate, 16×16×16 fragments, 64×64 block. Conversion is untimed. Baseline is **cuBLAS FP16 Tensor Cores**.

| N | Kernel GF/s | cuBLAS FP16 GF/s | % cuBLAS FP16 | Avg ms |
|---:|---:|---:|---:|---:|
| 1024 | 4546.95 | 34310.99 | 13.25 | 0.4723 |
| 2048 | 4369.71 | 40426.42 | 10.81 | 3.9316 |
| 4096 | 3489.60 | 36415.28 | 9.58 | 39.3853 |

cuBLAS at 40.4 TFLOPS (N=2048) is ~62% of T4 FP16 Tensor Core peak (~65 TFLOPS), so that baseline is healthy. The WMMA kernel is **3.5–4.5 TFLOPS, roughly 10% of it**, and about **1.7×** Stage 5's FP32 2.05 TFLOPS. This is a teaching WMMA — no async copy, no double buffering, 64×64 tile — not 60–80% of cuBLAS FP16. Do not inflate it.

On T4 this kernel matched `cublasGemmEx` bit for bit at all three sizes. On the 4090 the same kernel differs from cuBLAS in 1.0M / 4.2M / 16.7M elements at max rel 2.1e-06 / 3.2e-06 / 7.1e-06, well inside the 1e-2 gate. Both pass; see [bitwise matches](#about-the-bitwise-matches) for why the T4 zeros are not a checker bug.

Req GB/s in the harness still uses the no-reuse formula `(2MNK+MN)×4`. That number is meaningful vs peak for Stages 1–2. From Stage 3 on it is an upper bound, not DRAM traffic. Stage 6 uses 2-byte A/B in that formula.

### Per-size (same T4 session)

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Avg ms | Req GB/s | Max abs | Max rel | Fail elems |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1024 | 60.65 | 6293.62 | 0.96 | 35.4094 | 242.71 | 6.87e-05 | 1.27e-06 | 0 |
| Naive | 2048 | 61.29 | 6367.49 | 0.96 | 280.3112 | 245.21 | 1.53e-04 | 1.99e-06 | 0 |
| Naive | 4096 | 61.77 | 3453.24 | 1.79 | 2224.8618 | 247.13 | 0* | 0* | 0 |
| Coalesced | 1024 | 439.02 | 3617.94 | 12.13 | 4.8915 | 1756.94 | 6.87e-05 | 1.27e-06 | 0 |
| Coalesced | 2048 | 530.28 | 6535.73 | 8.11 | 32.3978 | 2121.63 | 1.53e-04 | 1.99e-06 | 0 |
| Coalesced | 4096 | 530.21 | 3405.89 | 15.57 | 259.2149 | 2121.11 | 0* | 0* | 0 |
| Tiled | 1024 | 764.43 | 6024.46 | 12.69 | 2.8093 | 3059.22 | 6.87e-05 | 1.27e-06 | 0 |
| Tiled | 2048 | 646.28 | 7017.24 | 9.21 | 26.5828 | 2585.74 | 1.53e-04 | 1.99e-06 | 0 |
| Tiled | 4096 | 656.99 | 3425.90 | 19.18 | 209.1946 | 2628.28 | 0* | 0* | 0 |
| Register | 1024 | 2989.25 | 6071.66 | 49.23 | 0.7184 | 11962.82 | 6.87e-05 | 1.27e-06 | 0 |
| Register | 2048 | 1887.46 | 6101.76 | 30.93 | 9.1021 | 7551.68 | 1.53e-04 | 1.99e-06 | 0 |
| Register | 4096 | 1813.60 | 3348.51 | 54.16 | 75.7825 | 7255.27 | 0* | 0* | 0 |
| Vectorized | 1024 | 2922.75 | 6150.01 | 47.52 | 0.7347 | 11696.69 | 6.87e-05 | 1.27e-06 | 0 |
| Vectorized | 2048 | 2073.36 | 6289.78 | 32.96 | 8.2860 | 8295.47 | 1.53e-04 | 1.99e-06 | 0 |
| Vectorized | 4096 | 2046.31 | 3328.22 | 61.48 | 67.1643 | 8186.24 | 0* | 0* | 0 |
| WMMA | 1024 | 4546.95 | 34310.99 | 13.25 | 0.4723 | 9102.78 | 0* | 0* | 0 |
| WMMA | 2048 | 4369.71 | 40426.42 | 10.81 | 3.9316 | 8743.69 | 0* | 0* | 0 |
| WMMA | 4096 | 3489.60 | 36415.28 | 9.58 | 39.3853 | 6980.90 | 0* | 0* | 0 |

\* Bitwise match against cuBLAS at this size. [Explained below.](#about-the-bitwise-matches)

Two things to read carefully in this table.

**cuBLAS FP32 drops to ~3400 GF/s at 4096 but sits near 6100–7000 at 1024 and 2048.** That is consistent across all five stages within the session, so it is not a per-kernel effect. The likely cause is thermal: the N=4096 cuBLAS run always follows the custom kernel at 4096, and for naive that means 13 launches of a 2.2-second kernel — about 29 seconds at full load — immediately beforehand. Treat the 4096 `% cuBLAS` column as flattered, and prefer the 2048 column when quoting a single ratio.

**Absolute numbers differ from earlier sessions on this repo.** A previous Colab T4 reported cuBLAS ~4170 GF/s at 4096 and Stage 5 at 2365. Different VM, different clocks. The *ratios* held across both sessions (fused v1 speedup 1.47× then, 1.48× now), which is why ratios are what get quoted and why this table was replaced wholesale rather than merged.

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

- **Naive (uncoalesced):** `threadIdx.x` is the row, `threadIdx.y` is the column. Adjacent warp threads write `C[row+i][col]` — N floats apart — so the C store and the A load issue 32 sectors per request instead of 4.
- **Coalesced:** `threadIdx.x` is the column, `threadIdx.y` is the row. Adjacent warp threads write `C[row][col+i]` — consecutive addresses — so the C store and the B load collapse into one 128-byte transaction.

Arithmetic is identical: one thread, one `C[i,j]`, K FMAs from global memory. Both predictions are [confirmed by the counters](#measured-memory-behaviour-nsight-compute-t4) at 16.5 and 2.5 sectors per request.

**Req GB/s** is requested traffic `(2·M·N·K + M·N) × 4` bytes over kernel time — not DRAM bandwidth. Both stages request the same byte count, so the stage-to-stage GB/s ratio equals the GFLOP/s ratio. The useful comparison is against the 320 GB/s DRAM peak:

- Naive sits at ~247 GB/s **below** peak. The kernel is latency-bound on uncoalesced 32-sector transactions, so wall-clock stretches and bytes/time look small.
- Coalesced sits at ~2120 GB/s, about **6.6×** peak. That is L1/L2 answering reuse the kernel does not express (each A row is reread across columns; each B column is reread across rows). Not a measurement bug.

## Measured memory behaviour (Nsight Compute, T4)

Nsight Compute **does** work on Colab T4. It was denied on RunPod (`ERR_NVGPUCTRPERM`), which is a container privilege policy rather than anything about the kernels — see [notes/what_failed.md](notes/what_failed.md). All figures below are `ncu --launch-count 1` at N=1024, collected by [`scripts/run_all.sh`](scripts/run_all.sh).

| Stage | Global load requests | Sectors | Sectors/req | L1 load traffic | Achieved occupancy |
|---|---:|---:|---:|---:|---:|
| Naive | 67,108,864 | 1,107,296,256 | **16.5** | 35.4 GB | 98.1% |
| Coalesced | 67,108,864 | 167,772,160 | **2.5** | 5.37 GB | 98.5% |
| Tiled | 2,097,152 | 8,388,608 | 4.0 | 268 MB | 100.0% |
| Register blocked | 1,048,576 | 4,145,178 | 3.95 | 133 MB | 66.6% |
| Vectorized | 262,144 | 4,107,656 | 15.67 | 131 MB | 81.6% |

### The counts reproduce the hand model exactly

Not approximately — to the digit. A naive warp issues, per k-iteration, one A load across 32 different rows (32 sectors) and one broadcast B load (1 sector), so 1024 iterations × 33 sectors × 32,768 warps = 1,107,296,256. Measured: 1,107,296,256. Coalesced turns that into a broadcast A (1 sector) and a contiguous B (4 sectors): 1024 × 5 × 32,768 = 167,772,160. Measured: 167,772,160.

The tiling stages check out the same way. A 32×32 tile means each block reads `(32×1024 + 1024×32) × 4` = 256 KB, times 1024 blocks = 256 MB against 268 MB measured. A 64×64 tile reads 512 KB per block across 256 blocks = 128 MB against 133 MB measured.

### Coalescing fixes sector efficiency; tiling fixes request count

These are different optimizations and the counters separate them cleanly.

Stage 2 leaves the request count untouched at 67.1M and cuts sectors per request by 6.6×. Everything after Stage 2 does the opposite: sectors per request stops improving, while requests collapse 67.1M → 2.1M → 1.05M → 262K, a **256× reduction**. Total L1 load traffic falls from 35.4 GB to 131 MB against a floor of 8 MB, which is what reading A and B exactly once would cost. That remaining 16× is the reuse gap between this ladder and cuBLAS.

### Sectors per request is not a lower-is-better metric

It only means anything relative to the access width. A perfectly coalesced `float` load is 4 sectors (32 lanes × 4 B = 128 B); a perfectly coalesced `float4` load is 16. So:

- Naive at 16.5 against an ideal of 4 is **4.1× wasted**.
- Coalesced at 2.5 beats 4 because its A load is a broadcast, costing 1 sector.
- Vectorized at 15.67 against an ideal of 16 is **optimal**, not a regression.

Stage 5 moves the same bytes as Stage 4 (131 vs 133 MB) using a quarter of the requests. That is precisely why `float4` bought only +12.8% here: there was no traffic left to save, only instruction issue. Anyone reading this column as lower-is-better would file Stage 5 as a bug.

### Occupancy runs inverse to performance

Naive and coalesced sit at 98%+ achieved occupancy and are the two slowest kernels in the ladder. Register blocking achieves 66.6% and is **30× faster than naive**. The two fastest stages have the two lowest occupancies.

Occupancy measures how many warp slots are filled, not how much work each warp retires. Register blocking deliberately trades occupancy for per-thread work: 72 registers and a 4×4 C patch per thread mean fewer resident warps, each doing 16 FMAs per shared-memory value. "Raise occupancy" is the reflex answer and this table is the counterexample.

The one place occupancy *was* the whole story is fused attention v1, below — and there the failure was not the occupancy percentage but the grid being too small to fill a single wave.

### Compile-time and runtime facts that need no counters

Still wired in, and the only profiling available on RunPod:

- `-Xptxas -v` reports registers, shared memory, and spill bytes per kernel at compile time.
- `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` give theoretical occupancy at runtime. Both `bench` and `bench_attn` print a table before their results.
- `nsys` uses CUPTI tracing rather than the restricted counters, and is attempted in `run_all.sh`. It is not installed in the Colab image.

Timings printed *under* `ncu` are inflated by serialized replay and must not be quoted: the same attention run reported 5.02 ms unfused under the profiler against 2.98 ms clean, and v1 flipped from winning to losing. Counters from profiled runs, wall-clock from unprofiled ones.

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

## Fused vs unfused attention (Tesla T4)

Single-head SDPA, `head_dim=64`, batch=1. Unfused is three kernels with a `seq×seq` score matrix in global memory. Fused is one kernel with the online-softmax (FlashAttention) recurrence and 32×64 K/V tiles in shared memory.

FLOPs counted as `4·seq·seq·d` (QKᵀ + PV). Softmax is extra work, not in the GFLOP/s numerator. The compare is host-side fused against unfused, not the GEMM device checker.

| seq | Unfused ms | v1 ms | **v2 ms** | v1 speedup | **v2 speedup** | v2/v1 | Unfused GF/s | **v2 GF/s** | Max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.3342 | 0.4982 | **0.0340** | 0.67× | **9.83×** | 14.7× | 50.2 | **493.7** | 1.12e-07 |
| 512 | 0.9206 | 0.9893 | **0.1088** | 0.93× | **8.46×** | 9.1× | 72.9 | **616.6** | 1.34e-07 |
| 1024 | 2.9752 | 2.0108 | **0.3508** | 1.48× | **8.48×** | 5.7× | 90.2 | **765.2** | 1.19e-07 |

Nothing past tolerance at any size, abs error ~1e-7.

**v2 wins at every sequence length, and wins hardest at the shortest one.** That last part matters: it is what disproves the explanation this section used to give.

## The fused attention bug: a grid too small to fill the machine

The original v1 write-up blamed cache capacity — T4 L2 is 4 MiB, the score matrix is 0.25 / 1.00 / 4.00 MiB, so fusion supposedly only pays once the working set exceeds L2, producing a crossover at seq=1024. Plausible, and wrong. Two things killed it:

1. On the 4090 (72 MiB L2), v1 was **worse**, not better: 0.38× / 0.37× / 0.45×. A cache-capacity story predicts a bigger cache helps the materializing path, not that it hurts the fused one by more.
2. v2 removes the crossover entirely and is *best* at seq=256, where the score matrix is 0.25 MiB and comfortably resident. If L2 capacity were the mechanism, seq=256 is exactly where fusion should have the least to offer.

`launchFusedAttention` was:

```cpp
fusedAttentionKernel<<<(seq + 31) / 32, 32>>>(...);  // one thread per Q row
```

At seq=1024 that is **32 blocks of one warp**. The unfused path meanwhile launches 1024 blocks of 1024 threads. The benchmark was never comparing fusion against materialization; it was comparing a well-parallelized three-kernel path against a kernel that could not fill the GPU.

### v2: one warp per Q row

[`src/09_attention_fused_v2.cu`](src/09_attention_fused_v2.cu) keeps the online-softmax math identical and changes only the thread mapping.

| | v1 | v2 | |
|---|---:|---:|---:|
| Q rows per | thread | warp | |
| Block | 32 threads | 128 threads (4 warps) | |
| Blocks at seq=1024 | 32 | 256 | |
| Warps at seq=1024 | 32 | 1024 | 32× |
| Registers / thread | 189 | 66 | |
| Shared memory / thread | 520 B | 142 B | |
| Theoretical occupancy | 9.4% | 37.5% | 4.0× |
| **Achieved occupancy** | **3.12%** | **33.19%** | **10.6×** |
| **Waves per SM** | **0.27** | **2.13** | **7.9×** |
| Measured speedup at seq=1024 | — | — | **5.7×** |

The last three rows are `ncu`; the rest are `-Xptxas -v` and the occupancy API.

**Two limiters were stacked, and the gap between theoretical and achieved occupancy separates them.** v2 realizes 88% of its theoretical occupancy (33.19 of 37.5). v1 realizes only 33% of its own (3.12 of 9.4), and `launch__waves_per_multiprocessor = 0.27` says why: with 32 blocks across 40 SMs, 8 SMs got no work at all and v1 never reached the 3 blocks/SM the hardware would have allowed. So the grid was too small *and* the per-thread footprint was too large — v1 asked for 16,640 bytes of shared memory to be used by 32 threads.

A 10.6× occupancy gain converting to a 5.7× speedup is the honest number. v2 buys its parallelism with shared-memory traffic and two warp-shuffle reductions per tile, so roughly half the theoretical gain shows up in wall-clock.

### What changed in the kernel

head_dim 64 across 32 lanes gives each lane two accumulator floats, replacing `acc[64]`. Inside a tile, lane `j` owns K row `j` and computes the whole 64-long dot from shared memory, so `s_j` stays in a register and the dot needs no shuffle; only the tile max and the sum reduce across lanes. `tile_k` and `tile_v` are padded to 65 floats per row so that `(j*65 + d) % 32 == (j + d) % 32`, putting the lane-varying reads on 32 distinct banks.

v1 stays in the binary. `./bench_attn` runs unfused, v1, and v2 side by side, so the before and after is measured rather than remembered.

### CUDA vs Triton (SGEMM, same T4 session)

Triton tiled matmul (`triton/matmul.py`, 64×64×32) against `torch.matmul`. Both sides pinned to IEEE FP32 (`input_precision="ieee"`, `allow_tf32=False`), same 2N³ GFLOP formula as the C++ harness.

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.0957 | 0.8332 | 1959.9 | 2577.4 | 76.0 | 7.63e-05 |
| 2048 | 7.7177 | 6.0385 | 2226.0 | 2845.0 | 78.2 | 1.60e-04 |
| 4096 | 52.0548 | 40.7782 | 2640.3 | 3370.4 | 78.3 | 0* |

\* Bitwise match at N=4096, same cause as the C++ harness. Quote the 2048 error.

Triton reaches **78.3% of cuBLAS** at 4096 against Stage 5's **61.5%**, and beats the five-stage CUDA ladder outright (2640 vs 2046 GF/s) in about 80 lines of Python. That is the productivity-versus-control point: Triton won on this GPU, and the CUDA ladder is how you see why.

Running with `--tf32` on the T4 produces byte-identical error (7.63e-05 / 1.60e-04) and the same ~78% ratio, which is the expected result — sm_75 has no TF32 path, so the flag is a no-op. On Ada it is [not a no-op](#triton-4090).

## RTX 4090 (RunPod, 2026-08-21)

Same `scripts/run_all.sh`, compiled `-arch=sm_89`. **GPU:** NVIDIA GeForce RTX 4090, 128 SMs, sm_89, 1536 threads/SM, 72 MiB L2. Peak DRAM 1008.10 GB/s (mem clock 10501 MHz, 384-bit bus). Headline numbers are N=4096. **Do not put these in the T4 table.**

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Req GB/s | Max abs err | Max rel err | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Naive (uncoalesced) | FP32 | 678.97 | 1.20 | 2716 | 3.36e-04† | 3.03e-06† | `threadIdx.x` → row |
| 2 | Coalesced | FP32 | 5578.42 | 9.87 | 22316 | 3.36e-04† | 3.03e-06† | **8.2×** naive |
| 3 | Shared-memory tiled | FP32 | 4492.91 | 7.91 | 17974 | 3.36e-04† | 3.03e-06† | **lost** to Stage 2 at every size |
| 4 | Register blocked | FP32 | 31660.35 | 56.00 | 126657 | 3.36e-04† | 3.03e-06† | **5.7×** coalesced |
| 5 | Vectorized loads | FP32 | 30414.17 | 54.20 | 121672 | 3.36e-04† | 3.03e-06† | **−3.9%** vs Stage 4 |
| 6 | WMMA / Tensor Cores | FP16→FP32 | 50150 | 30.3‡ | 100325 | 7.86e-04† | 7.10e-06† | vs cuBLAS **FP16**, not FP32 |

† Error quoted from N=4096. N=1024 FP32 matched cuBLAS bit for bit — inverted from the T4, where 4096 was the matching size, which is [the evidence that this tracks cuBLAS kernel selection](#about-the-bitwise-matches). WMMA differs from cuBLAS at all three sizes. Fail elems = 0 on every row.  
‡ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 (165 TFLOPS at 4096), **not** vs FP32 cuBLAS (56.4 TFLOPS). Do not put 30% and 56% in the same sentence.

cuBLAS FP32 at 56.4 TFLOPS is a healthy 4090 baseline. Coalesced/naive is **8.2×** (5578 / 679), same ratio as the previous 4090 session. Tiling lost everywhere (4493 vs 5578 at 4096) — a stronger miss than T4, and it is stable across both 4090 runs. Register blocking is still the real jump. Vectorized *lost* to register at 1024 and 4096 and won only at 2048; the earlier “+4.7% at 4096” did not hold, so quote Stage 4 as the FP32 ceiling of this ladder. WMMA **50.2 TFLOPS, 30% of FP16 cuBLAS** vs T4's 10% — same teaching kernel, Ada Tensor Cores are the matching baseline.

`ncu` printed `ERR_NVGPUCTRPERM` on every kernel. Sector counts stay a T4-only measurement. `nsys` is not installed in this image. `-Xptxas -v` and the occupancy API still work: fused v1 is 190 registers, **zero spill**, 16,640 B shared; v2 is 56 registers, zero spill, 18,176 B shared.

### Per-size (same 4090 session)

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Avg ms | Max abs | Max rel | Diff elems | Fail elems |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1024 | 618.25 | 48422.59 | 1.28 | 3.4735 | 0* | 0* | 0 | 0 |
| Naive | 2048 | 681.70 | 54156.63 | 1.26 | 25.2016 | 1.49e-04 | 1.94e-06 | 4043834 | 0 |
| Naive | 4096 | 678.97 | 56384.53 | 1.20 | 202.4222 | 3.36e-04 | 3.03e-06 | 16327639 | 0 |
| Coalesced | 1024 | 5552.43 | 46919.43 | 11.83 | 0.3868 | 0* | 0* | 0 | 0 |
| Coalesced | 2048 | 5590.35 | 54032.91 | 10.35 | 3.0731 | 1.49e-04 | 1.94e-06 | 4043834 | 0 |
| Coalesced | 4096 | 5578.42 | 56491.32 | 9.87 | 24.6376 | 3.36e-04 | 3.03e-06 | 16327639 | 0 |
| Tiled | 1024 | 4776.02 | 46811.43 | 10.20 | 0.4496 | 0* | 0* | 0 | 0 |
| Tiled | 2048 | 4805.44 | 53928.69 | 8.91 | 3.5751 | 1.49e-04 | 1.94e-06 | 4043834 | 0 |
| Tiled | 4096 | 4492.91 | 56787.70 | 7.91 | 30.5902 | 3.36e-04 | 3.03e-06 | 16327639 | 0 |
| Register | 1024 | 29952.63 | 47021.35 | 63.70 | 0.0717 | 0* | 0* | 0 | 0 |
| Register | 2048 | 31103.48 | 54102.60 | 57.49 | 0.5523 | 1.49e-04 | 1.94e-06 | 4043834 | 0 |
| Register | 4096 | 31660.35 | 56538.91 | 56.00 | 4.3410 | 3.36e-04 | 3.03e-06 | 16327639 | 0 |
| Vectorized | 1024 | 28728.11 | 45990.17 | 62.47 | 0.0748 | 0* | 0* | 0 | 0 |
| Vectorized | 2048 | 32716.88 | 53998.12 | 60.59 | 0.5251 | 1.49e-04 | 1.94e-06 | 4043834 | 0 |
| Vectorized | 4096 | 30414.17 | 56117.84 | 54.20 | 4.5189 | 3.36e-04 | 3.03e-06 | 16327639 | 0 |
| WMMA | 1024 | 45000.24 | 112750.11 | 39.91 | 0.0477 | 1.14e-04 | 2.12e-06 | 1023911 | 0 |
| WMMA | 2048 | 49057.08 | 165798.12 | 29.59 | 0.3502 | 2.44e-04 | 3.18e-06 | 4150276 | 0 |
| WMMA | 4096 | 50150.48 | 165374.23 | 30.33 | 2.7405 | 7.86e-04 | 7.10e-06 | 16713183 | 0 |

\* Bitwise match against cuBLAS at N=1024 on this binary.

### Attention (4090)

Occupancy API at seq=1024 (counters denied; these are theoretical):

| Kernel | Regs | Smem B | Thr/blk | Blocks | Warps | Occupancy |
|---|---:|---:|---:|---:|---:|---:|
| fused v1 | 190 | 16640 | 32 | 32 | **32** | 10.4% |
| fused v2 | 56 | 18176 | 128 | 256 | **1024** | 41.7% |

| seq | Unfused ms | v1 ms | **v2 ms** | v1 speedup | **v2 speedup** | v2/v1 | Unfused GF/s | **v2 GF/s** | Max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.1186 | 0.3123 | **0.0157** | 0.38× | **7.53×** | 19.8× | 141.5 | **1065** | 1.12e-07 |
| 512 | 0.2313 | 0.6173 | **0.0282** | 0.37× | **8.21×** | 21.9× | 290.1 | **2382** | 1.34e-07 |
| 1024 | 0.5575 | 1.2333 | **0.0634** | 0.45× | **8.80×** | 19.5× | 481.5 | **4236** | 1.19e-07 |

v1 still loses at every sequence length, and loses *harder* than on the T4. That is the observation that killed the L2 explanation: on a 128-SM GPU, 32 warps have further to fall. v2 wins **7.5–8.8×** over unfused at every length, about **20×** over v1. Fail elems = 0.

The occupancy API cannot report *achieved* occupancy here the way `ncu` did on T4 (3.12% vs 33.19%). Theoretical occupancy still shows the same 4× gap (10.4% vs 41.7%), and the grid is still 32 warps vs 1024.

### Triton (4090)

IEEE FP32 on both sides (`input_precision="ieee"`, `allow_tf32=False`):

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0500 | 0.0458 | 42913 | 46889 | 91.5 | 0* | 0* |
| 2048 | 0.3463 | 0.3127 | 49614 | 54933 | 90.3 | 1.60e-04 | 2.08e-06 |
| 4096 | 2.7704 | 2.4280 | 49611 | 56606 | **87.6** | 3.89e-04 | 3.24e-06 |

\* Bitwise match at N=1024, same pattern as the C++ harness on this GPU.

This is the fair row. Triton is **87.6% of cuBLAS FP32** at 4096 against Stage 4 CUDA **56.0%**, and max rel is `3.2e-6` — the same order as the CUDA ladder. torch.matmul 56606 GF/s matches the C++ cuBLAS row (56539 on register).

TF32 on both sides (`--tf32`), labelled, **not** comparable to the CUDA FP32 ladder:

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.0379 | 0.0321 | 56686 | 66803 | 84.9 | 3.90e-02 | 7.34e-04 |
| 2048 | 0.2512 | 0.2211 | 68400 | 77714 | 88.0 | 6.54e-02 | 8.50e-04 |
| 4096 | 1.9268 | 1.7073 | 71330 | 80499 | 88.6 | 1.03e-01 | 8.54e-04 |

An earlier 4090 run reported Triton at **123% of torch** with `max_abs ~ 9e-2`. That was TF32 Triton against FP32 torch (torch sat at ~58 TFLOPS, matching the C++ FP32 cuBLAS row). With both sides on TF32, Triton is 88.6% of a 80.5 TFLOPS baseline and the error stays at `1e-1`. Do not quote 123%. The tell that a run is TF32 is the error, not the speedup.

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
