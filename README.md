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

† Error quoted from N=2048 FP32 stages. N=4096 FP32 error is not quoted.  
‡ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 Tensor Cores (~41.5 TFLOPS at 4096), **not** vs FP32 cuBLAS (~4.2 TFLOPS). Do not put 10% and 55% in the same sentence.

Week 4 (same T4): fused attention wins only at seq=1024 (**1.47×** unfused); it **loses** at 256/512. Triton SGEMM is **80.8%** of `torch.matmul` at 4096 vs Stage 5 CUDA **55.6%** of cuBLAS. Details below.

Same kernels on an **RTX 4090** (RunPod, sm_89, separate session): [RTX 4090 section](#rtx-4090-runpod-2026-08-21). Do not mix T4 and 4090 rows.

The only difference between these two kernels is which thread index drives the row. (For square M=N=K — the sizes in this table — the two grid expressions evaluate to identical `dim3` values; they would differ for non-square.)

Both kernels use `dim3 block(32, 32)`. Coalesced / naive at 4096 is **9.4×** (580.13 / 61.73). That is a bit above the 5–8× rule of thumb because Stage 1 is a *true* uncoalesced mapping, not the accidentally-coalesced kernel that used to sit in `01_naive.cu`.

## Stages 3–5 (Tesla T4, N=4096)

- **Stage 3 `tiled`:** 725.61 GFLOP/s, **16.8%** of cuBLAS. Only **1.25×** coalesced, not the 2–3× textbook jump. At N=1024 it *lost* (365 vs 623) — working set already in L2, `__syncthreads()` is extra cost. Logged in [notes/what_failed.md](notes/what_failed.md). Left as the teaching kernel; did not tune it to chase Stage 2.
- **Stage 4 `register`:** 2202 GFLOP/s, **51.5%** of cuBLAS. **3.0×** tiled, **3.8×** coalesced. This is the real reuse jump: each thread holds a 4×4 C patch, each shared value feeds 16 FMAs.
- **Stage 5 `vectorized`:** 2365 GFLOP/s, **55.6%** of cuBLAS. **+7%** over register (below the 10–30% band, still a real win).

Same 2048 error on tiled, register, and vectorized as Stages 1–2 (`1.53e-04` / `1.99e-06`, 4,000,760 mismatches). That is the correctness proof that 3–5 compute the same math.

## Stage 6 WMMA (Tesla T4)

`--stage wmma`: FP16 A/B, FP32 accumulate, 16×16×16 fragments, 64×64 block. Conversion is untimed. Baseline is **cuBLAS FP16 Tensor Cores**.

| N | Kernel GF/s | cuBLAS FP16 GF/s | % cuBLAS FP16 | Avg ms |
|---:|---:|---:|---:|---:|
| 1024 | 2383 | 20126 | 11.8 | 0.90 |
| 2048 | 2585 | 30760 | 8.4 | 6.65 |
| 4096 | 4192 | 41512 | 10.1 | 32.79 |

cuBLAS 41.5 TFLOPS at 4096 is ~64% of T4 FP16 Tensor Core peak (~65 TFLOPS). That baseline is healthy. The WMMA kernel is **4.2 TFLOPS, ~10% of that cuBLAS**, about **1.8×** Stage 5's FP32 2.37 TFLOPS. This is a teaching WMMA (no async copy, no double-buffer, 64×64 tile), not 60–80% of cuBLAS FP16. Do not inflate it.

Correctness printed 0 mismatches at 1024, 2048, and 4096 with `max|ref|` 54 / 77 / 111. FP32 stages at 1024/2048 had ~1e6 / ~4e6 mismatches. Either WMMA matches `cublasGemmEx` bitwise (possible if both hit the same MMA) or the compare loop is still lying. Error column on T4 is **unverified**. The same WMMA kernel on the 4090 reports ~1e6 / ~4e6 / ~17e6 mismatches — the checker can see diffs; the T4 zeros are still not a pass. See [notes/what_failed.md](notes/what_failed.md).

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

\* Do not quote. GFLOP/s at 4096 recomputes from wall-clock (`2 N³ / (ms · 1e6)`); the error column does not.

cuBLAS GF/s at 1024 moved 6271 → 5262 between the two process runs. That is boost/clock noise, not a kernel bug. Prefer the 4096 % of cuBLAS (same ~4170 GF/s both times).

## Why coalescing matters here

A CUDA warp is 32 consecutive `threadIdx.x` values (same `threadIdx.y` in a 32×32 block). Global memory coalesces when those 32 threads hit a contiguous 128-byte span (4 × 32-byte sectors).

- **Naive (uncoalesced):** `threadIdx.x` is the row, `threadIdx.y` is the column. Adjacent warp threads write `C[row+i][col]` — N floats apart — so the C store (and the A load) issue ~32 sectors per request instead of 4.
- **Coalesced:** `threadIdx.x` is the column, `threadIdx.y` is the row. Adjacent warp threads write `C[row][col+i]` — consecutive addresses — so the C store (and the B load) collapse into one 128-byte transaction.

Arithmetic is identical: one thread, one `C[i,j]`, K FMAs from global memory.

**Req GB/s** is requested traffic `(2·M·N·K + M·N) × 4` bytes over kernel time — not DRAM bandwidth. Both stages request the same byte count, so the stage-to-stage GB/s ratio equals the GFLOP/s ratio. The useful comparison is against the 320 GB/s DRAM peak:

- Naive sits at ~247 GB/s **below** peak. The kernel is latency-bound on uncoalesced 32-sector transactions, so wall-clock stretches and bytes/time look small.
- Coalesced sits at ~2320 GB/s, about **7×** peak. That is L1/L2 answering reuse the kernel does not express (each A row is reread across columns; each B column is reread across rows). Not a measurement bug.

## Week 3 ncu (measured: counters denied)

Ran `ncu` on Colab T4 (expected deny) and on RunPod RTX 4090 as root with Nsight Compute 2025.1.1. Both printed:

`ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`

No `.ncu-rep` files. Sectors/request is **not measured**. Expected contrast if a privileged host ever allows counters: ~16.5 vs ~2.5 kernel-average; 32 vs 4 on the A-load and C-store. Logged in [notes/what_failed.md](notes/what_failed.md). This is a platform wall, not a kernel bug.

## Project layout

```text
.
├── README.md
├── benchmark/
│   ├── bench.cu
│   └── bench_attn.cu
├── notebooks/
│   ├── colab_week1.ipynb
│   ├── colab_week2.ipynb
│   └── colab_week4.ipynb
├── notes/
│   └── what_failed.md
├── profiles/
├── src/
│   ├── 01_naive.cu
│   ├── 02_coalesced.cu
│   ├── 03_tiled.cu
│   ├── 04_register_blocked.cu
│   ├── 05_vectorized.cu
│   ├── 06_wmma_tensorcore.cu
│   ├── 07_attention_unfused.cu
│   ├── 08_attention_fused.cu
│   ├── attn.cuh
│   ├── stage_registry.cu
│   └── stage_registry.cuh
└── triton/
    └── matmul.py
```

## Week 4 — Fused vs unfused attention (Tesla T4)

Single-head SDPA, `head_dim=64`, batch=1. Unfused is three kernels with a `seq×seq` score matrix in global memory. Fused is one kernel with online softmax (FlashAttention recurrence) and 32×64 K/V tiles in shared memory. Sequence length capped at 1024.

FLOPs counted as `4·seq·seq·d` (QKᵀ + PV). Softmax is extra work, not in the GFLOP/s numerator. Host-side fused vs unfused compare (not the GEMM device checker).

| seq | Unfused ms | Fused ms | Speedup | Unfused GF/s | Fused GF/s | S MiB | Max abs | Max rel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.555 | 0.829 | **0.67×** | 30.2 | 20.2 | 0.25 | 1.19e-07 | 9.59e-07 |
| 512 | 1.523 | 1.639 | **0.93×** | 44.1 | 41.0 | 1.00 | 1.42e-07 | 1.48e-06 |
| 1024 | 4.894 | 3.328 | **1.47×** | 54.9 | 80.7 | 4.00 | 1.15e-07 | 1.97e-06 |

Fusion **lost** at 256 and 512. It only wins at 1024. T4 L2 is 4 MiB; the unfused score matrix is 0.25 / 1.00 / 4.00 MiB. Below L2 the extra `__syncthreads()` and online-softmax state are pure overhead. At 1024 the `seq×seq` working set equals L2, so not writing it starts to pay. Same lesson as Stage 3 tiling losing at N=1024. Logged in [notes/what_failed.md](notes/what_failed.md).

Zero mismatches at all three sizes, abs error ~1e-7. That is a real host copy, not the N=4096 GEMM checker. Do not call this FlashAttention: 80 GFLOP/s is a teaching kernel (one thread per Q row). PyTorch SDPA was not run in this session.

```bash
cd cuda-kernel-optimization
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
nvcc -O3 -arch=$ARCH -lineinfo \
  benchmark/bench_attn.cu src/07_attention_unfused.cu src/08_attention_fused.cu \
  -o bench_attn
./bench_attn --seqs 256 512 1024
python3 triton/matmul.py --sizes 1024 2048 4096
```

### CUDA vs Triton (SGEMM, same T4 session)

Triton tiled matmul (`triton/matmul.py`, 64×64×32, `tl.dot`) vs `torch.matmul` (cuBLAS). Same 2N³ GFLOP formula as the C++ harness.

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.675 | 0.523 | 3181 | 4105 | 77.5 | 7.63e-05 |
| 2048 | 4.743 | 3.888 | 3622 | 4419 | 82.0 | 1.60e-04 |
| 4096 | 39.93 | 32.25 | 3442 | 4262 | 80.8 | 0* |

\* Same exact-zero pattern as the C++ checker at N=4096. Quote 2048 error. torch.matmul 4262 GF/s matches the C++ cuBLAS row (~4253 on vectorized). Triton is **~81% of cuBLAS** at 4096 vs Stage 5 CUDA **55.6%**. ~80 lines of Python beat the five-stage FP32 ladder (3442 vs 2365 GF/s). That is the productivity-vs-control point: Triton won on this GPU; the CUDA ladder is how you see *why*. `tl.dot` here is FP32, not Tensor Cores — do not mix this 81% with Stage 6's 10% of FP16 cuBLAS.

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

† FP32/WMMA error quoted from N=4096. N=1024 FP32 printed exact-zero (inverted from T4, where 4096 was the zero). WMMA has real mismatches at 1024/2048/4096.  
‡ Stage 6 `% cuBLAS` is vs `cublasGemmEx` FP16 (~154 TFLOPS at 4096), **not** vs FP32 cuBLAS (~58.8 TFLOPS). Do not put 30% and 55% in the same sentence.

cuBLAS FP32 ~58.8 TFLOPS is a healthy 4090 baseline. Coalesced/naive is **8.2×** (5603 / 683). Tiling lost everywhere (4517 vs 5603 at 4096) — stronger miss than T4. Register blocking is still the real jump. Vectorized *lost* at N=1024 vs register (27486 vs 30002) and won by 4.7% at 4096. WMMA **46.3 TFLOPS, 30% of FP16 cuBLAS** vs T4's 10% — same teaching kernel, Ada Tensor Cores are the matching baseline.

### Per-size (same 4090 session)

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Avg ms | Max abs | Max rel | Mismatches |
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

\* Do not quote N=1024 FP32 error on this 4090 binary.

### Attention (4090)

Fused **lost at every seq**. 4090 L2 is 72 MiB; the 4 MiB score matrix never spills. The T4 1.47× win at 1024 does not carry over.

| seq | Unfused ms | Fused ms | Speedup | Unfused GF/s | Fused GF/s | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.128 | 0.339 | **0.38×** | 130.9 | 49.6 | 1.19e-07 |
| 512 | 0.251 | 0.671 | **0.37×** | 267.1 | 100.1 | 1.42e-07 |
| 1024 | 0.607 | 1.340 | **0.45×** | 442.1 | 200.3 | 1.15e-07 |

Host-side compare: 0 mismatches, same ~1e-7 abs as T4.

### Triton (4090) — TF32, not FP32

| N | Triton ms | torch.matmul ms | Triton GF/s | torch GF/s | % torch | Max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.037 | 0.046 | 58429 | 47211 | 123.8 | 3.95e-02 |
| 2048 | 0.245 | 0.311 | 70134 | 55225 | 127.0 | 5.66e-02 |
| 4096 | 1.902 | 2.347 | 72244 | 58568 | 123.4 | 9.12e-02 |

Do **not** quote “Triton beats cuBLAS.” `max_abs` is ~1e-2 vs CUDA FP32 ~1e-4. On Ada, `tl.dot` in float32 uses **TF32 Tensor Cores**. torch.matmul 58568 GF/s matches the C++ FP32 cuBLAS row (~57885). Different math. T4 Triton 81% is the fair FP32 comparison (T4 has no TF32).

## Harness rules

- Correctness vs cuBLAS SGEMM (row-major via swapped A/B). Relative gate: `max_abs_diff / max(max_abs_ref, tiny) ≤ 1e-4` for FP32.
- `cudaEvent` timing: 3 warmup + 10 timed; mean excludes warmup. cuBLAS uses the same harness.
- GFLOP/s = `2·M·N·K / (ms · 1e6)` with the product in `double`.
- Seed A with 42 and B with 43 so square A and B differ, but runs stay deterministic.
- `getRegisteredStages()` lives only in `stage_registry.cu`. Attention is a separate binary (`bench_attn`), not a GEMM `--stage`.

## Run on Colab

Open [`notebooks/colab_week1.ipynb`](notebooks/colab_week1.ipynb). **Runtime → Change runtime type → Hardware accelerator = T4 GPU**, then **Runtime → Run all**. A CPU runtime has no `nvidia-smi` and no `nvcc`; that is the `command not found` / `sm_` error. Changing the accelerator reconnects you to a new VM, so clone again after the switch.

Week 4 attention + Triton: [`notebooks/colab_week4.ipynb`](notebooks/colab_week4.ipynb).

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
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
echo "building for $ARCH"
nvcc -O3 -arch=$ARCH -lineinfo benchmark/bench.cu src/*.cu -lcublas -o bench
```

```bash
./bench --stage naive --sizes 1024 2048 4096
./bench --stage coalesced --sizes 1024 2048 4096
```

If the GPU name in the harness header is not T4, do not edit the results table yourself — send the tables back.

## Build locally (when `nvcc` is available)

```bash
ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
nvcc -O3 -arch=$ARCH -lineinfo benchmark/bench.cu src/*.cu -lcublas -o bench
./bench --stage naive --sizes 1024 2048 4096
./bench --stage coalesced --sizes 1024 2048 4096
```

Unknown `--stage` names print the registered list and exit.
