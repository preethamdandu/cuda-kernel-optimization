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

Correctness printed 0 mismatches at 1024, 2048, and 4096 with `max|ref|` 54 / 77 / 111. FP32 stages at 1024/2048 had ~1e6 / ~4e6 mismatches. Either WMMA matches `cublasGemmEx` bitwise (possible if both hit the same MMA) or the compare loop is still lying. Error column is **unverified** until `./bench --stage vectorized --sizes 1024` still shows ~1e6 mismatches on this binary. See [notes/what_failed.md](notes/what_failed.md).

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

## Week 3 ncu (expected, not measured)

On a machine where Nsight Compute counters are allowed (rented 4090, not Colab):

- Kernel-average sectors/request ~16.5 (naive) vs ~2.5 (coalesced)
- The 32-vs-4 contrast lives in the A-load instruction specifically
- C store: 32 vs 4 sectors, collected via `op_st`

Colab `ncu` is expected to print `ERR_NVGPU_PERMISSION` / counters denied. That is a platform wall, not a kernel bug.

## Project layout

```text
.
├── README.md
├── benchmark/
│   └── bench.cu
├── notebooks/
│   └── colab_week1.ipynb
├── notes/
│   └── what_failed.md
├── profiles/
└── src/
    ├── 01_naive.cu
    ├── 02_coalesced.cu
    ├── 03_tiled.cu
    ├── 04_register_blocked.cu
    ├── 05_vectorized.cu
    ├── 06_wmma_tensorcore.cu
    ├── stage_registry.cu
    └── stage_registry.cuh
```

## Harness rules

- Correctness vs cuBLAS SGEMM (row-major via swapped A/B). Relative gate: `max_abs_diff / max(max_abs_ref, tiny) ≤ 1e-4` for FP32.
- `cudaEvent` timing: 3 warmup + 10 timed; mean excludes warmup. cuBLAS uses the same harness.
- GFLOP/s = `2·M·N·K / (ms · 1e6)` with the product in `double`.
- Seed A with 42 and B with 43 so square A and B differ, but runs stay deterministic.
- `src/*.cu` must link: `getRegisteredStages()` lives only in `stage_registry.cu`. Stages 3–6 are comment-only TUs.

## Run on Colab

Open [`notebooks/colab_week1.ipynb`](notebooks/colab_week1.ipynb). **Runtime → Change runtime type → Hardware accelerator = T4 GPU**, then **Runtime → Run all**. A CPU runtime has no `nvidia-smi` and no `nvcc`; that is the `command not found` / `sm_` error. Changing the accelerator reconnects you to a new VM, so clone again after the switch.

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
