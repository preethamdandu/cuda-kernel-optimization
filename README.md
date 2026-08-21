# CUDA Kernel Optimization Suite

SGEMM optimization ladder from an honest uncoalesced baseline through coalescing, tiling, register blocking, vectorized loads, and Tensor Cores — then the same ideas applied to fused attention.

**GPU:** Tesla T4, 40 SMs, sm_75. Peak DRAM 320.06 GB/s (from `cudaGetDeviceProperties`: mem clock 5001 MHz, 256-bit bus). Colab, 2026-08-21. Headline numbers are N=4096.

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Req GB/s | Max abs err | Max rel err | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Naive (uncoalesced) | FP32 | 61.73 | 1.48 | 247 | 1.53e-04† | 1.99e-06† | `threadIdx.x` → row |
| 2 | Coalesced | FP32 | 580.13 | 13.89 | 2321 | 1.53e-04† | 1.99e-06† | `threadIdx.x` → col |
| 3 | Shared-memory tiled | FP32 | TBD | TBD | TBD | TBD | TBD | Reuse A/B tiles in shared memory |
| 4 | Register blocked | FP32 | TBD | TBD | TBD | TBD | TBD | Each thread owns a small C tile |
| 5 | Vectorized loads | FP32 | TBD | TBD | TBD | TBD | TBD | Wider transfers such as `float4` |
| 6 | WMMA / Tensor Cores | FP16→FP32 | TBD | TBD | TBD | TBD | TBD | Compare against cuBLAS FP16 |

† Error quoted from N=2048. Both stages printed identical error there (same arithmetic). N=4096 printed `0.000e+00` for both abs and rel — that is not believable for a length-4096 FP32 dot product, so it is not in this table. See [notes/what_failed.md](notes/what_failed.md).

The only difference between these two kernels is which thread index drives the row. (For square M=N=K — the sizes in this table — the two grid expressions evaluate to identical `dim3` values; they would differ for non-square.)

Both kernels use `dim3 block(32, 32)`. Coalesced / naive at 4096 is **9.4×** (580.13 / 61.73). That is a bit above the 5–8× rule of thumb because Stage 1 is a *true* uncoalesced mapping, not the accidentally-coalesced kernel that used to sit in `01_naive.cu`.

## Stages 3–5 (code in, numbers next Colab run)

- **Stage 3 `tiled`:** 32×32 shared A/B tiles, pad +1 column, two `__syncthreads()`. Same output mapping as Stage 2. Expect ~2–3× over coalesced if reuse lands.
- **Stage 4 `register`:** 64×64 block tile, 16×16 threads, each thread a 4×4 register patch, K-panel 16. Biggest conceptual jump.
- **Stage 5 `vectorized`:** Stage 4 plus `float4` global→shared. Modest 10–30% if the extra bandwidth is the limiter.

Req GB/s in the harness still uses the no-reuse formula `(2MNK+MN)×4`. That number is meaningful vs peak for Stages 1–2. From Stage 3 on it is an upper bound, not DRAM traffic.

Do not fill TBD cells from memory. Paste Colab tables. Skip `--stage naive` this round; 4096 naive is ~2.2 s/iter.

```bash
./bench --stage tiled --sizes 1024 2048 4096
./bench --stage register --sizes 1024 2048 4096
./bench --stage vectorized --sizes 1024 2048 4096
```

### Per-size (same T4 session)

| Stage | N | Kernel GF/s | cuBLAS GF/s | % cuBLAS | Req GB/s | Peak DRAM | Max abs | Max rel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive | 1024 | 60.94 | 6271.39 | 0.97 | 243.87 | 320.06 | 6.87e-05 | 1.27e-06 |
| Naive | 2048 | 61.85 | 6661.33 | 0.93 | 247.46 | 320.06 | 1.53e-04 | 1.99e-06 |
| Naive | 4096 | 61.73 | 4170.76 | 1.48 | 246.95 | 320.06 | 0* | 0* |
| Coalesced | 1024 | 623.19 | 5262.86 | 11.84 | 2493.97 | 320.06 | 6.87e-05 | 1.27e-06 |
| Coalesced | 2048 | 580.51 | 6271.55 | 9.26 | 2322.62 | 320.06 | 1.53e-04 | 1.99e-06 |
| Coalesced | 4096 | 580.13 | 4175.41 | 13.89 | 2320.82 | 320.06 | 0* | 0* |

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
