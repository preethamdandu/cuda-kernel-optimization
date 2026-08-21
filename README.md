# CUDA Kernel Optimization Suite

SGEMM optimization ladder from an honest uncoalesced baseline through coalescing, tiling, register blocking, vectorized loads, and Tensor Cores — then the same ideas applied to fused attention.

**GPU:** Numbers TBD until Colab run; GPU name will come from harness header.

| Stage | Kernel | Precision | GFLOP/s | % cuBLAS | Req GB/s | Max abs err | Max rel err | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Naive (uncoalesced) | FP32 | TBD | TBD | TBD | TBD | TBD | `threadIdx.x` → row |
| 2 | Coalesced | FP32 | TBD | TBD | TBD | TBD | TBD | `threadIdx.x` → col |
| 3 | Shared-memory tiled | FP32 | TBD | TBD | TBD | TBD | TBD | Reuse A/B tiles in shared memory |
| 4 | Register blocked | FP32 | TBD | TBD | TBD | TBD | TBD | Each thread owns a small C tile |
| 5 | Vectorized loads | FP32 | TBD | TBD | TBD | TBD | TBD | Wider transfers such as `float4` |
| 6 | WMMA / Tensor Cores | FP16→FP32 | TBD | TBD | TBD | TBD | TBD | Compare against cuBLAS FP16 |

The only difference between these two kernels is which thread index drives the row. (For square M=N=K — the sizes in this table — the two grid expressions evaluate to identical `dim3` values; they would differ for non-square.)

Both kernels use `dim3 block(32, 32)`. Do not fill GFLOP/s, %, Req GB/s, or error from memory — paste Colab tables back into chat.

## Why coalescing matters here

A CUDA warp is 32 consecutive `threadIdx.x` values (same `threadIdx.y` in a 32×32 block). Global memory coalesces when those 32 threads hit a contiguous 128-byte span (4 × 32-byte sectors).

- **Naive (uncoalesced):** `threadIdx.x` is the row, `threadIdx.y` is the column. Adjacent warp threads write `C[row+i][col]` — N floats apart — so the C store (and the A load) issue ~32 sectors per request instead of 4.
- **Coalesced:** `threadIdx.x` is the column, `threadIdx.y` is the row. Adjacent warp threads write `C[row][col+i]` — consecutive addresses — so the C store (and the B load) collapse into one 128-byte transaction.

Arithmetic is identical: one thread, one `C[i,j]`, K FMAs from global memory.

**Req GB/s** is requested traffic `(2·M·N·K + M·N) × 4` bytes over kernel time — not DRAM bandwidth. Compare it to **Peak DRAM** printed from `cudaGetDeviceProperties` (T4 should be ~320 GB/s). Cache reuse means requested bytes overcount actual DRAM.

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
