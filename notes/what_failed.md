# What Failed

This file tracks dead ends, wrong assumptions, and regressions during kernel optimization. The goal is to preserve the reasoning behind each fix, not just the final fast kernel.

## 2026-08-21 — Naive kernel was already coalesced

### Attempt
- Stage: 1 (naive) then 2 (coalesced)
- Change tried: Ship a "naive" one-thread-per-output SGEMM, then remap threads in Stage 2 for coalescing.

### What happened
- Symptom: Stage 2 showed no speedup versus Stage 1.
- Evidence: Both kernels used `threadIdx.x` → column, which is the coalesced mapping. Stage 2 had nothing left to fix.

### Why it failed
- Root cause: My first 'naive' kernel was accidentally already coalesced — I'd mapped threadIdx.x to the column without realizing that was the optimization. Caught it when Stage 2 showed no speedup. Rewrote Stage 1 to the true uncoalesced mapping to establish an honest baseline.

### Next move
- Stage 1 now maps `threadIdx.x` → row (strided C stores / A loads). Stage 2 maps `threadIdx.x` → col. Same 32×32 block. Same arithmetic.

## 2026-08-21 — Chased a compiler bug that was not there: the exact-zero error

### Attempt
- Stage: 1 and 2 correctness vs cuBLAS at 1024 / 2048 / 4096 on Colab T4.

### What happened
- Symptom: N=1024 and N=2048 reported real error on both stages (`6.866e-05` / `1.269e-06` and `1.526e-04` / `1.987e-06`). N=4096 printed `0.000e+00` abs and rel with `max|ref| = 110.7`.
- I assumed the compare loop was being optimized away, and spent two rounds on it: replaced `std::max` with an explicit `if (diff > max_diff)`, then marked the pointers `volatile`. **Neither changed anything.**

### Why it failed
- It was never the compiler. Three things say so:
  1. The `volatile` version still printed zero, and `tests/test_error_stats.cpp` now compiles the same loop with `clang++ -O3` and catches a single perturbed element in a 4096² buffer. The loop works.
  2. The affected size **moved with the GPU**: zero at 4096 on the T4, zero at 1024 on the 4090. A compiler bug does not follow the hardware. A cuBLAS kernel-selection heuristic does.
  3. On the 4090, WMMA reported ~1e6 / ~4e6 / ~17e6 differing elements *in the same binary with the same compare loop*.
- Real root cause: all five FP32 kernels accumulate `acc += a[k]*b[k]` in strictly increasing k with a single accumulator. Whenever cuBLAS picks a kernel with that same order and no split-K, the results are **bit-identical**. Zero is the correct answer, not a broken checker.

### Next move
- Stop calling it a bug. The harness now prints a note explaining the bitwise match instead of a warning.
- This is also why Stages 1–5 always report *identical* error to each other: same FMA order, same rounding. That is a useful correctness signal, not a coincidence.

### Lesson
- "My tool is broken" was the expensive hypothesis and the wrong one. The cheap test — run the pure-host part of the tool against a known-bad input — should have come first. It takes no GPU and no nvcc.

## 2026-08-21 — WMMA zero error on T4, resolved by the 4090 run

### Attempt
- Stage: 6 vs `cublasGemmEx` FP16 at 1024 / 2048 / 4096.

### What happened
- On T4: 0 differing elements at all three sizes, `max|ref|` 54 / 77 / 111. Flagged unverified at the time.
- On 4090: the same kernel reported 1,023,911 / 4,150,276 / 16,713,183 differing elements with max rel 2.1e-06 / 3.2e-06 / 7.1e-06 — comfortably inside the 1e-2 gate.

### Why it failed
- It did not. Same story as the FP32 zeros: on T4 the WMMA kernel and `cublasGemmEx` happened to issue the same `m16n16k16` MMA sequence in the same order, so the accumulators matched bit for bit. On Ada, cuBLAS picks a different tiling and the results diverge by normal FP16-input rounding.

### Next move
- Quote the 4090 error column as the real correctness evidence for Stage 6. Both GPUs pass; only one of them is interesting.

## 2026-08-21 — 32×32 tiling lost to coalesced at N=1024

### Attempt
- Stage: 3 (shared-memory 32×32 tiles) vs Stage 2 (coalesced, no shared).

### What happened
- Symptom: expected 2–3× over coalesced. Measured 0.58× at 1024 (365 vs 623 GFLOP/s) and only 1.25× at 4096 (726 vs 580).
- Evidence: same T4 session, same error as Stage 2 at 1024/2048 (same math).

### Why it failed
- Root cause: one-thread-per-output 32×32 tiling still loads every A/B element from global once per output row/col of the *block*, but at N=1024 the coalesced kernel already streams from L2. Tiling adds two `__syncthreads()` per K-chunk and extra shared round-trips without enough extra reuse. The 2–3× textbook jump needs register blocking, not this tile size.

### Next move
- Leave Stage 3 as the teaching kernel. Do not retune it to beat Stage 2. Stage 4 (4×4 per thread) is the actual 3.8× jump.

## 2026-08-21 — Fused attention lost at seq=256 and 512

### Attempt
- Stage: Week 4 fused vs unfused SDPA on Colab T4 (`head_dim=64`).
- Change tried: one kernel, online softmax, 32×64 K/V tiles in smem, vs three kernels that materialize `seq×seq` scores.

### What happened
- Symptom: fused was **0.67×** unfused at 256 (0.829 vs 0.555 ms) and **0.93×** at 512. Only **1.47×** at 1024 (3.328 vs 4.894 ms).
- Evidence: host-side compare passed at all sizes (max abs ~1e-7, 0 differing elements past tolerance). Timing is real.

### Why it failed
- Written up at the time as an L2 story: T4 L2 is 4 MiB, unfused scores are 0.25 / 1.00 / 4.00 MiB, so fusion only starts avoiding real DRAM traffic at 1024. That effect is real but it is **not** the main term. See the entry below — the kernel was launching one warp per block.

### Next move
- Superseded by the v2 rewrite. Kept here because the original conclusion was wrong and the correction is the point.

## 2026-08-21 — ncu ERR_NVGPUCTRPERM on Colab T4 and RunPod 4090

### Attempt
- Stage: Week 3 profiling.
- Change tried: `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,... --kernel-name regex:naiveSgemmKernel --launch-count 1` (and coalesced / register) as **root** on RunPod RTX 4090, Nsight Compute 2025.1.1, CUDA 12.8. Same metrics on Colab T4 earlier.

### What happened
- Symptom: `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.`
- Evidence: no `profiles/*.ncu-rep` files (`Could not open report file`). Bench still printed GFLOP/s; only counters were denied. RunPod image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.

### Why it failed
- Root cause: the NVIDIA driver in the container/hypervisor has profiling restricted (`RmProfilingAdminOnly` / guest counter policy). Root inside the pod is not host privilege. Same class of wall as Colab, not a bad `ncu` flag.

### Next move
- Do not rent another 4090 for the same command. Sectors/request needs a **privileged** VM or host `NVreg_RestrictProfiling=0`. Until then the coalescing claim is the 8.2× wall-clock (4090) / 9.4× (T4), not ncu.

## 2026-08-21 — 32×32 tiling lost to coalesced at every size on 4090

### Attempt
- Stage: 3 vs 2 on RunPod RTX 4090.

### What happened
- Symptom: tiled was slower than coalesced at 1024 (4803 vs 5584), 2048 (4836 vs 5620), and 4096 (4517 vs 5603). T4 had a 1.25× win at 4096; that vanished.
- Evidence: same error as coalesced at 2048/4096 (same math).

### Why it failed
- Root cause: same teaching tile as T4. 4090's larger L2 makes the no-shared coalesced kernel even harder to beat without register blocking.

### Next move
- Leave Stage 3. Quote the 4090 loss; do not retune.

## 2026-08-21 — Fused attention lost at every seq on 4090

### Attempt
- Stage: fused vs unfused SDPA on RTX 4090 (`head_dim=64`).

### What happened
- Symptom: speedup **0.38× / 0.37× / 0.45×** at 256 / 512 / 1024. T4's 1.47× at 1024 did not repeat.
- Evidence: host-side abs error ~1e-7, nothing past tolerance (same as T4).

### Why it failed
- First explanation was again L2: the 4090 has 72 MiB, the score matrix is at most 4 MiB, so fusion never avoids a DRAM round-trip. Directionally true, and it is why the T4 crossover disappears. But it does not explain a **0.38×**, and it is not what a bigger L2 alone would do.

### Next move
- Look at the launch configuration before blaming the memory hierarchy. See below.

## 2026-08-21 — The real bug: fused attention was running 32 warps on a 128-SM GPU

### Attempt
- Re-read `src/08_attention_fused.cu` after the 4090 numbers came back worse than the T4 numbers, which the L2 story does not predict.

### What happened
- `launchFusedAttention` was `fusedAttentionKernel<<<(seq + 31) / 32, 32>>>`: **one thread per Q row, 32 threads per block**. At seq=1024 that is 32 blocks of one warp — **32 warps for the entire GPU**. A 4090 has 128 SMs, so 96 of them were idle for the whole kernel, and the 32 that had work ran a single warp each with no latency hiding.
- Meanwhile the unfused path launches `qkKernel` with 1024 blocks of 1024 threads. So the measurement was never "fusion versus materialization." It was "a well-parallelized three-kernel path versus a kernel that used a quarter of one percent of the machine."
- Each thread also held `acc[64]` plus `s[32]`, and the `s[]` loops are bounded by a runtime `tile`, so that array cannot stay in registers. `-Xptxas -v` now reports the spill directly.

### Why it failed
- I reached for an architectural explanation (cache hierarchy) for a number that had a launch-configuration cause. The L2 argument was plausible enough to stop the investigation, which is exactly what made it expensive. Two GPUs disagreeing in the *wrong direction* was the clue: a bigger, faster GPU making a kernel relatively worse points at parallelism, not at caches.

### Next move
- `src/09_attention_fused_v2.cu`: one warp per Q row, 128-thread blocks, 4 Q rows per block. At seq=1024 that is 256 blocks and **1024 warps**, a 32× increase. head_dim 64 splits over 32 lanes as two accumulator floats each, so `acc[64]` and `s[32]` are gone.
- Within a tile, lane j owns K row j and computes the full 64-long dot out of shared memory, so `s_j` stays in a register and the dot needs no shuffle. The two softmax reductions across the 32 K rows are `__shfl_xor_sync`.
- `tile_k` and `tile_v` are padded to 65 floats per row: `(j * 65 + d) % 32 == (j + d) % 32`, so the lane-varying reads hit 32 distinct banks.
- v1 is kept in the binary and benchmarked next to v2, so the occupancy table and the speedup are side by side rather than described from memory.
- `tests/test_fused_attention_math.cpp` simulates the v2 recurrence on the host and checks it against an FP64 reference at seq 1, 3, 32, 33, 64, 100, 128, 129, 256, plus a large-logit case that forces the rescale path. That validates the algorithm without a GPU.

## 2026-08-21 — Triton 123% of torch.matmul on 4090 is TF32

### Attempt
- Stage: `triton/matmul.py` vs `torch.matmul` on 4090.

### What happened
- Symptom: Triton 72244 vs torch 58568 GFLOP/s at 4096 (**123.4%**). Max abs **9.12e-02**. CUDA FP32 stages were ~3e-4 abs.
- Evidence: T4 Triton max abs was 7.6e-05 / 1.6e-04 (real FP32). 4090 error is two orders larger.

### Why it failed
- Root cause: Ada `tl.dot` on float32 tensors uses TF32 Tensor Cores. Not a fair vs cuBLAS FP32. Do not put 123% on a resume.

### Next move
- Quote T4 Triton **80.8% of torch.matmul** as the FP32 row. If a TF32 row is wanted later, set torch and Triton to TF32 explicitly and label it.
