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

## 2026-08-21 — N=4096 printed exact-zero error

### Attempt
- Stage: 1 and 2 correctness vs cuBLAS at 1024 / 2048 / 4096 on Colab T4.

### What happened
- Symptom: N=1024 and N=2048 reported identical error on both stages (`6.866e-05` / `1.269e-06` and `1.526e-04` / `1.987e-06`). N=4096 printed `0.000e+00` abs and rel for both stages.
- Evidence: GFLOP/s at 4096 recomputes from the reported ms (`2·4096³ / (2226.4778e-3 · 1e9) = 61.73`). Timing is fine. A length-4096 sequential FP32 dot vs cuBLAS should not be bitwise identical; √K · ε is ~7.6e-6 relative.

### Why it failed
- Root cause: not `std::max`. Stages 3–5 used the rewritten `if (diff > max_diff)` loop and still printed 0 mismatches / 0 abs at N=4096 with `max|ref|=110.7`. N=1024/2048 still show ~1e6 / ~4e6 mismatches. nvcc `-O3` is treating the two 16M-float buffers as equal (pointer aliasing / loop elision) only above that size.

### Next move
- Quote error from N=2048. Mark compare-loop pointers `volatile` so the loads cannot be skipped. If 4096 is still zero after that, move the checker out of the `.cu` into a `.cpp` compiled by the host compiler.

## 2026-08-21 — WMMA printed zero error at every size

### Attempt
- Stage: 6 vs `cublasGemmEx` FP16 at 1024 / 2048 / 4096.

### What happened
- Symptom: 0 mismatches, 0 abs, 0 rel at **all three** sizes. `max|ref|` was 54 / 77 / 111 (real). Timing recomputes (4096: 4192 GFLOP/s from 32.79 ms).
- Evidence: FP32 stages on the same checker at 1024/2048 reported ~1e6 / ~4e6 mismatches. WMMA hitting exact zero at 1024 is a new failure mode, not just the 4096 nvcc bug.

### Why it failed
- Root cause: unverified. Either the two Tensor Core paths are bit-identical, or the compare loop is still eliding diffs. Do not treat this as a correctness pass.

### Next move
- Re-run `./bench --stage vectorized --sizes 1024` on the WMMA binary. If mismatches come back, the checker still works and WMMA==cuBLAS is a real (surprising) result. If that is also zero, move the checker to a `.cpp` host TU.

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
- Evidence: host-side compare passed at all sizes (max abs ~1e-7, 0 mismatches). Timing is real.

### Why it failed
- Root cause: T4 L2 is 4 MiB. Unfused scores are 0.25 / 1.00 / 4.00 MiB. Fusion avoids a write that L2 was already covering, and pays `__syncthreads()` + running softmax state. Same shape as Stage 3 tiling losing at N=1024.

### Next move
- Leave the teaching kernel. The interview point is the crossover, not a 1.47× headline. Do not retune tiles to manufacture a win at 256. Optional later: PyTorch SDPA MATH/EFFICIENT on the same T4 (sm_75 has no FlashAttention CUDA backend).

## Planned — Colab ncu permission wall

### Attempt
- Stage: profiling (Week 3)
- Change tried: Run `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,...` on Colab T4 from `notebooks/colab_week1.ipynb`.

### What happened
- Symptom: expected `ERR_NVGPU_PERMISSION` / counters denied. Colab does not expose the privileged counters Nsight Compute needs.
- Evidence: not measured yet; this is a planned dead-end, not a surprise.

### Why it failed
- Root cause: Colab's NVIDIA driver/container blocks ncu hardware counters. This is a platform restriction, not a kernel bug.

### Next move
- Log the Colab failure, then collect sectors/request on a rented RTX 4090 in Week 3. Expected contrast (not measured here): kernel-average ~16.5 vs ~2.5 sectors/request; 32 vs 4 on the A-load instruction; C store 32 vs 4 via `op_st`.
