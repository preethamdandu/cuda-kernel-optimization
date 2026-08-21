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
