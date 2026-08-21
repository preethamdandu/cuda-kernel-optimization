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
- Root cause: not known from the first run. The 4096 rerun printed `max|ref|=110.7` and still `max abs=0` for both stages. cuBLAS produced a real matrix (magnitude O(100) is right for 16M random length-4096 dots). The comparison loop reported bitwise identity, which sequential FP32 vs cuBLAS does not do. Almost certainly nvcc `-O3` eating `max_diff = std::max(max_diff, fabs(a-b))` at N=16,777,216, not a perfect kernel.

### Next move
- Quote error from N=2048 in the README. Replaced the `std::max` reduction with a plain `if (diff > max_diff)` loop and a mismatch counter. Re-run `./bench --stage naive --sizes 4096` once after `git pull` if you want the honest 4096 error; not required to close Week 1.

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
