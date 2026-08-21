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
- Leave Stage 3 as the teaching kernel. Do not retune it to beat Stage 2. Stage 4 (4×4 per thread) is the actual jump.

### Update — the N=1024 loss did not reproduce
- A later full T4 session had tiled **beating** coalesced at every size: 764 vs 439 at 1024, 646 vs 530 at 2048, 657 vs 530 at 4096. The 0.58× is not a stable finding.
- Both sessions agree that N=1024 is where the measurements are least trustworthy. Coalesced at 1024 alone read 623, 439, and 513 GF/s across three runs on the same code — a 1.4× spread from clocks and cache state on a shared Colab VM. The 4096 numbers are stable to within a percent.
- What *is* stable, and is now measured, is the mechanism: tiling cuts global load requests **32×** (67.1M → 2.1M at N=1024) while leaving sectors-per-request at the coalesced ideal of 4. So the shared-memory reuse is real and doing exactly what it should. What it does not buy is enough arithmetic intensity per thread to matter, which is Stage 4's job.
- Lesson: a single-session ratio at the smallest problem size is not a result. Quote 4096, or quote a range across sessions.

## 2026-08-21 — Fused attention lost at seq=256 and 512

### Attempt
- Stage: Week 4 fused vs unfused SDPA on Colab T4 (`head_dim=64`).
- Change tried: one kernel, online softmax, 32×64 K/V tiles in smem, vs three kernels that materialize `seq×seq` scores.

### What happened
- Symptom: fused was **0.67×** unfused at 256 (0.829 vs 0.555 ms) and **0.93×** at 512. Only **1.47×** at 1024 (3.328 vs 4.894 ms).
- Evidence: host-side compare passed at all sizes (max abs ~1e-7, 0 differing elements past tolerance). Timing is real.

### Why it failed
- Written up at the time as an L2 story: T4 L2 is 4 MiB, unfused scores are 0.25 / 1.00 / 4.00 MiB, so fusion only starts avoiding real DRAM traffic at 1024. See the entry below — the kernel was launching one warp per block.
- The v2 measurements retired the L2 argument outright rather than demoting it. v2 wins **9.83×** at seq=256, its *largest* margin, at the size where the score matrix is 0.25 MiB and trivially L2-resident. If cache capacity were the mechanism, 256 is exactly where fusion should have had the least to offer.

### Next move
- Superseded by the v2 rewrite. Kept here because the original conclusion was wrong and the correction is the point.

## 2026-08-21 — ncu ERR_NVGPUCTRPERM on RunPod 4090, then it worked on Colab T4

### Attempt
- Stage: Week 3 profiling.
- Change tried: `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,... --kernel-name regex:naiveSgemmKernel --launch-count 1` (and coalesced / register) as **root** on RunPod RTX 4090, Nsight Compute 2025.1.1, CUDA 12.8.

### What happened
- On RunPod: `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.` No `profiles/*.ncu-rep` files. Bench still printed GFLOP/s; only counters were denied. Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- An earlier Colab attempt was also recorded as denied, and I generalized that into "counters are denied on every platform," which went into the README as a standing limitation.
- **On Colab T4, re-running the same metrics through `scripts/run_all.sh` returned real counters.** Every stage, first try, no flags changed.

### Why it failed
- RunPod's root cause is real: the driver in that container has profiling restricted (`RmProfilingAdminOnly` / guest counter policy), and root inside a pod is not host privilege.
- The mistake was the generalization. Two denials on two platforms became "denied everywhere," and the README then justified having no profiling evidence at all rather than retrying the cheap platform. The free environment worked the whole time.

### Next move
- Sector counts and achieved occupancy for every kernel now come from Colab, and are in the README under "Measured memory behaviour." RunPod is for the 4090 wall-clock only.
- Keep the `ncu` block in `run_all.sh` non-fatal so both outcomes get recorded per platform instead of being assumed.
- The second 4090 session (same image) denied `ncu` again and does not ship `nsys`. CUPTI tracing is not a workaround on this pod. Do not rent another 4090 to retry counters.

### Lesson
- A platform limitation is a claim about a specific platform. "It failed here" and "it fails everywhere" needed one more free retry to separate, and that retry produced the strongest evidence in the repo.

## 2026-08-21 — Predicted a register spill; ptxas says zero

### Attempt
- Stage: diagnosing why fused attention v1 was slow, before writing v2.
- Claim made: v1 keeps `float acc[64]` plus `float s[32]` per thread, and the `s[]` loops are bounded by a runtime `tile` rather than a constant, so the array cannot be register-allocated and must spill to local memory. This was written into the README and this log as a root cause **alongside** the occupancy argument.

### What happened
- Added `-Xptxas -v` specifically to prove it. It reported the opposite:

```
fusedAttentionKernel: 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
                      Used 189 registers, 16640 bytes smem
```

- `cudaFuncGetAttributes` agrees: `localSizeBytes = 0`. No spill at any point. On the 4090 the same kernel compiled to 190 registers, still zero spill.

### Why it failed
- The compiler fully unrolled the tile loops and kept all 96 floats in registers, at the cost of 189 registers per thread — high, but under the 255 cap, so no spill was needed.
- The occupancy half of the diagnosis was right and the mechanism was wrong. v1's real limiters, both now measured: only 32 blocks for 40 SMs (`launch__waves_per_multiprocessor = 0.27`, so 8 SMs idle and the resident blocks never reach the 3/SM the hardware allows), and 16,640 bytes of shared memory requested by a 32-thread block, which is 520 bytes per thread.
- Achieved occupancy 3.12% against 9.4% theoretical. v2 gets 33.19% against 37.5%.

### Next move
- Delete "spill" from the README's three references to it and quote registers, shared-memory-per-thread, and waves/SM instead. v2 still wins for the reason predicted, just not by the mechanism predicted.

### Lesson
- This is the third time in this project a plausible mechanism got asserted before it was measured, and the second time the tool added to prove a claim is what refuted it. Register pressure and register *spilling* are different failure modes with different fixes; 189 registers with no spill would have been a real problem on a kernel that had enough blocks to care.

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
- Each thread also held `acc[64]` plus `s[32]`. I predicted this spilled to local memory; `-Xptxas -v` later showed it does not — 189 registers, zero spill. See the separate entry on that. The per-thread cost that *does* bite is shared memory: 16,640 bytes for a 32-thread block, 520 bytes per thread.
- Measured on T4 once `ncu` was working: v1 achieves **3.12%** occupancy against 9.4% theoretical, at **0.27 waves per SM**. Under one wave means the grid cannot cover the GPU even once, which is why v1 falls short of its own theoretical ceiling. v2 achieves 33.19% against 37.5% theoretical at 2.13 waves/SM.

### Why it failed
- I reached for an architectural explanation (cache hierarchy) for a number that had a launch-configuration cause. The L2 argument was plausible enough to stop the investigation, which is exactly what made it expensive. Two GPUs disagreeing in the *wrong direction* was the clue: a bigger, faster GPU making a kernel relatively worse points at parallelism, not at caches.

### Next move
- `src/09_attention_fused_v2.cu`: one warp per Q row, 128-thread blocks, 4 Q rows per block. At seq=1024 that is 256 blocks and **1024 warps**, a 32× increase. head_dim 64 splits over 32 lanes as two accumulator floats each, so `acc[64]` and `s[32]` are gone.
- Within a tile, lane j owns K row j and computes the full 64-long dot out of shared memory, so `s_j` stays in a register and the dot needs no shuffle. The two softmax reductions across the 32 K rows are `__shfl_xor_sync`.
- `tile_k` and `tile_v` are padded to 65 floats per row: `(j * 65 + d) % 32 == (j + d) % 32`, so the lane-varying reads hit 32 distinct banks.
- v1 is kept in the binary and benchmarked next to v2, so the occupancy table and the speedup are side by side rather than described from memory.
- `tests/test_fused_attention_math.cpp` simulates the v2 recurrence on the host and checks it against an FP64 reference at seq 1, 3, 32, 33, 64, 100, 128, 129, 256, plus a large-logit case that forces the rescale path. That validates the algorithm without a GPU.

### Result (T4, measured)
- **9.83× / 8.46× / 8.48×** over unfused at seq 256 / 512 / 1024, against v1's 0.67× / 0.93× / 1.48×. Fusion now wins everywhere, and by the most at the shortest sequence.
- 765 GFLOP/s at seq=1024 against unfused's 90.
- A 10.6× achieved-occupancy gain converted into a 5.7× speedup over v1. v2 pays for its parallelism in shared-memory traffic and two warp-shuffle reductions per tile, so about half the theoretical gain reaches wall-clock. Worth stating as a ratio rather than implying occupancy converts one-for-one.

### Result (4090, measured)
- v1 still **0.38× / 0.37× / 0.45×**. v2 **7.53× / 8.21× / 8.80×** over unfused, **~20×** over v1, 4236 GFLOP/s at seq=1024.
- Occupancy API: 32 vs 1024 warps, 10.4% vs 41.7% theoretical. `ncu` denied (`ERR_NVGPUCTRPERM`), so there is no achieved-occupancy number on this GPU. `ptxas`: v1 190 registers, **zero spill**; v2 56 registers, zero spill. Same "no spill" finding as T4.

## 2026-08-21 — Triton 123% of torch.matmul on 4090 is TF32

### Attempt
- Stage: `triton/matmul.py` vs `torch.matmul` on 4090.

### What happened
- Symptom: Triton 72244 vs torch 58568 GFLOP/s at 4096 (**123.4%**). Max abs **9.12e-02**. CUDA FP32 stages were ~3e-4 abs.
- Evidence: T4 Triton max abs was 7.6e-05 / 1.6e-04 (real FP32). 4090 error is two orders larger.

### Why it failed
- Root cause: Ada `tl.dot` on float32 tensors uses TF32 Tensor Cores. Not a fair vs cuBLAS FP32. Do not put 123% on a resume.

### Fix, verified on T4 and 4090
- `triton/matmul.py` now passes `input_precision="ieee"` to `tl.dot`, sets `torch.backends.cuda.matmul.allow_tf32 = False`, and gates on 1e-4 relative. A `--tf32` flag reports the Tensor Core row separately with the precision printed in the header.
- On T4 both modes give byte-identical error (7.63e-05 / 1.60e-04) and the same ~78% ratio: sm_75 has no TF32 path, so the flag is a no-op. The T4 comparison was always fair.
- On 4090 IEEE FP32: Triton **87.6%** of torch at 4096, max rel **3.24e-6**, torch 56606 GF/s matching the C++ cuBLAS FP32 row. That is the check the T4 could not provide.
- On 4090 TF32-vs-TF32: Triton 88.6% of torch at 4096, max abs **1.03e-01**, torch 80499 GF/s. The old 123% was TF32 Triton against FP32 torch. With both sides on TF32 the ratio falls below 100% and the error stays two orders above IEEE.

### Next move
- Quote 4090 IEEE **87.6%** and T4 IEEE **78.3%** as the FP32 rows. Quote the TF32 row only with the precision labelled. Never quote 123%.
