#!/usr/bin/env bash
# One script for every GPU so the T4 and 4090 sessions cannot drift apart.
#
#   bash scripts/run_all.sh            # build + host tests + all benchmarks
#   bash scripts/run_all.sh --quick    # skip naive at 4096 (the ~200 ms/iter one)
#
# -Xptxas -v prints registers, shared memory, and spill bytes per kernel. That
# is compile-time information, so it works on Colab and RunPod even though both
# deny the hardware counters Nsight Compute needs.

set -euo pipefail

QUICK=0
if [ "${1:-}" = "--quick" ]; then
  QUICK=1
fi

if ! command -v nvidia-smi >/dev/null || ! command -v nvcc >/dev/null; then
  echo "No GPU or no nvcc in this environment." >&2
  echo "Colab: Runtime -> Change runtime type -> T4 GPU." >&2
  exit 1
fi

ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "=== GPU: $GPU  building for $ARCH ==="
echo

echo "=== Host-only tests (no GPU needed, run here for the record) ==="
c++ -O3 -std=c++17 tests/test_error_stats.cpp -o test_error_stats
./test_error_stats
echo
c++ -O3 -std=c++17 tests/test_fused_attention_math.cpp -o test_attn_math
./test_attn_math
echo

echo "=== Building bench (registers / spills below) ==="
nvcc -O3 -arch="$ARCH" -lineinfo -Xptxas -v \
  benchmark/bench.cu \
  src/01_naive.cu src/02_coalesced.cu src/03_tiled.cu \
  src/04_register_blocked.cu src/05_vectorized.cu src/06_wmma_tensorcore.cu \
  src/stage_registry.cu \
  -lcublas -o bench
echo

echo "=== Building bench_attn (registers / spills below) ==="
nvcc -O3 -arch="$ARCH" -lineinfo -Xptxas -v \
  benchmark/bench_attn.cu \
  src/07_attention_unfused.cu src/08_attention_fused.cu src/09_attention_fused_v2.cu \
  -o bench_attn
echo

for stage in naive coalesced tiled register vectorized wmma; do
  echo "=== ./bench --stage $stage ==="
  if [ "$QUICK" = "1" ] && [ "$stage" = "naive" ]; then
    ./bench --stage "$stage" --sizes 1024 2048
  else
    ./bench --stage "$stage" --sizes 1024 2048 4096
  fi
  echo
done

echo "=== ./bench_attn (unfused vs fused v1 vs fused v2) ==="
./bench_attn --seqs 256 512 1024
echo

echo "=== Triton, IEEE FP32 on both sides ==="
python3 triton/matmul.py --sizes 1024 2048 4096 || echo "(triton FP32 run failed; see above)"
echo

echo "=== Triton, TF32 on both sides (labelled, not comparable to the CUDA ladder) ==="
python3 triton/matmul.py --sizes 1024 2048 4096 --tf32 || echo "(triton TF32 run failed; see above)"
echo

echo "=== nsys (CUPTI tracing, often allowed where ncu counters are not) ==="
if command -v nsys >/dev/null; then
  mkdir -p profiles
  nsys profile --force-overwrite true -t cuda -o profiles/sgemm_1024 \
    ./bench --stage register --sizes 1024 >/dev/null 2>&1 || echo "nsys profile failed"
  nsys stats --report cuda_gpu_kern_sum profiles/sgemm_1024.nsys-rep 2>/dev/null ||
    echo "nsys stats unavailable"
else
  echo "nsys not installed in this image."
fi
echo

echo "=== ncu (expected to be denied; recorded so the wall is documented) ==="
if command -v ncu >/dev/null; then
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
    --kernel-name regex:naiveSgemmKernel --launch-count 1 \
    ./bench --stage naive --sizes 1024 2>&1 | grep -E "ERR_|sectors|requests" || true
else
  echo "ncu not installed in this image."
fi

echo
echo "=== done on $GPU ($ARCH) ==="
