"""Tiled SGEMM in Triton, following the official blocked-matmul tutorial.

Compares one kernel against torch.matmul (cuBLAS) at 1024/2048/4096.
This is the productivity-vs-control row: ~80 lines of Python vs the CUDA
register-blocked / vectorized ladder, on the same T4.
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


def _sync() -> None:
    torch.cuda.synchronize()


def bench_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - start) * 1e3 / iters


def gflops(n: int, ms: float) -> float:
    return 2.0 * (n**3) / (ms * 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton tiled SGEMM vs torch.matmul")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device. Run this on Colab T4, not the Mac.")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("dtype: float32  |  Triton BLOCK 64x64x32  |  baseline = torch.matmul (cuBLAS)")
    print()
    print(
        f"{'N':>6} {'triton_ms':>12} {'torch_ms':>12} {'triton_GF/s':>14} "
        f"{'torch_GF/s':>14} {'% torch':>10} {'max_abs':>12}"
    )

    for n in args.sizes:
        torch.manual_seed(0)
        a = torch.empty((n, n), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
        b = torch.empty((n, n), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)

        c_triton = triton_matmul(a, b)
        c_torch = torch.matmul(a, b)
        max_abs = (c_triton - c_torch).abs().max().item()

        triton_ms = bench_ms(lambda: triton_matmul(a, b))
        torch_ms = bench_ms(lambda: torch.matmul(a, b))
        t_g = gflops(n, triton_ms)
        b_g = gflops(n, torch_ms)
        pct = 100.0 * t_g / b_g if b_g > 0 else 0.0
        print(
            f"{n:6d} {triton_ms:12.4f} {torch_ms:12.4f} {t_g:14.1f} "
            f"{b_g:14.1f} {pct:9.1f}% {max_abs:12.3e}"
        )


if __name__ == "__main__":
    main()
