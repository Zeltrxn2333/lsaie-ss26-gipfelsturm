#!/usr/bin/env python3
"""Microbench: cuBLAS FP8 vs DeepGEMM FP8 at Qwen3-14B MLP GEMM shapes.

Run inside alps3-mp container with DeepGEMM venv on PYTHONPATH:
    PYTHONPATH=/iopsstor/scratch/cscs/$USER/venvs/deepgemm:$PYTHONPATH \
        python3 bench/deepgemm_microbench.py

Times forward FP8 GEMM at the shapes Qwen3-14B 1n4g MBS=2 TP=4 actually uses:
    FC1 (gate+up):  [M=2048, K=5120] @ [K=5120, N=8704]
    FC2 (down):     [M=2048, K=4352] @ [K=4352, N=5120]
    Q proj:         [M=2048, K=5120] @ [K=5120, N=1280]
    O proj:         [M=2048, K=1280] @ [K=1280, N=5120]

Reports TFLOP/s achieved by each backend, ratio DeepGEMM / cuBLAS.
"""
from __future__ import annotations

import time
import torch


def cuBLAS_fp8_gemm(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    """Cast bf16 inputs to fp8, run cuBLAS FP8 GEMM via torch._scaled_mm, output bf16."""
    M, K = a_bf16.shape
    K2, N = b_bf16.shape
    assert K == K2
    a_amax = a_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    b_amax = b_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    scale_a = 448.0 / a_amax
    scale_b = 448.0 / b_amax
    a_fp8 = (a_bf16 * scale_a).to(torch.float8_e4m3fn)
    b_fp8 = (b_bf16 * scale_b).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        a_fp8, b_fp8.t().contiguous().t(),
        scale_a=1.0 / scale_a, scale_b=1.0 / scale_b,
        out_dtype=torch.bfloat16,
    )
    return out


def deepgemm_fp8_gemm(a_bf16: torch.Tensor, b_bf16: torch.Tensor, dg) -> torch.Tensor:
    """DeepGEMM FP8 GEMM: prepares scaled tensors then calls dg.gemm_fp8_fp8_bf16_nt."""
    M, K = a_bf16.shape
    K2, N = b_bf16.shape
    a_amax = a_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    b_amax = b_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    scale_a = 448.0 / a_amax
    scale_b = 448.0 / b_amax
    a_fp8 = (a_bf16 * scale_a).to(torch.float8_e4m3fn)
    b_fp8 = (b_bf16 * scale_b).to(torch.float8_e4m3fn)
    # DeepGEMM expects per-block scales for A (128x128 blocks) and per-tensor for B
    # For microbench we use simple per-tensor scales for both (cuBLAS-compatible form).
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a_bf16.device)
    # Try DeepGEMM's standard FP8 GEMM API
    try:
        dg.gemm_fp8_fp8_bf16_nt(
            (a_fp8, torch.full((M, 1), 1.0 / scale_a, dtype=torch.float32, device=a_fp8.device)),
            (b_fp8, torch.full((N, 1), 1.0 / scale_b, dtype=torch.float32, device=b_fp8.device)),
            out,
        )
    except Exception as e:
        raise RuntimeError(f"DeepGEMM call failed: {e}")
    return out


def bench_gemm(name: str, fn, a: torch.Tensor, b: torch.Tensor, warmup: int = 5, iters: int = 50):
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(a, b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - start) / iters
    M, K = a.shape
    _, N = b.shape
    flops = 2 * M * K * N
    tflops = flops / dt / 1e12
    print(f"  {name:<22}  {dt*1e3:7.3f} ms   {tflops:7.1f} TFLOP/s")
    return tflops


def main() -> int:
    import deep_gemm as dg
    print(f"DeepGEMM version: {dg.__version__}")
    device = "cuda"
    SHAPES = [
        ("FC1 (gate+up)",  2048, 5120, 8704),
        ("FC2 (down)",     2048, 4352, 5120),
        ("Q proj",         2048, 5120, 1280),
        ("O proj",         2048, 1280, 5120),
    ]
    print()
    print(f"{'GEMM':<22}  {'shape':<26}  cuBLAS_FP8         DeepGEMM_FP8       speedup")
    for name, M, K, N in SHAPES:
        print(f"{name:<22}  M={M} K={K:<5} N={N:<5}", end="  ")
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
        b = torch.randn(K, N, dtype=torch.bfloat16, device=device) * 0.5
        cublas_tf = bench_gemm("cuBLAS FP8", lambda x, y: cuBLAS_fp8_gemm(x, y), a, b)
        try:
            dg_tf = bench_gemm("DeepGEMM FP8", lambda x, y: deepgemm_fp8_gemm(x, y, dg), a, b)
            print(f"  speedup: {dg_tf/cublas_tf:.2f}×")
        except Exception as e:
            print(f"  DeepGEMM FAILED: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
