#!/usr/bin/env python3
"""Microbench: cuBLAS FP8 vs DeepGEMM FP8 at Qwen3-14B MLP GEMM shapes.

Run inside alps3-mp container with DeepGEMM venv on PYTHONPATH:
    PYTHONPATH=/iopsstor/scratch/cscs/$USER/venvs/deepgemm:$PYTHONPATH \
        python3 bench/deepgemm_microbench.py

Times *only the GEMM call* (FP8 cast / scale prep is done once outside the loop).
"""
from __future__ import annotations

import time
import torch


BLOCK = 128


def prep_cublas_fp8(a_bf16: torch.Tensor, b_bf16: torch.Tensor):
    M, K = a_bf16.shape
    _, N = b_bf16.shape
    a_amax = a_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    b_amax = b_bf16.abs().max().to(torch.float32).clamp(min=1e-4)
    scale_a = 448.0 / a_amax
    scale_b = 448.0 / b_amax
    a_fp8 = (a_bf16 * scale_a).to(torch.float8_e4m3fn)
    b_fp8 = (b_bf16 * scale_b).to(torch.float8_e4m3fn)
    inv_a = (1.0 / scale_a).to(torch.float32)
    inv_b = (1.0 / scale_b).to(torch.float32)
    # cuBLAS FP8 wants B in "transposed" form (column-major == .t().contiguous().t())
    b_fp8_for_cublas = b_fp8.t().contiguous().t()
    return (a_fp8, b_fp8_for_cublas, inv_a, inv_b)


def run_cublas_fp8(state) -> torch.Tensor:
    a_fp8, b_fp8, inv_a, inv_b = state
    return torch._scaled_mm(
        a_fp8, b_fp8,
        scale_a=inv_a, scale_b=inv_b,
        out_dtype=torch.bfloat16,
    )


def prep_deepgemm_fp8(a_bf16: torch.Tensor, b_bf16: torch.Tensor, dg):
    """Prepare per-token scales for A and per-128x128-block scales for B (DeepGEMM NT layout)."""
    M, K = a_bf16.shape
    _, N = b_bf16.shape
    device = a_bf16.device
    # Per-token amax for A
    a_amax = a_bf16.abs().amax(dim=1).to(torch.float32).clamp(min=1e-4)
    scale_a = 448.0 / a_amax  # [M]
    a_fp8 = (a_bf16 * scale_a[:, None]).to(torch.float8_e4m3fn)
    inv_a = (1.0 / scale_a).contiguous()  # [M]

    # B in [N, K] layout (NT = "B already transposed compared to A@B")
    b_nt = b_bf16.t().contiguous()  # [N, K]
    # 128x128 block amax via reshape + max along block dims
    pad_N = (-N) % BLOCK
    pad_K = (-K) % BLOCK
    if pad_N or pad_K:
        b_padded = torch.nn.functional.pad(b_nt, (0, pad_K, 0, pad_N))
    else:
        b_padded = b_nt
    Np, Kp = b_padded.shape
    Nb, Kb = Np // BLOCK, Kp // BLOCK
    b_blocks = b_padded.view(Nb, BLOCK, Kb, BLOCK)
    b_block_amax = b_blocks.abs().amax(dim=(1, 3)).to(torch.float32).clamp(min=1e-4)  # [Nb, Kb]
    scale_b = 448.0 / b_block_amax  # [Nb, Kb]
    # Cast B per-block: broadcast scale to [Nb, BLOCK, Kb, BLOCK] form
    scale_b_full = scale_b[:, None, :, None].expand(Nb, BLOCK, Kb, BLOCK).reshape(Np, Kp)
    b_fp8_padded = (b_padded * scale_b_full).to(torch.float8_e4m3fn)
    b_fp8 = b_fp8_padded[:N, :K].contiguous()
    inv_b = (1.0 / scale_b).contiguous()

    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    return (a_fp8, inv_a, b_fp8, inv_b, out, dg)


def run_deepgemm_fp8(state) -> torch.Tensor:
    a_fp8, inv_a, b_fp8, inv_b, out, dg = state
    dg.fp8_gemm_nt((a_fp8, inv_a), (b_fp8, inv_b), out)
    return out


def bench(name: str, run_fn, state, M: int, K: int, N: int,
          warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        run_fn(state)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_fn(state)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - start) / iters
    flops = 2 * M * K * N
    tflops = flops / dt / 1e12
    print(f"    {name:<20}  {dt*1e3:7.3f} ms   {tflops:7.1f} TFLOP/s")
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
    for name, M, K, N in SHAPES:
        print(f"\n{name} :: M={M} K={K} N={N}")
        a = (torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
        b = (torch.randn(K, N, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
        cublas_state = prep_cublas_fp8(a, b)
        try:
            dg_state = prep_deepgemm_fp8(a, b, dg)
        except Exception as e:
            print(f"    DeepGEMM prep FAILED: {e}")
            bench("cuBLAS FP8", run_cublas_fp8, cublas_state, M, K, N)
            continue
        cublas_tf = bench("cuBLAS FP8", run_cublas_fp8, cublas_state, M, K, N)
        try:
            dg_tf = bench("DeepGEMM FP8", run_deepgemm_fp8, dg_state, M, K, N)
            print(f"    → speedup DG/cuBLAS: {dg_tf/cublas_tf:.3f}×  ({(dg_tf/cublas_tf - 1)*100:+.1f}%)")
        except Exception as e:
            print(f"    DeepGEMM call FAILED: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
