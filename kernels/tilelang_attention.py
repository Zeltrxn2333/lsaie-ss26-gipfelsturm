"""TileLang attention wrapper.

Forward via TileLang's flash-attention example kernel; backward via PyTorch's math SDPA
(reference) as TileLang core doesn't ship a backward kernel for this layout. The
backward is **not** TileLang-accelerated, so end-to-end numbers should be read as
"TileLang fwd, PyTorch math bwd".

Inputs: [B, H, N, D] contiguous bf16/fp16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Lazy-initialised, JIT-compiled per shape.
_tilelang_kernels: dict = {}


def _get_tilelang_fwd(batch: int, heads: int, seq_q: int, seq_kv: int,
                     dim_qk: int, dim_v: int, causal: bool, dtype):
    """Return a JIT-compiled TileLang forward kernel for this shape."""
    key = (batch, heads, seq_q, seq_kv, dim_qk, dim_v, causal, dtype)
    if key in _tilelang_kernels:
        return _tilelang_kernels[key]

    import tilelang
    import tilelang.language as T

    BLOCK_M, BLOCK_N = 128, 128
    scale = (1.0 / (dim_qk ** 0.5)) * 1.44269504  # 1/log(2)
    type_str = "bfloat16" if dtype == torch.bfloat16 else "float16"

    @tilelang.jit
    def main_kernel(
        Q: T.Tensor((batch, heads, seq_q, dim_qk), type_str),
        K: T.Tensor((batch, heads, seq_kv, dim_qk), type_str),
        V: T.Tensor((batch, heads, seq_kv, dim_v), type_str),
        Output: T.Tensor((batch, heads, seq_q, dim_v), type_str),
    ):
        with T.Kernel(T.ceildiv(seq_q, BLOCK_M), heads, batch, threads=128) as (bx, by, bz):
            Q_shared = T.alloc_shared((BLOCK_M, dim_qk), type_str)
            K_shared = T.alloc_shared((BLOCK_N, dim_qk), type_str)
            V_shared = T.alloc_shared((BLOCK_N, dim_v), type_str)
            acc_o = T.alloc_fragment((BLOCK_M, dim_v), "float32")
            acc_s = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            scores_max = T.alloc_fragment((BLOCK_M,), "float32")
            scores_max_prev = T.alloc_fragment((BLOCK_M,), "float32")
            scores_scale = T.alloc_fragment((BLOCK_M,), "float32")
            scores_sum = T.alloc_fragment((BLOCK_M,), "float32")
            logsum = T.alloc_fragment((BLOCK_M,), "float32")
            T.copy(Q[bz, by, bx * BLOCK_M : (bx + 1) * BLOCK_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(acc_s.dtype))
            loop_range = T.ceildiv((bx + 1) * BLOCK_M, BLOCK_N) if causal else T.ceildiv(seq_kv, BLOCK_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, by, k * BLOCK_N : (k + 1) * BLOCK_N, :], K_shared)
                T.copy(V[bz, by, k * BLOCK_N : (k + 1) * BLOCK_N, :], V_shared)
                if causal:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        acc_s[i, j] = T.if_then_else(bx * BLOCK_M + i >= k * BLOCK_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(acc_s.dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(BLOCK_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(BLOCK_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.gemm(acc_s, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            for i, j in T.Parallel(BLOCK_M, dim_v):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, by, bx * BLOCK_M : (bx + 1) * BLOCK_M, :])

    _tilelang_kernels[key] = main_kernel
    return main_kernel


class _TileLangAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # q,k,v: [B, H, N, D] (GQA already expanded upstream)
        B, H, N, D = q.shape
        kernel = _get_tilelang_fwd(B, H, N, N, D, D, causal, q.dtype)
        o = torch.empty_like(q)
        kernel(q, k, v, o)
        ctx.save_for_backward(q, k, v, o)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        # PyTorch math SDPA backward as reference (TileLang doesn't ship backward for this shape).
        q, k, v, _ = ctx.saved_tensors
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        with torch.enable_grad():
            with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=ctx.causal, scale=ctx.sm_scale)
            dq, dk, dv = torch.autograd.grad(out_ref, (q, k, v), do)
        return dq, dk, dv, None, None


def tilelang_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       causal: bool, sm_scale: float) -> torch.Tensor:
    return _TileLangAttention.apply(q, k, v, causal, sm_scale)
