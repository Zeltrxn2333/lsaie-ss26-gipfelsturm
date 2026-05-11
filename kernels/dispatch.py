"""Dispatch wrapper for custom attention kernels.

Megatron's local DotProductAttention.forward sees Q/K/V in Megatron-native layout:
    Q : [sq, b, np, hn]
    K : [sk, b, ng, hn]
    V : [sk, b, ng, hn]
and returns:
    out : [sq, b, np, hn]  (later reshaped to [sq, b, hp])

For GQA (np > ng), KV must be repeated along the head dim to match Q.

This module exports a single `run_attention(q, k, v, causal, softmax_scale)` entry point
that selects the right kernel based on `GIPFEL_ATTN_KERNEL` env var.
"""

from __future__ import annotations

import os

import torch


def _repeat_kv(k: torch.Tensor, v: torch.Tensor, n_q: int, n_kv: int):
    if n_q == n_kv:
        return k, v
    repeat = n_q // n_kv
    return k.repeat_interleave(repeat, dim=2), v.repeat_interleave(repeat, dim=2)


def _bshd_to_bhsd(t: torch.Tensor) -> torch.Tensor:
    """[sq, b, h, d] -> [b, h, sq, d] (contiguous)."""
    return t.permute(1, 2, 0, 3).contiguous()


def _bhsd_to_bshd(t: torch.Tensor) -> torch.Tensor:
    """[b, h, sq, d] -> [sq, b, h, d] (contiguous)."""
    return t.permute(2, 0, 1, 3).contiguous()


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """Dispatch to the selected attention kernel.

    Args:
        q: [sq, b, np, hn]
        k: [sk, b, ng, hn]
        v: [sk, b, ng, hn]
        causal: causal mask
        softmax_scale: 1/sqrt(d) typically

    Returns:
        out: [sq, b, np, hn]
    """
    kernel = os.environ.get("GIPFEL_ATTN_KERNEL", "").lower()
    if not kernel:
        raise RuntimeError(
            "kernels.dispatch.run_attention called without GIPFEL_ATTN_KERNEL set"
        )

    np_, ng = q.size(2), k.size(2)
    k, v = _repeat_kv(k, v, np_, ng)
    q_bhsd = _bshd_to_bhsd(q)
    k_bhsd = _bshd_to_bhsd(k)
    v_bhsd = _bshd_to_bhsd(v)

    if kernel == "triton":
        from .triton_attention import triton_attention

        out_bhsd = triton_attention(q_bhsd, k_bhsd, v_bhsd, causal, softmax_scale)
    elif kernel == "tilelang":
        from .tilelang_attention import tilelang_attention

        out_bhsd = tilelang_attention(q_bhsd, k_bhsd, v_bhsd, causal, softmax_scale)
    else:
        raise ValueError(
            f"GIPFEL_ATTN_KERNEL={kernel!r} not recognized (use 'triton' or 'tilelang')"
        )

    return _bhsd_to_bshd(out_bhsd)
