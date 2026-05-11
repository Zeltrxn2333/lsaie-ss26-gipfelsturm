"""Custom attention kernels for Megatron-LM dispatch via GIPFEL_ATTN_KERNEL env var.

When `--attention-backend local` is set on Megatron and `GIPFEL_ATTN_KERNEL` is one of
`triton` / `tilelang`, the patched `DotProductAttention.forward` calls a kernel from this
package instead of running its built-in QK^T / softmax / *V loop.

All kernels share the same wrapper signature:

    out = run(q, k, v, causal, softmax_scale)

with input tensors in `[sq, b, np, hn]` / `[sk, b, ng, hn]` layout (Megatron's BSHD) and
output in `[sq, b, np, hn]`. The wrapper handles GQA KV repetition before the kernel call.
"""

from .dispatch import run_attention  # noqa: F401
