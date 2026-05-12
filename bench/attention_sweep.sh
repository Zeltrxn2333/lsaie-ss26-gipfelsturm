#!/bin/bash
# bench/attention_sweep.sh
#
# Full 6-backend × 6-seq_len attention sweep on Qwen3-14B (1 node × 4 GPU,
# TP=4 PP=1 DP=1, MBS=1, stock TE path). 36 jobs total.
#
# Backends:
#   unfused  — Standard math SDPA           (--attention-backend unfused)
#   flash    — FlashAttention-2 (container) (--attention-backend flash, no FA3 venv)
#   fused    — cuDNN FusedAttention         (--attention-backend fused)
#   flash+fa3 — FlashAttention-3 (built)    (--attention-backend flash + GIPFEL_USE_FA3=1)
#   triton   — OpenAI tutorial Triton kernel via GIPFEL_ATTN_KERNEL=triton
#   tilelang — TileLang flash-attn kernel via GIPFEL_ATTN_KERNEL=tilelang
#
# All custom kernels (triton/tilelang) plug into TEDotProductAttention.forward
# via patches/0003-custom-attention-kernels.patch, so LayerNorm / sequence-parallel
# / fused softmax are identical across runs — only the attention kernel changes.
#
# Usage:
#   bash bench/attention_sweep.sh [--dry-run] [--only backend]
#
# Examples:
#   bash bench/attention_sweep.sh                          # submit all 36 jobs
#   bash bench/attention_sweep.sh --dry-run                # list the 36 jobs
#   bash bench/attention_sweep.sh --only triton            # submit just the 6 Triton jobs

set -euo pipefail

DRY_RUN=0
ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SEQ_LENS=(512 1024 2048 4096 8192 16384)
# Backend tuples: name : attn_backend_flag : use_fa3 : custom_kernel
BACKENDS=(
    "unfused:unfused:0:"
    "flash:flash:0:"
    "fused:fused:0:"
    "flash+fa3:flash:1:"
    "triton::0:triton"
    "tilelang::0:tilelang"
)

# Clear any inherited overrides — set what we test explicitly.
unset GIPFEL_FP8 GIPFEL_NO_MASTER_WEIGHTS GIPFEL_RECOMPUTE \
      GIPFEL_EXP_AVG_DTYPE GIPFEL_EXP_AVG_SQ_DTYPE \
      GIPFEL_NO_OVERLAP_PG GIPFEL_NO_OVERLAP_GR \
      GIPFEL_MBS GIPFEL_NUM_LAYERS GIPFEL_EP GIPFEL_CP \
      GIPFEL_GPUS_PER_NODE GIPFEL_ZERO GIPFEL_TP GIPFEL_PP

export GIPFEL_ACCOUNT=lp160
export GIPFEL_PARTITION=normal
export GIPFEL_WORKDIR=/users/$USER/gipfelsturm
export GIPFEL_TIME=00:25:00

total=0
for seq in "${SEQ_LENS[@]}"; do
    for be in "${BACKENDS[@]}"; do
        IFS=':' read -r name attn_be fa3 kernel <<< "$be"
        if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
            continue
        fi
        export GIPFEL_SEQ_LEN=$seq
        export GIPFEL_USE_FA3=$fa3
        if [ -n "$kernel" ]; then
            export GIPFEL_ATTN_KERNEL=$kernel
            unset GIPFEL_ATTN_BACKEND
        else
            export GIPFEL_ATTN_BACKEND=$attn_be
            unset GIPFEL_ATTN_KERNEL
        fi
        total=$((total + 1))
        tag="seq=${seq} backend=${name}"
        if [ "$DRY_RUN" = "1" ]; then
            echo "[$total] WOULD SUBMIT: $tag"
        else
            echo "[$total] submit: $tag"
            ./launch-mp.sh throughput qwen3-14b 15 1 2>&1 | tail -1
        fi
    done
done

echo
echo "Total jobs: $total"
