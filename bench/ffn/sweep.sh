#!/bin/bash
# bench/ffn/sweep.sh
#
# FFN ablation: 8 configs on Qwen3-14B 1 node × 4 GPU, seq=4096, FA3 attention.
#   {MBS=1, MBS=2} × {BF16, FP8} × {Megatron native SwiGLU, Liger Triton SwiGLU}
#
# Tests:
#   1. Does FP8 help at small M (TP=4 + MBS=1 → M=1024)?  → NO (cuBLAS FP8 -4%)
#   2. Does FP8 help at larger M (MBS=2 → M=2048)?        → YES (cuBLAS FP8 +35%)
#   3. Does Liger Triton SwiGLU beat Megatron's native?   → marginal / no
#
# Usage:
#   bash bench/ffn/sweep.sh [--dry-run] [--only NAME]
#
# Configurable via env vars (defaults preserve Daint lp160):
#   GIPFEL_ACCOUNT / GIPFEL_PARTITION / GIPFEL_WORKDIR / GIPFEL_TIME
#   MODEL (default qwen3-14b)
#   NODES (default 1)
#   ITERS (default 15)
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

# Tuple: name : MBS : FP8 : SwiGLU_kernel
CONFIGS=(
    "mbs1-bf16-megatron:1:0:"
    "mbs1-bf16-liger:1:0:liger"
    "mbs1-fp8-megatron:1:1:"
    "mbs1-fp8-liger:1:1:liger"
    "mbs2-bf16-megatron:2:0:"
    "mbs2-bf16-liger:2:0:liger"
    "mbs2-fp8-megatron:2:1:"
    "mbs2-fp8-liger:2:1:liger"
)

unset GIPFEL_NO_MASTER_WEIGHTS GIPFEL_RECOMPUTE GIPFEL_EXP_AVG_DTYPE \
      GIPFEL_EXP_AVG_SQ_DTYPE GIPFEL_NO_OVERLAP_PG GIPFEL_NO_OVERLAP_GR \
      GIPFEL_NUM_LAYERS GIPFEL_EP GIPFEL_CP GIPFEL_GPUS_PER_NODE \
      GIPFEL_ZERO GIPFEL_ATTN_KERNEL GIPFEL_TIMING

export GIPFEL_ACCOUNT=${GIPFEL_ACCOUNT:-lp160}
export GIPFEL_PARTITION=${GIPFEL_PARTITION:-normal}
export GIPFEL_WORKDIR=${GIPFEL_WORKDIR:-/users/$USER/gipfelsturm}
export GIPFEL_TIME=${GIPFEL_TIME:-00:25:00}
export GIPFEL_SEQ_LEN=4096
export GIPFEL_TP=4
export GIPFEL_PP=1
export GIPFEL_USE_FA3=1
export GIPFEL_ATTN_BACKEND=flash

MODEL=${MODEL:-qwen3-14b}
NODES=${NODES:-1}
ITERS=${ITERS:-15}

total=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name mbs fp8 swiglu_kernel <<< "$cfg"
    if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
        continue
    fi
    export GIPFEL_MBS=$mbs
    export GIPFEL_FP8=$fp8
    if [ -n "$swiglu_kernel" ]; then
        export GIPFEL_SWIGLU_KERNEL=$swiglu_kernel
    else
        unset GIPFEL_SWIGLU_KERNEL
    fi
    total=$((total + 1))
    if [ "$DRY_RUN" = "1" ]; then
        echo "[$total] WOULD SUBMIT: $name"
    else
        echo "[$total] submit: $name"
        ./launch-mp.sh throughput "$MODEL" "$ITERS" "$NODES" 2>&1 | tail -1
    fi
done

echo
echo "Total jobs: $total"
