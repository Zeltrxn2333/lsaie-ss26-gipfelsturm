#!/bin/bash
# bench/attention_backend_sweep.sh
#
# Phase-1 attention backend × seq_len sweep, narrowed scope:
#   - Qwen3-14B on 1 node × 4 GPU (TP=4 PP=1 CP=1 DP=1, world=4)
#   - Qwen3-8B  on 1 node × 1 GPU (TP=1 PP=1 CP=1 DP=1, world=1)
# Each tested at seq_len ∈ {512, 1024, 2048, 4096, 8192, 16384}
# across 4 native backends. 2 × 6 × 4 = 48 jobs total.
#
# Usage:
#   bash bench/attention_backend_sweep.sh [--dry-run]

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SEQ_LENS=(512 1024 2048 4096 8192 16384)
# Backend tuples: <name>:<--attention-backend value>:<use_fa3?>
BACKENDS=(
    "unfused:unfused:0"
    "fa2:flash:0"
    "fa3:flash:1"
    "cudnn:fused:0"
)

# Clear inherited overrides; we set what we test explicitly.
unset GIPFEL_FP8 GIPFEL_NO_MASTER_WEIGHTS GIPFEL_RECOMPUTE \
      GIPFEL_EXP_AVG_DTYPE GIPFEL_EXP_AVG_SQ_DTYPE \
      GIPFEL_NO_OVERLAP_PG GIPFEL_NO_OVERLAP_GR \
      GIPFEL_MBS GIPFEL_NUM_LAYERS GIPFEL_PP GIPFEL_EP GIPFEL_CP

export GIPFEL_ACCOUNT=lp160
export GIPFEL_PARTITION=normal
export GIPFEL_WORKDIR=/users/$USER/gipfelsturm
export GIPFEL_TIME=00:20:00

total=0
submit_for() {
    local model=$1
    local tp=$2
    local gpus_per_node=$3
    local nodes=1
    for seq in "${SEQ_LENS[@]}"; do
        for be in "${BACKENDS[@]}"; do
            IFS=':' read -r name flag fa3 <<< "$be"
            export GIPFEL_TP=$tp
            export GIPFEL_GPUS_PER_NODE=$gpus_per_node
            export GIPFEL_SEQ_LEN=$seq
            export GIPFEL_ATTN_BACKEND=$flag
            export GIPFEL_USE_FA3=$fa3
            total=$((total + 1))
            tag="${model} world=$((nodes * gpus_per_node)) tp=${tp} seq=${seq} backend=${name}"
            if [ "$DRY_RUN" = "1" ]; then
                echo "[$total] WOULD SUBMIT: $tag"
            else
                echo "[$total] submit: $tag"
                ./launch-mp.sh throughput "$model" 15 "$nodes" 2>&1 | tail -1
            fi
        done
    done
}

# Qwen3-14B on 1 node × 4 GPU, TP=4
submit_for qwen3-14b 4 4
# Qwen3-8B on 1 node × 1 GPU, TP=1
submit_for qwen3-8b 1 1

echo
echo "Total jobs: $total"
