#!/bin/bash
# Triton attention kernel sweep over seq_len, Qwen3-14B 1n4g, stock TE path.
# Submits 6 jobs (seq 512 / 1024 / 2048 / 4096 / 8192 / 16384) with the
# Triton kernel hooked into TEDotProductAttention via patch 0003.
set -euo pipefail

unset GIPFEL_USE_FA3 GIPFEL_FP8 GIPFEL_NO_MASTER_WEIGHTS GIPFEL_RECOMPUTE \
      GIPFEL_EXP_AVG_DTYPE GIPFEL_EXP_AVG_SQ_DTYPE \
      GIPFEL_NO_OVERLAP_PG GIPFEL_NO_OVERLAP_GR \
      GIPFEL_MBS GIPFEL_NUM_LAYERS GIPFEL_EP GIPFEL_CP \
      GIPFEL_GPUS_PER_NODE GIPFEL_ATTN_BACKEND GIPFEL_ZERO

export GIPFEL_ACCOUNT=lp160
export GIPFEL_PARTITION=normal
export GIPFEL_WORKDIR=/users/$USER/gipfelsturm
export GIPFEL_TIME=00:25:00
export GIPFEL_TP=4
export GIPFEL_PP=1

KERNEL=${1:-triton}  # triton or tilelang
for seq in 512 1024 2048 4096 8192 16384; do
    export GIPFEL_SEQ_LEN=$seq
    export GIPFEL_ATTN_KERNEL=$KERNEL
    echo "=== $KERNEL seq=$seq ==="
    ./launch-mp.sh throughput qwen3-14b 15 1 2>&1 | tail -1
done
