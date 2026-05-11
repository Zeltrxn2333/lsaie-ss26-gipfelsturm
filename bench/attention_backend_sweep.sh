#!/bin/bash
# bench/attention_backend_sweep.sh
#
# Submits the full Phase-1 sweep:
#   2 models × 6 seq_lens × 4 native backends × 3 CP values = 144 jobs.
#
# Each (model, seq_len, backend, CP) combination is one sbatch job at
# the corresponding node count (NODES = CP, since TP=4 PP=1 DP=1).
#
# Usage:
#   bash bench/attention_backend_sweep.sh [--dry-run]
#
# Run on a Daint login node from /users/$USER/gipfelsturm.
# Assumes FA3 venv is at /iopsstor/scratch/cscs/$USER/venvs/fa3.

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

MODELS=(qwen3-8b qwen3-14b)
SEQ_LENS=(512 1024 2048 4096 8192 16384)
CPS=(1 2 4)
# Backend tuples: <name>:<--attention-backend value>:<use_fa3?>
BACKENDS=(
    "unfused:unfused:0"
    "fa2:flash:0"
    "fa3:flash:1"
    "cudnn:fused:0"
)

# Clear any prior overrides that could pollute (we only set what we test).
unset GIPFEL_FP8 GIPFEL_NO_MASTER_WEIGHTS GIPFEL_RECOMPUTE \
      GIPFEL_EXP_AVG_DTYPE GIPFEL_EXP_AVG_SQ_DTYPE \
      GIPFEL_NO_OVERLAP_PG GIPFEL_NO_OVERLAP_GR \
      GIPFEL_MBS GIPFEL_NUM_LAYERS GIPFEL_TP GIPFEL_PP GIPFEL_EP

export GIPFEL_ACCOUNT=lp160
export GIPFEL_PARTITION=normal
export GIPFEL_WORKDIR=/users/$USER/gipfelsturm
export GIPFEL_TIME=00:20:00

total=0
for model in "${MODELS[@]}"; do
    for seq in "${SEQ_LENS[@]}"; do
        for cp in "${CPS[@]}"; do
            # Skip CP combinations that can't divide num_heads on Qwen3-14B (40H).
            # TP=4, CP*TP must divide 40 → CP ∈ {1,2}. CP=4 needs heads%16=0 → 40%16≠0.
            # For qwen3-8b (32H), CP*TP must divide 32 → CP ∈ {1,2,4}. ✓
            if [ "$model" = "qwen3-14b" ] && [ "$cp" = "4" ]; then
                continue
            fi
            # CP requires seq_len % (CP*2) == 0; safe for all our seq_lens since powers of 2.
            for be in "${BACKENDS[@]}"; do
                IFS=':' read -r name flag fa3 <<< "$be"
                nodes=$cp
                export GIPFEL_SEQ_LEN=$seq
                export GIPFEL_CP=$cp
                export GIPFEL_ATTN_BACKEND=$flag
                export GIPFEL_USE_FA3=$fa3
                total=$((total + 1))
                tag="${model} seq=${seq} cp=${cp} backend=${name} nodes=${nodes}"
                if [ "$DRY_RUN" = "1" ]; then
                    echo "[$total] WOULD SUBMIT: $tag"
                else
                    echo "[$total] submit: $tag"
                    ./launch-mp.sh throughput "$model" 15 "$nodes" 2>&1 | tail -1
                fi
            done
        done
    done
done

echo
echo "Total jobs: $total"
