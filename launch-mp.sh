#!/bin/bash
#
# launch-mp.sh — personal launcher adding model-parallel support (TP, PP)
# on top of launch.sh. Adds 32b and 140b sizes for Challenge 2 tiers.
#
# Usage: ./launch-mp.sh <mode> <model_size> [steps] [nodes]
#
# Modes:   throughput | train    (same semantics as launch.sh)
# Sizes:   125m | 350m | 760m | 1.5b | 3b | 8b | 32b | 140b
# Nodes:   default from tier (140b needs >=4); max 8
#
# Default MP per tier (README Challenge 2):
#   8b   → TP=1, PP=1   (single-GPU tier)
#   32b  → TP=4, PP=1   (single-node tier; intra-node TP over NVLink-C2C)
#   140b → TP=4, PP=4   (multi-node tier; PP across Slingshot-11)
# All defaults overridable via GIPFEL_TP / GIPFEL_PP.
#
# Examples:
#   ./launch-mp.sh throughput 32b 50 1            # single-node TP=4
#   ./launch-mp.sh throughput 140b 50 4           # multi-node TP=4 PP=4
#   GIPFEL_TP=4 ./launch-mp.sh throughput 8b 50 1 # 8b with forced TP=4

set -euo pipefail

MODE=${1:?Usage: ./launch-mp.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch-mp.sh <mode> <model_size> [steps] [nodes]}

################ Cluster / account parameterization ################
GIPFEL_ACCOUNT=${GIPFEL_ACCOUNT:-infra01}
GIPFEL_PARTITION=${GIPFEL_PARTITION:-}
GIPFEL_WORKDIR=${GIPFEL_WORKDIR:-/users/schlag/gipfelsturm}
GIPFEL_DATA_PREFIX=${GIPFEL_DATA_PREFIX:-/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small}

################ Memory / precision knobs (opt-in) ################
# GIPFEL_CPU_OFFLOAD=N   — N in [0, 1] = offload fraction. 1.0 full, 0.5 half, 0 off.
# GIPFEL_RECOMPUTE=1     — --recompute-activations (selective recompute in attention)
# GIPFEL_RECOMPUTE=full  — --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 (full recompute)
# GIPFEL_FP8=1           — --fp8-format hybrid --fp8-recipe delayed --fp8-param-gather (Hopper-safe)
GIPFEL_CPU_OFFLOAD=${GIPFEL_CPU_OFFLOAD:-0}
GIPFEL_RECOMPUTE=${GIPFEL_RECOMPUTE:-0}
GIPFEL_FP8=${GIPFEL_FP8:-0}
# GIPFEL_MAIN_PARAMS_FP16=1 — store main (master) params in fp16 instead of fp32.
GIPFEL_MAIN_PARAMS_FP16=${GIPFEL_MAIN_PARAMS_FP16:-0}
# GIPFEL_EXP_AVG_DTYPE / GIPFEL_EXP_AVG_SQ_DTYPE — Adam m/v dtype {fp32|fp16|bf16|fp8}
#   Default bf16 when MP>1 (saves 16 GB/rank vs fp32). fp8 saves another 16 GB on 32B.
GIPFEL_EXP_AVG_DTYPE=${GIPFEL_EXP_AVG_DTYPE:-bf16}
GIPFEL_EXP_AVG_SQ_DTYPE=${GIPFEL_EXP_AVG_SQ_DTYPE:-bf16}
# GIPFEL_NCCL_TUNE=1 — disable NVLS SHARP and tighten CUDA alloc split (~3-5 GB HBM back).
GIPFEL_NCCL_TUNE=${GIPFEL_NCCL_TUNE:-0}
# GIPFEL_OPTIMIZER=muon|sgd — override Adam with a lower-state optimizer.
GIPFEL_OPTIMIZER=${GIPFEL_OPTIMIZER:-adam}
# GIPFEL_NO_OVERLAP_PG=1 — disable --overlap-param-gather. With DP=1 there is
#   nothing to overlap anyway, but the double-buffer still eats ~16 GB/rank.
GIPFEL_NO_OVERLAP_PG=${GIPFEL_NO_OVERLAP_PG:-0}
# GIPFEL_NO_OVERLAP_GR=1 — disable --overlap-grad-reduce (frees grad double-buffer ~8-16 GB)
GIPFEL_NO_OVERLAP_GR=${GIPFEL_NO_OVERLAP_GR:-0}
# GIPFEL_DDP_BUCKET_SIZE — bytes. Default Megatron=40M. Smaller = less NCCL peak.
GIPFEL_DDP_BUCKET_SIZE=${GIPFEL_DDP_BUCKET_SIZE:-0}
# GIPFEL_NO_MASTER_WEIGHTS=1 — requires patches/0002-no-master-weights-option.patch.
#   Skips TE FusedAdam's master weight tensor, saving ~16 GB/rank at cost of
#   bf16-precision Adam updates (fine for throughput benchmarks).
GIPFEL_NO_MASTER_WEIGHTS=${GIPFEL_NO_MASTER_WEIGHTS:-0}
# GIPFEL_NUM_LAYERS — override NUM_LAYERS for the selected model size. Useful for
#   shrinking 32B to a ~28B variant (56 layers) that fits single-node.
GIPFEL_NUM_LAYERS=${GIPFEL_NUM_LAYERS:-0}
# GIPFEL_MBS — override the per-model-size default micro-batch-size.
GIPFEL_MBS=${GIPFEL_MBS:-0}
# GIPFEL_TP_COMM_OVERLAP=1 — emit --tp-comm-overlap (TE userbuffers TP comm/compute overlap).
GIPFEL_TP_COMM_OVERLAP=${GIPFEL_TP_COMM_OVERLAP:-0}
# GIPFEL_TIMING=2 — emit --timing-log-level 2 + --barrier-with-L1-time for per-phase breakdown
GIPFEL_TIMING=${GIPFEL_TIMING:-0}
# GIPFEL_USE_FA3=1 — prepend FA3 venv to PYTHONPATH so TE auto-uses flash-attn-3
GIPFEL_USE_FA3=${GIPFEL_USE_FA3:-0}
# GIPFEL_ATTN_BACKEND — explicit --attention-backend value (flash/fused/unfused/auto/local).
#   If unset, FA3 toggle picks flash; otherwise auto.
GIPFEL_ATTN_BACKEND=${GIPFEL_ATTN_BACKEND:-}
# GIPFEL_CP — context-parallel-size. Default 1. Megatron shards seq dim across CP ranks.
#   Requires world_size = TP × PP × CP × DP and num_heads divisible by (TP × CP).
GIPFEL_CP=${GIPFEL_CP:-1}
# GIPFEL_ATTN_KERNEL — custom attention kernel. When set, forces
#   --attention-backend local and exports the var into the job env so the
#   patched DotProductAttention.forward dispatches to kernels/ package.
#   Values: triton, tilelang. Default empty (no override).
GIPFEL_ATTN_KERNEL=${GIPFEL_ATTN_KERNEL:-}
# GIPFEL_ZERO — ZeRO sharding stage via Megatron FSDP. Default 0 (ZeRO-1 via
#   --use-distributed-optimizer, the current behavior).
#     0 = --use-distributed-optimizer (MCore dist-opt; ZeRO-1 equivalent)
#     1 = --use-megatron-fsdp --data-parallel-sharding-strategy optim
#     2 = --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads (ZeRO-2)
#     3 = --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params (ZeRO-3 / full FSDP)
#   When non-zero, --use-distributed-optimizer / overlap flags are dropped
#   (megatron-fsdp manages comm internally).
GIPFEL_ZERO=${GIPFEL_ZERO:-0}
# GIPFEL_MEM — SLURM --mem value (MB). Default 460000; Daint has ~480 GB per node.
GIPFEL_MEM=${GIPFEL_MEM:-460000}
# GIPFEL_CPU_OFFLOADING_LAYERS=N — TE activation offload for N layers (distinct
#   from --optimizer-cpu-offload). Offloads layer activations to CPU during fwd
#   and fetches for bwd. Mutually exclusive with recompute.
GIPFEL_CPU_OFFLOADING_LAYERS=${GIPFEL_CPU_OFFLOADING_LAYERS:-0}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        TIME=${GIPFEL_TIME:-00:30:00}
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch-mp.sh train <model_size> <steps> [nodes]}
        TIME=${GIPFEL_TIME:-02:30:00}
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config ################
# 32b shape: exact Qwen 2.5-32B (64 layers, h=5120, ffn=27648, heads=40, kv=8, head_dim=128).
# 140b shape: Qwen-style dense extrapolation (no published Qwen dense at 140B).
#   80 layers, h=12288, ffn=32768, heads=96, kv=8, head_dim=128.
#   Satisfies TP=4 and PP=4 divisibility: h/4, ffn/4, heads/4, kv_heads/4, layers/4 all integers.
#   Total params ~145B (80 × 1.8B/layer + 1.2B embed with GPT-2 vocab).
NUM_EXPERTS=0   # 0 = dense; MoE cases set this >0
MOE_TOPK=0
DEFAULT_EP=1
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;   FFN=2048;   HEADS=12; KV_HEADS=4
        MBS=16; DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    350m)
        NUM_LAYERS=24;  HIDDEN=1024;  FFN=2816;   HEADS=16; KV_HEADS=4
        MBS=8;  DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    760m)
        NUM_LAYERS=24;  HIDDEN=1536;  FFN=4096;   HEADS=16; KV_HEADS=4
        MBS=4;  DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    1.5b)
        NUM_LAYERS=48;  HIDDEN=1600;  FFN=4352;   HEADS=20; KV_HEADS=4
        MBS=4;  DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    3b)
        NUM_LAYERS=32;  HIDDEN=3072;  FFN=8192;   HEADS=24; KV_HEADS=8
        MBS=4;  DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    8b)
        NUM_LAYERS=32;  HIDDEN=4096;  FFN=14336;  HEADS=32; KV_HEADS=8
        MBS=2;  DEFAULT_NODES=4; DEFAULT_TP=1; DEFAULT_PP=1
        ;;
    qwen3-8b)
        # Real Qwen3-8B: 36 layers, h=4096, ffn=12288, 32 Q heads, 8 KV heads.
        NUM_LAYERS=36; HIDDEN=4096; FFN=12288; HEADS=32; KV_HEADS=8
        MBS=1; DEFAULT_NODES=1; DEFAULT_TP=4; DEFAULT_PP=1
        ;;
    qwen3-14b)
        # Real Qwen3-14B: 40 layers, h=5120, ffn=17408, 40 Q heads, 8 KV heads.
        NUM_LAYERS=40; HIDDEN=5120; FFN=17408; HEADS=40; KV_HEADS=8
        MBS=1; DEFAULT_NODES=1; DEFAULT_TP=4; DEFAULT_PP=1
        ;;
    qwen3-32b)
        # Real Qwen3-32B: 64 layers, h=5120, ffn=25600, 64 Q heads, 8 KV heads.
        NUM_LAYERS=64; HIDDEN=5120; FFN=25600; HEADS=64; KV_HEADS=8
        MBS=1; DEFAULT_NODES=2; DEFAULT_TP=4; DEFAULT_PP=1
        ;;
    32b)
        # Qwen 2.5-32B: 64 layers, h=5120, ffn=27648, 40 heads, 8 kv.
        NUM_LAYERS=64;  HIDDEN=5120;  FFN=27648;  HEADS=40; KV_HEADS=8
        MBS=1;  DEFAULT_NODES=1; DEFAULT_TP=4; DEFAULT_PP=1
        ;;
    llama3-70b)
        # Real Llama 3.1-70B: 80 layers, h=8192, ffn=28672, 64 Q heads, 8 KV heads.
        # 4 nodes natural fit: TP=4 within node, PP=2 across nodes for memory, DP=2.
        NUM_LAYERS=80; HIDDEN=8192; FFN=28672; HEADS=64; KV_HEADS=8
        MBS=1; DEFAULT_NODES=4; DEFAULT_TP=4; DEFAULT_PP=2
        ;;
    qwen2.5-72b)
        # Real Qwen 2.5-72B: 80 layers, h=8192, ffn=29568, 64 Q heads, 8 KV heads.
        NUM_LAYERS=80; HIDDEN=8192; FFN=29568; HEADS=64; KV_HEADS=8
        MBS=1; DEFAULT_NODES=4; DEFAULT_TP=4; DEFAULT_PP=2
        ;;
    140b)
        NUM_LAYERS=80;  HIDDEN=12288; FFN=32768;  HEADS=96; KV_HEADS=8
        MBS=1;  DEFAULT_NODES=4; DEFAULT_TP=4; DEFAULT_PP=4
        ;;
    mixtral-8x7b)
        # Real Mixtral 8x7B: 32L/h=4096/ffn=14336/32H/8KV + 8 experts top-2.
        # 47B total / 13B active. Standard MoE, no shared expert.
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        NUM_EXPERTS=8; MOE_TOPK=2
        MBS=1; DEFAULT_NODES=2; DEFAULT_TP=4; DEFAULT_PP=1; DEFAULT_EP=2
        ;;
    mixtral-8x22b)
        # Real Mixtral 8x22B: 56L/h=6144/ffn=16384/48H/8KV + 8 experts top-2.
        # 141B total / 39B active. Needs >=8 nodes for stock-default fit.
        NUM_LAYERS=56; HIDDEN=6144; FFN=16384; HEADS=48; KV_HEADS=8
        NUM_EXPERTS=8; MOE_TOPK=2
        MBS=1; DEFAULT_NODES=8; DEFAULT_TP=4; DEFAULT_PP=1; DEFAULT_EP=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m|350m|760m|1.5b|3b|8b|qwen3-8b|qwen3-14b|qwen3-32b|32b|llama3-70b|qwen2.5-72b|140b|mixtral-8x7b|mixtral-8x22b"
        exit 1
        ;;
esac

NODES=${4:-$DEFAULT_NODES}
TP=${GIPFEL_TP:-$DEFAULT_TP}
PP=${GIPFEL_PP:-$DEFAULT_PP}
EP=${GIPFEL_EP:-$DEFAULT_EP}
CP=${GIPFEL_CP:-1}

# Optional layer-count override (e.g. reduce 32B from 64->56 for single-node fit).
if [ "$GIPFEL_NUM_LAYERS" != "0" ]; then
    NUM_LAYERS=$GIPFEL_NUM_LAYERS
fi
# Optional MBS override.
if [ "$GIPFEL_MBS" != "0" ]; then
    MBS=$GIPFEL_MBS
fi
GPUS_PER_NODE=${GIPFEL_GPUS_PER_NODE:-4}
WORLD_SIZE=$((NODES * GPUS_PER_NODE))
MP_SIZE=$((TP * PP))

# Sanity checks
if (( WORLD_SIZE < MP_SIZE )); then
    echo "ERROR: world_size=$WORLD_SIZE < TP*PP=$MP_SIZE (need nodes >= $((MP_SIZE / GPUS_PER_NODE)))" >&2
    exit 1
fi
if (( NUM_LAYERS % PP != 0 )); then
    echo "ERROR: NUM_LAYERS=$NUM_LAYERS not divisible by PP=$PP" >&2
    exit 1
fi
if (( HIDDEN % TP != 0 )) || (( HEADS % TP != 0 )) || (( KV_HEADS % TP != 0 )); then
    echo "ERROR: HIDDEN/HEADS/KV_HEADS must be divisible by TP=$TP (got $HIDDEN/$HEADS/$KV_HEADS)" >&2
    exit 1
fi
if (( NUM_EXPERTS > 0 )); then
    if (( NUM_EXPERTS % EP != 0 )); then
        echo "ERROR: NUM_EXPERTS=$NUM_EXPERTS not divisible by EP=$EP" >&2
        exit 1
    fi
    DP=$(( WORLD_SIZE / (TP * PP * EP) ))
    if (( DP < 1 )) || (( WORLD_SIZE != TP * PP * EP * DP )); then
        echo "ERROR: world=$WORLD_SIZE != TP($TP)*PP($PP)*EP($EP)*DP" >&2
        exit 1
    fi
fi
if (( CP > 1 )); then
    if (( HEADS % (TP * CP) != 0 )); then
        echo "ERROR: HEADS=$HEADS not divisible by TP*CP=$((TP*CP))" >&2
        exit 1
    fi
    EFFECTIVE_EP=$(( NUM_EXPERTS > 0 ? EP : 1 ))
    DP_AFTER_CP=$(( WORLD_SIZE / (TP * PP * EFFECTIVE_EP * CP) ))
    if (( DP_AFTER_CP < 1 )) || (( WORLD_SIZE != TP * PP * EFFECTIVE_EP * CP * DP_AFTER_CP )); then
        echo "ERROR: world=$WORLD_SIZE != TP($TP)*PP($PP)*EP($EFFECTIVE_EP)*CP($CP)*DP (must be exact)" >&2
        exit 1
    fi
fi

GBS=256
SEQ_LEN=${GIPFEL_SEQ_LEN:-4096}
if (( CP > 1 )) && (( SEQ_LEN % (CP * 2) != 0 )); then
    echo "ERROR: SEQ_LEN=$SEQ_LEN not divisible by CP*2=$((CP*2)) (Megatron CP requirement)" >&2
    exit 1
fi
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-tp${TP}pp${PP}"
if (( CP > 1 )); then
    JOB_NAME="${JOB_NAME}cp${CP}"
fi
if [ "$GIPFEL_ZERO" != "0" ]; then
    JOB_NAME="${JOB_NAME}-zero${GIPFEL_ZERO}"
fi
if [ -n "$GIPFEL_ATTN_KERNEL" ]; then
    JOB_NAME="${JOB_NAME}-${GIPFEL_ATTN_KERNEL}"
elif [ -n "$GIPFEL_ATTN_BACKEND" ]; then
    JOB_NAME="${JOB_NAME}-${GIPFEL_ATTN_BACKEND}"
    if [ "$GIPFEL_USE_FA3" = "1" ]; then
        JOB_NAME="${JOB_NAME}fa3"
    fi
fi
JOB_NAME="${JOB_NAME}-${TRAINING_STEPS}s-${NODES}n"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${GIPFEL_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=288
#SBATCH --mem=${GIPFEL_MEM}
#SBATCH --no-requeue
SBATCH_DIRECTIVES

if [ -n "$GIPFEL_PARTITION" ]; then
    echo "#SBATCH --partition=${GIPFEL_PARTITION}" >> "$SCRIPT"
fi

cat >> "$SCRIPT" << BODY_PARAM

echo "START TIME: \$(date)"
echo "MP config: TP=${TP} PP=${PP} (world=${WORLD_SIZE}) adam_m=${GIPFEL_EXP_AVG_DTYPE} adam_v=${GIPFEL_EXP_AVG_SQ_DTYPE} fp8=${GIPFEL_FP8} recompute=${GIPFEL_RECOMPUTE} optimizer=${GIPFEL_OPTIMIZER} nccl_tune=${GIPFEL_NCCL_TUNE}"

################ Configs ################
WORKDIR=${GIPFEL_WORKDIR}
DATA_PREFIX=${GIPFEL_DATA_PREFIX}
BODY_PARAM

if [ "$GIPFEL_NCCL_TUNE" = "1" ]; then
    cat >> "$SCRIPT" << 'NCCL_TUNE'
export NCCL_NVLS_ENABLE=0
export NCCL_BUFFSIZE=4194304
NCCL_TUNE
fi

# FA3 venv prepend (when GIPFEL_USE_FA3=1) — TE 2.11 auto-uses flash_attn_3 if present.
# NVTE_FUSED_ATTN=0 NVTE_FLASH_ATTN=1 forces TE flash-attn path (which dispatches to FA3 if installed,
# else FA2). Otherwise TE's auto heuristic on Hopper prefers cuDNN FusedAttention.
if [ "$GIPFEL_USE_FA3" = "1" ]; then
    cat >> "$SCRIPT" << FA3_EOF
export PYTHONPATH=/iopsstor/scratch/cscs/\$USER/venvs/fa3:\$PYTHONPATH
FA3_EOF
fi

# Megatron FSDP requires CUDA_DEVICE_MAX_CONNECTIONS > 1.
if [ "$GIPFEL_ZERO" != "0" ]; then
    echo "export CUDA_DEVICE_MAX_CONNECTIONS=32" >> "$SCRIPT"
fi

cat >> "$SCRIPT" << 'BODY'
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
BODY

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-tp${TP}pp${PP}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$WORKDIR:$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

if [ -n "$GIPFEL_ATTN_KERNEL" ]; then
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl local
)
else
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)
fi

SETUP

# Precision-aware-optimizer only works with adam + distributed optimizer in core_v0.16.1.
# Megatron FSDP (GIPFEL_ZERO != 0) is incompatible.
if [ "$GIPFEL_OPTIMIZER" = "adam" ] && [ "$GIPFEL_ZERO" = "0" ]; then
    cat >> "$SCRIPT" << 'PAO'
TRANSFORMER_ENGINE_ARGS+=(
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)
PAO
    if [ "$GIPFEL_NO_MASTER_WEIGHTS" = "1" ]; then
        echo 'TRANSFORMER_ENGINE_ARGS+=(--no-master-weights)' >> "$SCRIPT"
    fi
fi

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer ${GIPFEL_OPTIMIZER}
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << DISTRIBUTED

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
DISTRIBUTED

# Megatron FSDP path. When ZeRO != 0, replace dist-opt + overlap flags with FSDP flags.
if [ "$GIPFEL_ZERO" != "0" ]; then
    case "$GIPFEL_ZERO" in
        1) FSDP_STRATEGY=optim ;;
        2) FSDP_STRATEGY=optim_grads ;;
        3) FSDP_STRATEGY=optim_grads_params ;;
        *) echo "ERROR: GIPFEL_ZERO must be 0/1/2/3 (got $GIPFEL_ZERO)" >&2; exit 1 ;;
    esac
    echo "    --use-megatron-fsdp" >> "$SCRIPT"
    echo "    --data-parallel-sharding-strategy $FSDP_STRATEGY" >> "$SCRIPT"
    echo "    --ckpt-format fsdp_dtensor" >> "$SCRIPT"
# Muon asserts against --use-distributed-optimizer and overlap in core_v0.16.1.
elif [ "$GIPFEL_OPTIMIZER" != "muon" ] && [ "$GIPFEL_OPTIMIZER" != "dist_muon" ]; then
    echo "    --use-distributed-optimizer" >> "$SCRIPT"
    if [ "$GIPFEL_NO_OVERLAP_GR" != "1" ]; then
        echo "    --overlap-grad-reduce" >> "$SCRIPT"
    fi
    if [ "$GIPFEL_NO_OVERLAP_PG" != "1" ] && [ "$GIPFEL_NO_OVERLAP_GR" != "1" ]; then
        echo "    --overlap-param-gather" >> "$SCRIPT"
    fi
fi
if [ "$GIPFEL_DDP_BUCKET_SIZE" != "0" ]; then
    echo "    --ddp-bucket-size $GIPFEL_DDP_BUCKET_SIZE" >> "$SCRIPT"
fi
if [ "$GIPFEL_TP_COMM_OVERLAP" = "1" ]; then
    echo "    --tp-comm-overlap" >> "$SCRIPT"
fi
if [ "$GIPFEL_TIMING" != "0" ]; then
    echo "    --timing-log-level $GIPFEL_TIMING" >> "$SCRIPT"
fi
if [ -n "$GIPFEL_ATTN_KERNEL" ]; then
    # Custom Triton/TileLang kernel: route through Megatron's local DotProductAttention
    # which we patched (patches/0003-custom-attention-kernels.patch).
    echo "    --attention-backend local" >> "$SCRIPT"
    echo "    --transformer-impl local" >> "$SCRIPT"
elif [ -n "$GIPFEL_ATTN_BACKEND" ]; then
    echo "    --attention-backend $GIPFEL_ATTN_BACKEND" >> "$SCRIPT"
elif [ "$GIPFEL_USE_FA3" = "1" ]; then
    echo "    --attention-backend flash" >> "$SCRIPT"
fi

if (( TP > 1 )); then
    echo "    --sequence-parallel" >> "$SCRIPT"
fi
if (( NUM_EXPERTS > 0 )); then
    echo "    --expert-model-parallel-size $EP" >> "$SCRIPT"
fi
if (( CP > 1 )); then
    echo "    --context-parallel-size $CP" >> "$SCRIPT"
fi
# Note: P2P overlap in PP is ON by default in Megatron core_v0.16.1; only
# --no-overlap-p2p-communication exists (store_false). No positive flag.

cat >> "$SCRIPT" << 'DIST_CLOSE'
)
DIST_CLOSE

if (( NUM_EXPERTS > 0 )); then
    cat >> "$SCRIPT" << MOE_ARGS_EOF

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk ${MOE_TOPK}
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)
MOE_ARGS_EOF
else
    cat >> "$SCRIPT" << 'MOE_ARGS_EOF'

MOE_ARGS=()
MOE_ARGS_EOF
fi

# Memory-saving optimizer args for MP runs.
# Always: bf16 Adam m,v when MP>1 (halves fp32 Adam states from 65 to 32 GB/rank on 32B).
# Opt-in via GIPFEL_CPU_OFFLOAD=1: --optimizer-cpu-offload (Grace LPDDR, slow).
#   Warning: 4 × 32 GB bf16 Adam = 128 GB host RAM per node, plus pinned/param buffers,
#   plus libs — may exceed the --mem=460000 cgroup budget. Verify before using.
# Opt-in via GIPFEL_FP8=1: add --fp8-format hybrid + --fp8-param-gather (halves param storage).
# Opt-in via GIPFEL_RECOMPUTE={1,full}: --recompute-activations or full recompute.
if [ "$GIPFEL_OPTIMIZER" = "adam" ] && [ "$GIPFEL_ZERO" = "0" ]; then
    # exp-avg-dtype != fp32 is only valid alongside --use-precision-aware-optimizer.
    MEMORY_LINES="    --exp-avg-dtype $GIPFEL_EXP_AVG_DTYPE"$'\n'"    --exp-avg-sq-dtype $GIPFEL_EXP_AVG_SQ_DTYPE"
    if [ "$GIPFEL_MAIN_PARAMS_FP16" = "1" ]; then
        MEMORY_LINES+=$'\n'"    --main-params-dtype fp16"
    fi
else
    MEMORY_LINES=""
fi
if [ "$GIPFEL_CPU_OFFLOAD" != "0" ] && [ -n "$GIPFEL_CPU_OFFLOAD" ]; then
    MEMORY_LINES+=$'\n'"    --optimizer-cpu-offload"$'\n'"    --optimizer-offload-fraction ${GIPFEL_CPU_OFFLOAD}"
fi
if [ "$GIPFEL_FP8" = "1" ]; then
    # FP8 compute (hybrid+delayed). Note: --fp8-param-gather would trigger
    # multi_tensor_adam_fp8_cuda which asserts master_param is fp32, but
    # precision-aware-optimizer + store_param_remainders stores it as int16.
    # Gated by GIPFEL_FP8_PARAM_GATHER so default is safe.
    MEMORY_LINES+=$'\n'"    --fp8-format hybrid"$'\n'"    --fp8-recipe delayed"$'\n'"    --fp8-amax-history-len 16"$'\n'"    --fp8-amax-compute-algo max"
    if [ "${GIPFEL_FP8_PARAM_GATHER:-0}" = "1" ]; then
        MEMORY_LINES+=$'\n'"    --fp8-param-gather"
    fi
fi
case "$GIPFEL_RECOMPUTE" in
    1|selective)
        MEMORY_LINES+=$'\n'"    --recompute-activations"
        ;;
    full)
        MEMORY_LINES+=$'\n'"    --recompute-granularity full"$'\n'"    --recompute-method uniform"$'\n'"    --recompute-num-layers 1"
        ;;
esac
if [ "$GIPFEL_CPU_OFFLOADING_LAYERS" != "0" ]; then
    MEMORY_LINES+=$'\n'"    --cpu-offloading-num-layers $GIPFEL_CPU_OFFLOADING_LAYERS"
fi

if (( TP * PP > 1 )); then
    cat >> "$SCRIPT" << MEMORY

MEMORY_ARGS=(
$MEMORY_LINES
)
MEMORY
else
    cat >> "$SCRIPT" << 'MEMORY'

MEMORY_ARGS=()
MEMORY
fi

cat >> "$SCRIPT" << 'LOGGING_HEAD'

LOGGING_ARGS=(
    --log-throughput
    --log-progress
LOGGING_HEAD

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${MEMORY_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3-mp --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT (TP=${TP} PP=${PP} nodes=${NODES} mbs=${MBS})"
sbatch "$SCRIPT"
