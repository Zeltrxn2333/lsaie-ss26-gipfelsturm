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
# GIPFEL_CPU_OFFLOAD=1   — offload optimizer state to Grace LPDDR (slow; only if GPU OOM and no other levers help)
# GIPFEL_RECOMPUTE=1     — --recompute-activations (selective recompute in attention)
# GIPFEL_RECOMPUTE=full  — --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 (full recompute)
# GIPFEL_FP8=1           — --fp8-format hybrid --fp8-recipe delayed --fp8-param-gather (Hopper-safe)
GIPFEL_CPU_OFFLOAD=${GIPFEL_CPU_OFFLOAD:-0}
GIPFEL_RECOMPUTE=${GIPFEL_RECOMPUTE:-0}
GIPFEL_FP8=${GIPFEL_FP8:-0}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch-mp.sh train <model_size> <steps> [nodes]}
        TIME=02:30:00
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
    32b)
        NUM_LAYERS=64;  HIDDEN=5120;  FFN=27648;  HEADS=40; KV_HEADS=8
        MBS=1;  DEFAULT_NODES=1; DEFAULT_TP=4; DEFAULT_PP=1
        ;;
    140b)
        NUM_LAYERS=80;  HIDDEN=12288; FFN=32768;  HEADS=96; KV_HEADS=8
        MBS=1;  DEFAULT_NODES=4; DEFAULT_TP=4; DEFAULT_PP=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m|350m|760m|1.5b|3b|8b|32b|140b"
        exit 1
        ;;
esac

NODES=${4:-$DEFAULT_NODES}
TP=${GIPFEL_TP:-$DEFAULT_TP}
PP=${GIPFEL_PP:-$DEFAULT_PP}
GPUS_PER_NODE=4
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

GBS=256
SEQ_LEN=4096
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-tp${TP}pp${PP}-${TRAINING_STEPS}s-${NODES}n"

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
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

if [ -n "$GIPFEL_PARTITION" ]; then
    echo "#SBATCH --partition=${GIPFEL_PARTITION}" >> "$SCRIPT"
fi

cat >> "$SCRIPT" << BODY_PARAM

echo "START TIME: \$(date)"
echo "MP config: TP=${TP} PP=${PP} (world=${WORLD_SIZE})"

################ Configs ################
WORKDIR=${GIPFEL_WORKDIR}
DATA_PREFIX=${GIPFEL_DATA_PREFIX}
BODY_PARAM

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
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

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
    --optimizer adam
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
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
DISTRIBUTED

if (( TP > 1 )); then
    echo "    --sequence-parallel" >> "$SCRIPT"
fi
if (( PP > 1 )); then
    echo "    --overlap-p2p-communication" >> "$SCRIPT"
fi

cat >> "$SCRIPT" << 'DIST_CLOSE'
)
DIST_CLOSE

# Memory-saving optimizer args for MP runs.
# Always: bf16 Adam m,v when MP>1 (halves fp32 Adam states from 65 to 32 GB/rank on 32B).
# Opt-in via GIPFEL_CPU_OFFLOAD=1: --optimizer-cpu-offload (Grace LPDDR, slow).
#   Warning: 4 × 32 GB bf16 Adam = 128 GB host RAM per node, plus pinned/param buffers,
#   plus libs — may exceed the --mem=460000 cgroup budget. Verify before using.
# Opt-in via GIPFEL_FP8=1: add --fp8-format hybrid + --fp8-param-gather (halves param storage).
# Opt-in via GIPFEL_RECOMPUTE={1,full}: --recompute-activations or full recompute.
MEMORY_LINES="    --exp-avg-dtype bf16"$'\n'"    --exp-avg-sq-dtype bf16"
if [ "$GIPFEL_CPU_OFFLOAD" = "1" ]; then
    MEMORY_LINES+=$'\n'"    --optimizer-cpu-offload"$'\n'"    --optimizer-offload-fraction 1.0"
fi
if [ "$GIPFEL_FP8" = "1" ]; then
    MEMORY_LINES+=$'\n'"    --fp8-format hybrid"$'\n'"    --fp8-recipe delayed"$'\n'"    --fp8-amax-history-len 16"$'\n'"    --fp8-amax-compute-algo max"$'\n'"    --fp8-param-gather"
fi
case "$GIPFEL_RECOMPUTE" in
    1|selective)
        MEMORY_LINES+=$'\n'"    --recompute-activations"
        ;;
    full)
        MEMORY_LINES+=$'\n'"    --recompute-granularity full"$'\n'"    --recompute-method uniform"$'\n'"    --recompute-num-layers 1"
        ;;
esac

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
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT (TP=${TP} PP=${PP} nodes=${NODES} mbs=${MBS})"
sbatch "$SCRIPT"
