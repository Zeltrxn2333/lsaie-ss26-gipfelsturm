# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Gipfelsturm is a distributed LLM training harness for the CSCS Alps/Clariden supercomputer (GH200 nodes, Slingshot-11 interconnect, SLURM). It wraps [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (git submodule, pinned to `core_v0.16.1`) with a single `launch.sh` entry point and supports two challenge tracks: loss-in-fixed-wall-clock and max tokens/sec/GPU throughput.

Training always uses the Nemotron-ClimbMix `climbmix_small` dataset pre-tokenized with the GPT-2 BPE tokenizer (for direct comparability with nanoGPT/nanochat baselines). Binary `.bin`/`.idx` files live on shared capstor storage — **do not re-tokenize**; point `DATA_PREFIX` at the existing artifact.

## Primary commands

All training and benchmarking goes through `./launch.sh`, which generates a self-contained SLURM script under `logs/<job-name>.sbatch` (for reproducibility) and submits it via `sbatch`.

```bash
./launch.sh <mode> <model_size> [steps] [nodes]
# modes:        throughput | train
# model_size:   125m | 350m | 760m | 1.5b | 3b | 8b
# steps:        required for train; defaults to 50 for throughput
# nodes:        default 4, max 8

./launch.sh throughput 760m              # 50-step perf probe, no logging
./launch.sh throughput 8b 50 1           # single-node throughput baseline
./launch.sh train 1.5b 3000 8            # 3000 training steps on 8 nodes
```

Infrastructure sanity check (NCCL all-reduce bus-bandwidth benchmark, intra- and inter-node):

```bash
sbatch test-infra.sbatch                 # 4 nodes × 4 GPUs, ~10 min
```

Expected healthy numbers on 4×GH200: ~340 GB/s intra-node (NVLink), ~93 GB/s inter-node (Slingshot-11).

## Execution environment

- **Cluster:** Clariden (Swiss AI Initiative partition on Alps), SLURM account `infra01`.
- **Container (EDF):** `alps3` — NGC PyTorch 26.01-py3 + alps extensions (NCCL 2.29.3 patched, libfabric 2.5.0a1, OpenMPI 5.0.9, nvshmem 3.4.5). Definition in `alps3.toml`; copy to `~/.edf/` on Clariden before use. All `srun` calls use `--environment=alps3`.
- **Storage layout:**
  - `/capstor/store/cscs/swissai/infra01/datasets/...` — shared datasets (ClimbMix parquet + pre-tokenized `.bin`/`.idx`).
  - `/iopsstor/scratch/cscs/$USER/gipfelsturm/...` — per-user scratch for logs, tensorboard, checkpoints, HF/Triton/Inductor caches. **Three-week deletion policy.**
- **Fixed SLURM shape:** 1 task/node, 4 GPUs/node, 288 CPUs/task, 460 GB mem, `--no-requeue`. Training uses `numactl --membind=0-3` and `pmix` with `--network=disable_rdzv_get`.
- **W&B:** auto-enabled in `train` mode iff `WANDB_API_KEY` is exported; otherwise `WANDB_MODE=disabled`. Not enabled in `throughput` mode.
- **Checkpointing:** currently disabled in practice due to a known SIGSEGV bug in Megatron checkpoint saving on GH200/ARM64 ([Megatron-LM #1861](https://github.com/NVIDIA/Megatron-LM/issues/1861)).

## Architecture: how launch.sh works

`launch.sh` is a shell metaprogram that emits a bash sbatch script. Understanding its structure matters because edits to training args almost always happen here.

- **Two `case` blocks at the top** set (a) mode config — steps, time limit, eval cadence, warmup, logging extras, W&B on/off — and (b) per-model config — `NUM_LAYERS`, `HIDDEN`, `FFN`, `HEADS`, `KV_HEADS`, and a hand-tuned `MBS`. `GBS=256` and `SEQ_LEN=4096` are fixed.
- **Script assembly uses multiple heredocs with different quoting rules**. Unquoted delimiters (`<< CONFIGS`) interpolate shell vars at generation time — values baked into the emitted file. Quoted delimiters (`<< 'SETUP'`) emit literally — `$VAR` is expanded at runtime inside the job. Getting this backward silently bakes launcher-side values where runtime values were intended, or vice versa. When adding args, pick the heredoc that matches when the value is known.
- **Hardcoded `WORKDIR=/users/schlag/gipfelsturm`** inside the generated script. If running under a different user, this path needs updating in the heredoc body (not just locally) — `$USER` is used for scratch paths but the launcher's own repo path is fixed.
- All `torchrun` arg arrays (`NETWORK_SIZE_ARGS`, `TRAINING_ARGS`, `REGULARIZATION_ARGS`, `LEARNING_RATE_ARGS`, `DISTRIBUTED_ARGS`, `LOGGING_ARGS`, `TOKENIZER_ARGS`, `DATA_ARGS`, …) are assembled separately and concatenated at the bottom into `TRAINING_CMD`. Default model-parallel config is `TP=1, PP=1` with distributed optimizer + overlap; change there for multi-node MP experiments.
- **Patches are applied at job start**, not at submit time: the generated script runs `git checkout -- .` then `git apply $WORKDIR/patches/*.patch` inside `Megatron-LM/` under a `flock` so concurrent jobs don't race. This means the submodule is kept "clean" in the repo; every job re-applies patches fresh.

## Megatron-LM patch workflow

Megatron-LM is a submodule pinned to an upstream tag. Local modifications live as standalone patch files in `patches/` — **never commit changes inside the submodule**.

To add a patch: make your edits inside `Megatron-LM/`, `git diff > ../patches/NNNN-name.patch`, then `git checkout -- .` to restore the submodule. Prepend a `#`-commented header to the patch explaining *what* it does, *where* in the code (file + function), and *how to relocate the edit* if line numbers shift in a future Megatron version — this is what makes patches maintainable across upgrades. Verify with `git apply --check ../patches/NNNN-name.patch`. Keep each patch narrowly scoped to a single concern so it can be dropped or re-based independently.

## Dataset and tokenizer notes

- Pre-tokenized binary prefix: `/capstor/.../climbmix_small_megatron/climbmix_small` (passed to Megatron as `--data-path $DATA_PREFIX`, which appends `.bin`/`.idx`).
- Raw parquet shards: `/capstor/.../climbmix_small/` (re-conversion via `data/parquet_to_megatron.py` + `data/convert_data.sbatch`, re-download via `data/download_climbmix.sh`).
- `--tokenizer-type GPT2BPETokenizer` with `data/gpt2-vocab.json` and `data/gpt2-merges.txt`. Any switch of tokenizer requires re-tokenizing the whole dataset.
- Split is hardcoded `99,1,0` (train/val/test).

## Conventions

- All model architectures are transformer-style: RoPE, GQA (`--num-query-groups`), SwiGLU, RMSNorm, untied embeddings, no bias on linears, dropout 0.
- Default optimizer: Adam (β1=0.9, β2=0.95), wd=0.1, clip=1.0, lr=3e-4, constant schedule with short warmup. Precision: bf16 everywhere including main grads (`--main-grads-dtype bf16`), via precision-aware optimizer and Transformer Engine.
- Logs and generated sbatch scripts land under `./logs/` (gitignored conceptually — don't commit).

---

## Local additions (not in upstream)

The fork adds Challenge-2 model-parallel support, env-var-driven knobs for cross-cluster use, one Megatron patch, and an ablation log. None of these touch upstream-shared files; they live as standalone additions.

### `launch-mp.sh` — model-parallel launcher

Drop-in alternative to `launch.sh`. Same `<mode> <size> [steps] [nodes]` interface, but adds:

- **New sizes**: `32b` (Qwen 2.5-32B exact: 64 layers, h=5120, ffn=27648, 40 heads, 8 kv heads), `140b` (Qwen-style dense extrapolation: 80 layers, h=12288, ffn=32768, 96 heads, 8 kv heads).
- **Per-tier MP defaults** matching the README Challenge 2 table: 8b → TP=1 PP=1, 32b → TP=4 PP=1, 140b → TP=4 PP=4. Override via `GIPFEL_TP` / `GIPFEL_PP`.
- **Preflight divisibility checks**: rejects configs where `WORLD_SIZE < TP*PP`, `NUM_LAYERS % PP != 0`, or `HIDDEN/HEADS/KV_HEADS % TP != 0`.
- **Cross-cluster parameterization**: `GIPFEL_ACCOUNT`, `GIPFEL_PARTITION`, `GIPFEL_WORKDIR`, `GIPFEL_DATA_PREFIX`, `GIPFEL_MEM`, `GIPFEL_TIME` — defaults preserve Clariden / `infra01` / `schlag` behavior.
- **Memory and precision knobs** (all opt-in, default off): `GIPFEL_MBS`, `GIPFEL_NUM_LAYERS`, `GIPFEL_SEQ_LEN`, `GIPFEL_RECOMPUTE` (`1` selective / `full`), `GIPFEL_FP8` (`hybrid+delayed` recipe), `GIPFEL_FP8_PARAM_GATHER`, `GIPFEL_EXP_AVG_DTYPE`, `GIPFEL_EXP_AVG_SQ_DTYPE`, `GIPFEL_MAIN_PARAMS_FP16`, `GIPFEL_NO_OVERLAP_PG`, `GIPFEL_NO_OVERLAP_GR`, `GIPFEL_DDP_BUCKET_SIZE`, `GIPFEL_TP_COMM_OVERLAP`, `GIPFEL_NCCL_TUNE`, `GIPFEL_CPU_OFFLOAD` (fraction 0..1), `GIPFEL_CPU_OFFLOADING_LAYERS`, `GIPFEL_OPTIMIZER` (adam/sgd/muon/dist_muon), `GIPFEL_TIMING` (1 or 2 for `--timing-log-level`), `GIPFEL_NO_MASTER_WEIGHTS` (requires patch 0002), `GIPFEL_USE_FA3` (requires FA3 venv).

Recipes that have been validated end-to-end on Daint (account `lp160`, partition `normal`):

```bash
# Best 2-node 32B throughput → 2253 tok/s/GPU, 445 TFLOP/s/GPU
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
GIPFEL_MEM=800000 GIPFEL_MBS=2 GIPFEL_USE_FA3=1 \
./launch-mp.sh throughput 32b 20 2

# Best 4-node 32B throughput → 2311 tok/s/GPU (+14% via FP8, only fits at DP=4)
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_FP8=1 \
./launch-mp.sh throughput 32b 20 4

# Multi-node 140B tier (8 nodes minimum so DP=2 shards optimizer state)
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=02:00:00 \
./launch-mp.sh throughput 140b 15 8

# Single-node 32B-shape pretraining (only fits with shrunk layer count + every memory saver)
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=01:00:00 GIPFEL_MEM=800000 \
GIPFEL_EXP_AVG_DTYPE=fp8 GIPFEL_EXP_AVG_SQ_DTYPE=fp8 \
GIPFEL_NO_MASTER_WEIGHTS=1 GIPFEL_NO_OVERLAP_PG=1 \
GIPFEL_RECOMPUTE=full GIPFEL_NUM_LAYERS=48 \
./launch-mp.sh throughput 32b 15 1
```

### `patches/0002-no-master-weights-option.patch`

Adds a `--no-master-weights` CLI flag to Megatron core_v0.16.1 that passes `master_weights=False` to TE FusedAdam. Skips per-parameter fp32/fp16/int16 master allocation. Saves ~one param-sized buffer per rank at the cost of bf16-precision Adam updates.

- Triggered by setting `GIPFEL_NO_MASTER_WEIGHTS=1` when calling `launch-mp.sh`.
- Auto-applied at job start by the same `flock + git apply` block as patch 0001.
- Useful only on memory-constrained MP configs (e.g., single-node 32B). On 2-node 32B the patch is harmless but does not improve throughput.

### `patches/README-32B-single-node-brainstorm.md`

Documents the five plans evaluated for fitting 32B on a single node. Plan 5 (no-master-weights) is what patch 0002 implements. Read this if you want to extend the memory-reduction work (e.g., rebuild TransformerEngine with smaller workspace, or add CUDA managed-memory spill via NVLink-C2C).

### `experiments/32b-single-node-ablation.md`

Full log of every single-node 32B attempt (19 runs covering env-var combinations, patch attempts, layer reductions). Documents which knobs help, which are no-ops on Daint, and where the per-rank 95 GB HBM ceiling is dominated by TE workspace (not model state).

### Flash-Attention 3 (built from source, not in repo)

The container ships flash-attn 2.7.4 (FA2) but TE 2.11 already imports `flash_attn_3` if present. To get FA3:

```bash
# One-time, ~1 hour on a Daint compute node
mkdir -p /iopsstor/scratch/cscs/$USER/{venvs/fa3,build}
cd /iopsstor/scratch/cscs/$USER/build
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
# Inside alps3 container (via srun --environment=alps3):
export FLASH_ATTN_CUDA_ARCHS=90a MAX_JOBS=8
pip install --no-build-isolation --target=/iopsstor/scratch/cscs/$USER/venvs/fa3 .

# Post-install fix: setup.py installs .py files at venv root, not inside the package
cd /iopsstor/scratch/cscs/$USER/venvs/fa3
mv flash_attn_interface.py flash_attn_config.py flash_attn_3/
touch flash_attn_3/__init__.py

# Strip the torch + nvidia + … dependencies pip pulled in (they conflict with the container's torch)
rm -rf torch torch-*.dist-info functorch torchgen pkg_resources nvidia* networkx* sympy* mpmath* triton* setuptools* jinja2* MarkupSafe* markupsafe* filelock* fsspec* typing_extensions* ninja* packaging* bin _distutils_hack distutils-precedence.pth
```

After this one-time setup, `GIPFEL_USE_FA3=1` in any `launch-mp.sh` invocation prepends the venv to `PYTHONPATH` and adds `--attention-backend flash` so Megatron forces TE to dispatch attention through FA3. On 32B / head_dim=128 the gain is small (~+0.8%) because cuDNN's fused attention is already well-tuned at this shape.

### Profiling

Set `GIPFEL_TIMING=2` to add `--timing-log-level 2` to the Megatron command. Per-iteration timer breakdowns (`forward-compute`, `backward-compute`, `optimizer-inner-step`, `all-grads-sync`, `params-all-gather`, …) show up at every `--log-interval`. On 32B 2-node baseline this revealed compute is 99.7% of iter time; communication is fully hidden by `--overlap-grad-reduce` / `--overlap-param-gather`.

### Megatron-LM submodule note

`launch-mp.sh` reuses the same `flock + git checkout -- . && git apply patches/*.patch` block as `launch.sh`. Both patches (0001, 0002) apply cleanly to the pinned `core_v0.16.1`. Never commit modifications inside `Megatron-LM/`; always go through `patches/`.
