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
