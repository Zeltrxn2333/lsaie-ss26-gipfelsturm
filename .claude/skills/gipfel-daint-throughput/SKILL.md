---
name: gipfel-daint-throughput
description: Submit a gipfelsturm throughput-mode job on CSCS Daint under project lp160 and report post-warmup tok/s/GPU. Use when the user asks to run, benchmark, or measure a gipfelsturm model size (125m, 350m, 760m, 1.5b, 3b, 8b) on Daint.
---

# Gipfel Daint throughput baseline

Submit a single-node (or multi-node) `throughput` job from `launch.sh` on Daint, wait for completion, extract post-warmup `tokens/sec/GPU`, compare against the Clariden README baselines.

## Environment (fixed for this session)

- SSH host alias: `cscs-daint` (resolves to `daint.cscs.ch` via `ela.cscs.ch`, user `ashen`)
- SLURM account: `lp160`; partition: `normal` (1-day cap; `debug` is 30-min)
- Daint WORKDIR: `/users/ashen/gipfelsturm` — a git clone of `git@github.com:Zeltrxn2333/lsaie-ss26-gipfelsturm.git`
- EDF: `~/.edf/alps3.toml` (already installed on Daint)
- Dataset: `/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small.{bin,idx}` — world-readable, no staging

## Syncing edits

Local edits under `/scratch2/aoshen/lsaie-ss26-gipfelsturm/` → `git commit && git push`, then on Daint `cd /users/ashen/gipfelsturm && git pull`. Do NOT rsync. When pulling, submodule objects travel under `.git/modules/Megatron-LM/` automatically; the launcher's `flock` + `git checkout -- . && git apply patches/*.patch` handles patch application per job.

## Submit one size

```bash
SIZE=125m    # or 350m 760m 1.5b 3b 8b
NODES=1      # default 1 for the README baseline; can be 4 or 8 for multi-node
ssh cscs-daint "cd /users/ashen/gipfelsturm && \
  export GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal GIPFEL_WORKDIR=/users/ashen/gipfelsturm && \
  ./launch.sh throughput $SIZE 50 $NODES" 2>&1 | tail -5
```

The launcher prints `Submitted batch job <JID>`; capture `<JID>` for monitoring. Job name is `gipfel-throughput-<SIZE>-50s-<NODES>n`; log is `logs/<jobname>-<JID>.log` under WORKDIR.

## Fire a sweep in parallel

Submit all sizes in one ssh, return `size JID` pairs:

```bash
ssh cscs-daint 'cd /users/ashen/gipfelsturm && \
  export GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal GIPFEL_WORKDIR=/users/ashen/gipfelsturm && \
  for s in 125m 350m 760m 1.5b 3b 8b; do \
    jid=$(./launch.sh throughput $s 50 1 2>&1 | awk "/Submitted batch job/{print \$4}"); \
    echo "$s $jid"; \
  done'
```

Each job is 1 node × 30 min walltime cap; they queue independently and run in parallel as the scheduler has capacity.

## Monitor to completion

Terminal states: `COMPLETED | FAILED | TIMEOUT | CANCELLED | NODE_FAIL | OUT_OF_MEMORY | PREEMPTED | BOOT_FAIL`.

**Pitfall:** `sacct -j <jid> --noheader -o State` left-pads with spaces — anchor the match loosely (not `^COMPLETED`). Use `grep -wE 'COMPLETED|FAILED|...'` or strip with `xargs`.

```bash
ssh cscs-daint "JIDS='3249151 3249152 ...'; \
  for j in \$JIDS; do \
    s=\$(sacct -j \$j --noheader -o State 2>/dev/null | head -1 | xargs); \
    echo \"\$j \$s\"; \
  done"
```

Prefer the `Monitor` tool for long-running waits — one event per iteration until terminal state, then final log grep.

## Extract post-warmup tok/s/GPU

Warmup is `LR_WARMUP_ITERS=10` (throughput mode). Use iters 20..50 to avoid warmup bleed and first-iter compile/index cost:

```bash
ssh cscs-daint "LOG=/users/ashen/gipfelsturm/logs/gipfel-throughput-<SIZE>-50s-<NODES>n-<JID>.log; \
  grep -oE 'iteration +[0-9]+/.*tokens/sec/GPU: [0-9]+' \$LOG | \
  awk -F'iteration +| /|tokens/sec/GPU: ' '\$2+0>=20 {sum+=\$NF; n+=1} END {if(n)printf \"mean_tok_s_gpu=%d (n=%d)\n\", sum/n, n}'"
```

Also report median via `sort -n | awk` if iter-to-iter variance is high (125m showed SD≈5k tok/s/GPU on Daint).

## Reference values

| Model | MBS | Clariden README | Daint (iters 20-50 mean) | Ratio |
|-------|-----|-----------------|---------------------------|-------|
| 125m  | 16  | 54,671          | ~40,000                   | 73%   |
| 350m  | 8   | 62,711          | 58,364                    | 93%   |
| 760m  | 6   | 74,994          | 58,380                    | 78%   |
| 1.5b  | 4   | 34,054          | 35,503                    | **104%** |
| 3b    | 4   | 19,842          | 20,316                    | **102%** |
| 8b    | 2   | 10,882          | 11,079                    | **102%** |

Measured 2026-04-21 on Daint/lp160, partition=normal, 1 node. Compute-bound sizes (1.5B+) match or beat Clariden; small sizes (125m-760m) drop 7-27% and this is I/O / warmup dominated, not a hardware delta. Daint and Clariden are both GH200 vClusters on the same Alps hardware pool (`docs.cscs.ch/alps/hardware/`).

## Challenge 2 tiers (`launch-mp.sh`)

For 32B / 140B / or TP>1 experiments, use `launch-mp.sh` (fork of `launch.sh` with MP support). Usage:

```bash
./launch-mp.sh throughput <size> [steps] [nodes]
# new sizes: 32b (Qwen 2.5-32B exact: 64L/h=5120/ffn=27648/40H/8KV)
#            140b (Qwen-style dense extrapolation: 80L/h=12288/ffn=32768/96H/8KV)
# Per-size MP defaults: 8b→TP=1/PP=1, 32b→TP=4/PP=1, 140b→TP=4/PP=4
# Override via GIPFEL_TP / GIPFEL_PP.
```

Env-var knobs (all opt-in):

- `GIPFEL_TIME=HH:MM:SS` — override walltime (default 00:30:00 throughput / 02:30:00 train).
- `GIPFEL_RECOMPUTE={1|full}` — `--recompute-activations` (selective) or full recompute.
- `GIPFEL_FP8=1` — `--fp8-format hybrid --fp8-recipe delayed --fp8-param-gather` (Hopper-safe).
- `GIPFEL_CPU_OFFLOAD=1` — `--optimizer-cpu-offload` (host cgroup OOMs easily on 4 ranks × bf16 Adam; avoid unless needed).
- `GIPFEL_MAIN_PARAMS_FP16=1` — fp16 master params instead of fp32 (halves the hidden 32 GB/rank precision-aware master-param cost).

Automatic when `TP*PP>1`:
- `--exp-avg-dtype bf16 --exp-avg-sq-dtype bf16` (halves Adam state from fp32 to bf16).
- `--sequence-parallel` (when TP>1).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in all runs.

**Measured MP results (2026-04-21, Daint/lp160):**

| Tier | Config | Nodes | Steps | tok/s/GPU | TFLOP/s/GPU | Iter time |
|------|--------|-------|-------|-----------|-------------|-----------|
| 8B   | TP=1 PP=1          | 1 | 50 | 11,079 | ~135 | ~7 s   |
| 32B  | TP=4 PP=1 DP=2     | 2 | 20 | 2,076  | 411  | 63 s   |
| 140B | TP=4 PP=4 DP=2     | 8 | 15 | 581    | 453  | 56 s   |

TFLOP/s/GPU rises with model size (bigger matmuls, higher arithmetic intensity). tok/s/GPU drops because larger models process fewer tokens per FLOP.

**Pitfalls learned:**

1. **32B TP=4 single-node does NOT fit** under Megatron core_v0.16.1 with any combination of fp8 Adam, FP8 compute, FP8 param-gather, no-overlap-PG, recompute (selective or full), optimizer CPU offload 0.3–0.7, TE activation offload, or Muon. 14 attempts, all OOM. Root cause: precision-aware distributed optimizer with DP=1 has ~65 GB of state per rank that can't be sharded, plus 10+ GB unavoidable TE/NCCL workspace. Use ≥2 nodes.
2. **`--overlap-p2p-communication` does NOT exist** as a positive flag in core_v0.16.1; only `--no-overlap-p2p-communication` (store_false, default True). Default P2P overlap is already on.
3. **`--optimizer-cpu-offload` has subtle host memory pressure**: Megatron's allocation pattern during iterations grows pinned buffers over time, even at offload fraction 0.5. Default `--mem=460000` is too tight for any offload config; raise `GIPFEL_MEM=800000` (Daint has 870 GB per node). Even then, full recompute + CPU offload together can blow host mem mid-iteration.
4. **`--fp8-param-gather` is INCOMPATIBLE with precision-aware-optimizer** in core_v0.16.1: TE's `multi_tensor_adam_fp8_cuda` kernel asserts master_param is fp32, but `store_param_remainders=True` (default on bf16) stores it as int16. No CLI knob disables remainders. Don't combine these two; keep `GIPFEL_FP8_PARAM_GATHER=0`.
5. **Muon requires `Emerging Optimizers` package**, not installed in the alps3 image. Unavailable.
6. **`PYTORCH_CUDA_ALLOC_CONF=...,max_split_size_mb:128` cuts throughput 2.2×** on 2-node 32B (2076 → 940 tok/s/GPU). The cap serializes large-tensor allocations. Use `expandable_segments:True` alone.
7. **140B requires ≥8 nodes** under this recipe: TP=4·PP=4=16 GPUs = 4 nodes minimum, but DP=1 at 4-node forces unsharded optimizer = OOM. 8 nodes gives DP=2, comfortable fit.

## 32B throughput ablation (2026-04-23)

Measured at SEQ_LEN=4096, GBS=256, MBS=1, TP=4, PP=1, 20 steps, iters 5-20 mean:

| Run | Nodes | DP | Extra knobs | tok/s/GPU | TFLOP/s/GPU | Iter time | Note |
|-----|-------|----|-----------  |-----------|-------------|-----------|------|
| F6  | 2     | 2  | bf16 default                    | 2036 | 401 | 64 s | Baseline |
| F9  | 4     | 4  | bf16 default                    | 2024 | 402 | 32 s | Linear DP scale |
| F8  | 4     | 4  | + `GIPFEL_FP8=1` (no param-gather) | **2311** | **458** | 28 s | **+14% over bf16** |
| F4  | 2     | 2  | + `GIPFEL_EXP_AVG_DTYPE=fp8`    | 622  | 122 | 211 s | fp8 Adam HURTS perf 3× |
| F7b | 2     | 2  | + `GIPFEL_FP8=1`                | OOM @ 95 GB init | — | — | FP8 workspace doesn't fit DP=2 |
| F10 | 2     | 2  | + `GIPFEL_FP8=1 GIPFEL_NO_OVERLAP_PG=1` | OOM @ init | — | — | Still doesn't fit; need DP≥4 for FP8 |
| **M2** | **2** | **2** | **+ `GIPFEL_MBS=2`** | **2236** | **442** | 58 s | **+9.7% from arithmetic intensity** |
| M2_FA3 | 2 | 2 | + MBS=2 + `GIPFEL_USE_FA3=1` (forces `--attention-backend flash` + FA3 venv) | **2253** | **445** | 58 s | +0.8% over M2 (cuDNN was already optimal at head_dim=128) |
| M2_TPO | 2 | 2 | + MBS=2 + `--tp-comm-overlap` | NCCL/OFI userbuffers init failed | — | — | Skip on Slingshot-11 |
| M4    | 2 | 2 | + MBS=4                       | OOM iter 2 | — | — | activation memory wall |
| M4R   | 2 | 2 | + MBS=4 + recompute selective | OOM via NCCL "calloc async" | — | — | still doesn't fit |
| M4F   | 2 | 2 | + MBS=4 + recompute full      | ~1700 (worse) | — | — | full recompute kills throughput |
| F8C   | 2 | 2 | + FP8 + no-master + cpu-offload 0.3 + recompute | crashed iter 2 | — | — | doesn't fit cleanly |

## Patches (Megatron source-level changes)

- **`patches/0001-log-tokens-per-sec-to-wandb.patch`** — adds `tokens-per-sec-per-gpu` to log/TB/W&B output.
- **`patches/0002-no-master-weights-option.patch`** — adds `--no-master-weights` to skip TE FusedAdam's fp32 master allocation. Saves init time but not peak HBM (TE workspace dominates). Useful as platform for further memory work, not as standalone speedup.
- **No 0003 patch needed for FA3.** TE 2.11 in alps3 already has the FA3 import path. Just install flash-attn-3 to a venv and prepend to PYTHONPATH; pass `--attention-backend flash` to opt out of TE's auto cuDNN choice. The launch-mp.sh `GIPFEL_USE_FA3=1` knob does both.

## FA3 install (one-time, ~1 hour build)

flash-attn-3 wheels not on PyPI. Built from source:

```bash
cd /iopsstor/scratch/cscs/$USER/build
git clone --depth=1 https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
# In alps3 container:
export FLASH_ATTN_CUDA_ARCHS=90a MAX_JOBS=8
pip install --no-build-isolation --target=/iopsstor/scratch/cscs/$USER/venvs/fa3 .
# After build, fix package layout (.py files install at venv root, must be in flash_attn_3/):
cd /iopsstor/scratch/cscs/$USER/venvs/fa3
mv flash_attn_interface.py flash_attn_config.py flash_attn_3/
touch flash_attn_3/__init__.py
# Strip pulled-in deps that conflict with container's torch:
rm -rf torch torch-*.dist-info functorch torchgen pkg_resources nvidia* networkx* sympy* mpmath* triton* setuptools* jinja2* MarkupSafe* markupsafe* filelock* fsspec* typing_extensions* ninja* packaging* bin _distutils_hack distutils-precedence.pth
```

Build needs MAX_JOBS=8 (not higher) on a 4-GPU node with `--mem=800000` to avoid host OOM during ~50 .cu file compile. Total walltime ~1 hour.

## Profile (timing-log-level=2) data — 2n 32B baseline

iter 63.6 s, breakdown:

| Phase | Time (ms) | % of iter |
|-------|-----------|-----------|
| forward-compute  | 21,640 | 34% |
| backward-compute | 41,750 | 66% |
| optimizer (full step) | 119 | 0.2% |
| all-grads-sync   | 1.5 | <0.01% |
| params-all-gather | 0.01 | <0.001% |

**Compute is 99.7% of iter time** — communication and optimizer are not bottlenecks. Throughput optimizations must target compute. FA3 only marginally helps because attention is small share of compute (FFN dominates). FP8 compute would be the big win but doesn't fit at 2-node DP=2.

## Recommended max-throughput config for 2-node 32B

```bash
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
GIPFEL_MEM=800000 GIPFEL_MBS=2 GIPFEL_USE_FA3=1 \
./launch-mp.sh throughput 32b 20 2
```

Yields **2253 tok/s/GPU, 445 TFLOP/s/GPU** (≈22% MFU on bf16 peak). The +14% headroom on 4-node FP8 (2311 tok/s/GPU) reflects DP=4 fitting FP8 workspace; on 2-node we're constrained by per-rank memory.

**Key findings:**
- **FP8 compute (hybrid recipe, no param-gather) gives a clean +14% throughput** on 4-node 32B. Free win at ≥4 nodes.
- **fp8 Adam m/v is a 3× THROUGHPUT REGRESSION** in core_v0.16.1. The fp8 Adam cuda kernel is slow here; stay on bf16.
- **FP8 compute needs DP≥4** to fit its workspace on top of optimizer state (2-node OOMs even with no-overlap-PG).
- bf16 per-GPU throughput is DP-invariant: 2n 2036 ≈ 4n 2024, confirming no interconnect bottleneck at these shapes.

**Recipe for maximum 32B throughput on this cluster:** 4 nodes, `GIPFEL_FP8=1`, everything else default. 2311 tok/s/GPU = 458 TFLOP/s/GPU ≈ 22.9% MFU (vs 2000 TFLOP peak GH200 bf16-equivalent; FP8 theoretical peak is 4000 TFLOP so ~11.5% of FP8 peak — still room to push).

## Failure modes seen, and fixes

- `pyxis: couldn't chdir to /users/schlag: Permission denied` → `alps3.toml` had `workdir = "/users/schlag"`; set it to `"/"`. Fixed in commit `ab12228`.
- `git apply` fails in the submodule → submodule gitdir pointer or `.git/modules/Megatron-LM/` missing on Daint. When syncing via git, this is handled automatically; when rsyncing, do not exclude `.git/modules/`.
- Job COMPLETED in <60s with no training output → spank/pyxis container init failed; check the log head for `spank_pyxis.so task_init() failed`.
