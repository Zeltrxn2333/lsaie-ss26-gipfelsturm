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

1. **32B TP=4 single-node OOMs** regardless of recompute/FP8 because `--use-precision-aware-optimizer` keeps a fp32 *master* param copy (32.5 GB/rank) that isn't sharded when DP=1. Baseline exceeds 95 GB HBM before activations. Solutions: 2+ nodes (DP≥2 shards via `--use-distributed-optimizer`), or `GIPFEL_MAIN_PARAMS_FP16=1` + `GIPFEL_RECOMPUTE=full` (tight, not verified working here).
2. **`--overlap-p2p-communication` does NOT exist** as a positive flag in core_v0.16.1; only `--no-overlap-p2p-communication` (store_false, default True). Positive form causes argparse to reject the whole command and kill all ranks after ~65 s. Default P2P overlap is already on.
3. **`--optimizer-cpu-offload` without a larger `--mem`** triggers host cgroup OOM: 4 ranks × bf16 Adam (32 GB each) = 128 GB plus pinned buffers plus libs exceeds 460 GB cgroup on top of other tenants.
4. **140B requires ≥8 nodes** under this recipe: TP=4·PP=4=16 GPUs minimum = 4 nodes, but DP=1 at 4-node forces fp32 master params + Adam unsharded = 108 GB/rank > 95 GB HBM. 8 nodes gives DP=2, shards everything 50/50, comfortable fit.

## Failure modes seen, and fixes

- `pyxis: couldn't chdir to /users/schlag: Permission denied` → `alps3.toml` had `workdir = "/users/schlag"`; set it to `"/"`. Fixed in commit `ab12228`.
- `git apply` fails in the submodule → submodule gitdir pointer or `.git/modules/Megatron-LM/` missing on Daint. When syncing via git, this is handled automatically; when rsyncing, do not exclude `.git/modules/`.
- Job COMPLETED in <60s with no training output → spank/pyxis container init failed; check the log head for `spank_pyxis.so task_init() failed`.
