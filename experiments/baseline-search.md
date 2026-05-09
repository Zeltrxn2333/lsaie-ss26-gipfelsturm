# Baseline Search — Comprehensive Report

**Goal**: identify a *good* baseline configuration for the LSAIE Challenge-2
throughput track. "Good" means:

1. **Real, published model** (not a synthetic extrapolation).
2. **Reasonable parallelism** — TP locked within a node (NVLink-C2C),
   PP across nodes for memory, EP across nodes for MoE expert sharding,
   DP fills the rest. World size must equal `TP × PP × EP × DP` exactly.
3. **Stock Megatron defaults** — no `GIPFEL_USE_FA3`, no `GIPFEL_FP8`, no
   `GIPFEL_NO_MASTER_WEIGHTS`, no `GIPFEL_RECOMPUTE`, no `GIPFEL_EXP_AVG_DTYPE`,
   etc. Only the unconditional knobs that come with `launch-mp.sh`:
   `--use-distributed-optimizer`, `--overlap-grad-reduce`,
   `--overlap-param-gather`, `--sequence-parallel`, bf16 mixed precision.
4. **Naturally moderate or low MFU** — the baseline should leave clear
   headroom for the optimization techniques the project demonstrates.

Cluster: **CSCS Daint GH200** (4 GH200 / node, NVLink-C2C 900 GB/s intra,
Slingshot-11 ~93 GB/s inter; 95 GB usable HBM per GPU). Megatron-LM
core_v0.16.1.

GH200 bf16 dense peak ≈ **494 TFLOP/s/GPU**. MFU = TFLOP/s/GPU ÷ 494.
For MoE, Megatron's `--log-throughput` counts FLOPs over **active**
parameters (top-K of N experts), so MFU is comparable across dense and MoE.

Constraint per project scope: **total parameters ≤ 200B**. (Excludes
Llama 3.1-405B, DeepSeek V2-236B, Qwen3-235B-A22B.)

---

## All experiments (chronological, 22 runs)

### Round 1 — initial sweep on Qwen 2.5 (existing model cases)

| # | Model | Real | Nodes | TP | PP | EP | DP | MBS | TFLOP/s | MFU | n | Job |
|---|-------|------|-------|----|----|----|----|----|---------|------|---|-----|
| 1 | Qwen 2.5-8B   | ✓ | 1 | 1 | 1 | — | 4 | 2 | 495.8 | **100 %** | 16 | 3383408 |
| 2 | Qwen 2.5-32B  | ✓ | 2 | 4 | 1 | — | 2 | 1 | 405.1 | **82.0 %** | 16 | 3383409 |
| 3 | Qwen 2.5-32B  | ✓ | 4 | 4 | 1 | — | 4 | 1 | 396.5 | **80.3 %** | 16 | 3383410 |
| 4 | 140B (synth)  | ✗ | 8 | 4 | 4 | — | 2 | 1 | 445.1 | **90.1 %** | 11 | 3383415 |

### Round 2 — added real Qwen3 + Llama 3 + Qwen 2.5-72B cases to launch-mp.sh

| # | Model | Real | Nodes | TP | PP | EP | DP | TFLOP/s | MFU | Status |
|---|-------|------|-------|----|----|----|----|---------|------|--------|
| 5 | Qwen3-14B     | ✓ | 1 | 4 | 1 | — | 1 | 339.2 | **68.7 %** | ✓ (3383834) |
| 6 | Qwen3-14B     | ✓ | 1 | 1 | 1 | — | 4 | — | — | OOM init (3383842) — TP=1 doesn't fit |
| 7 | Qwen3-32B     | ✓ | 2 | 4 | 1 | — | 2 | 383.9 | **77.7 %** | ✓ (3383835) |
| 8 | Llama 3.1-70B | ✓ | 4 | 4 | 2 | — | 2 | 446.0 | 90.3 % | partial — OOM at iter 14 (3383841) |
| 9 | Qwen 2.5-72B  | ✓ | 4 | 4 | 2 | — | 2 | — | — | OOM iter 1 (3383837) |

### Round 3 — PP=4 DP=1 fix for the 70B/72B class

The PP=2 DP=2 configurations OOM because TP=4 PP=2 leaves 9 B params/rank,
where weight + grad + optimizer state (with dist-opt sharding /2) ≈ 63 GB,
and the remaining 32 GB are eaten by activations + TE/cuBLAS workspace.
PP=4 DP=1 cuts per-rank weight to 4.5 B params at the cost of giving up
dist-opt sharding — net: smaller per-rank weight wins, fits in budget.
PP bubble at num_microbatches = 256 / 4 stages = 0.78 % is negligible.

| # | Model | Real | Nodes | TP | PP | EP | DP | TFLOP/s | MFU | Status |
|---|-------|------|-------|----|----|----|----|---------|------|--------|
| 10 | Llama 3.1-70B | ✓ | 4 | 4 | 4 | — | 1 | **444.2** | **89.9 %** | ✓ (3383848) |
| 11 | Qwen 2.5-72B  | ✓ | 4 | 4 | 4 | — | 1 | **437.7** | **88.6 %** | ✓ (3383847) |

### Round 4 — added MoE wiring + Mixtral cases

`launch-mp.sh` extended with:
- `GIPFEL_EP` env var, per-tier `DEFAULT_EP`
- `MOE_ARGS` block with Megatron's official Mixtral flags:
  `--num-experts`, `--moe-router-topk`, `--moe-router-load-balancing-type aux_loss`,
  `--moe-aux-loss-coeff 1e-2`, `--moe-grouped-gemm`,
  `--moe-token-dispatcher-type alltoall`, `--expert-model-parallel-size $EP`
- New cases `mixtral-8x7b` (47 B / 13 B active) and `mixtral-8x22b`
  (141 B / 39 B active)

| # | Model | Real | Nodes | TP | PP | EP | DP | TFLOP/s | MFU | Status |
|---|-------|------|-------|----|----|----|----|---------|------|--------|
| 12 | Mixtral 8x7B  | ✓ | 2 | 4 | 1 | 2 | 1 | — | — | OOM init (3383849) |
| 13 | Mixtral 8x22B | ✓ | 8 | 4 | 1 | 4 | 2 | — | — | OOM init (3383850); needs ≥ 16 nodes |
| 14 | **Mixtral 8x7B** | ✓ | **4** | 4 | 1 | 2 | 2 | **188.8** | **38.2 %** | ✓ (3383851) |

### Round 5 — Mixtral 8x7B 2n exhaustive MP sweep (all OOM)

Goal: confirm whether **any** legal MP fits Mixtral 8x7B on 2 nodes with
stock defaults. Answer: **no**.

47 B params / 8 GPUs ≈ 5.9 B per rank. Per-rank state (weight + grad +
optimizer) ≈ 60 GB regardless of how the 8-way split is partitioned across
TP / PP / EP / DP. Activations + TE workspace + cuBLAS/grouped-GEMM
scratch add another 30+ GB → all configurations exceed the 95 GB ceiling.

| # | MP (2n, world=8) | TFLOP/s | MFU | Status | Job |
|---|------------------|---------|------|--------|-----|
| 15 | TP=8 PP=1 EP=1 DP=1 | — | — | OOM init (TP cross-node, would also be slow) | 3383859 |
| 16 | TP=4 PP=2 EP=1 DP=1 | — | — | OOM init (PP cross-node) | 3383860 |
| 17 | TP=2 PP=1 EP=2 DP=2 | — | — | OOM init | 3383861 |
| 18 | TP=4 PP=1 EP=1 DP=2 | — | — | OOM init (EP=1 ⇒ 8 experts replicated per rank) | 3383862 |
| 19 | TP=2 PP=1 EP=4 DP=1 | — | — | OOM init | 3383865 |
| 20 | **TP=1 PP=1 EP=8 DP=1** | **247** | **50 %** for 1 iter | iter 1 succeeds, iter 2 OOM at grouped_linear backward | 3383866 |

### Round 6 — Mixtral 8x7B 2n + GIPFEL_RECOMPUTE=full (still OOM)

Activation recomputation only saves activation memory (the kept-from-forward
buffers used by backward), not the optimizer state, weight, or grad. The 60 GB
per-rank state floor remains.

| # | MP + recompute=full | TFLOP/s | Status | Job |
|---|---------------------|---------|--------|-----|
| 21 | TP=4 EP=2 DP=1 + recompute=full | — | OOM init | 3383867 |
| 22 | **TP=1 EP=8 DP=1 + recompute=full** | **265** for 1 iter (53.6 % MFU) | iter 1 succeeds, iter 2 OOM at grouped_linear | 3383868 |

To make Mixtral 8x7B fit on 2 nodes one would need to *also* drop the
"stock default" constraint with `GIPFEL_EXP_AVG_DTYPE=fp8`,
`GIPFEL_EXP_AVG_SQ_DTYPE=fp8`, and/or `GIPFEL_NO_MASTER_WEIGHTS=1` — each
of which is an explicit memory-saving optimization, not part of the baseline.

---

## Final results table (all stable runs, sorted by MFU descending)

| Rank | Config | MFU | TFLOP/s/GPU | Real model |
|------|--------|------|-------------|------------|
| 1 | Qwen 2.5-8B   1n TP=1 PP=1 DP=4 MBS=2 | **100.0 %** | 495.8 | ✓ |
| 2 | 140B-synth    8n TP=4 PP=4 DP=2       |   90.1 % | 445.1 | ✗ synthetic |
| 3 | Llama 3.1-70B 4n TP=4 PP=4 DP=1       |   89.9 % | 444.2 | ✓ |
| 4 | Qwen 2.5-72B  4n TP=4 PP=4 DP=1       |   88.6 % | 437.7 | ✓ |
| 5 | Qwen 2.5-32B  2n TP=4 PP=1 DP=2       |   82.0 % | 405.1 | ✓ |
| 6 | Qwen 2.5-32B  4n TP=4 PP=1 DP=4       |   80.3 % | 396.5 | ✓ |
| 7 | Qwen3-32B     2n TP=4 PP=1 DP=2       |   77.7 % | 383.9 | ✓ |
| 8 | **Qwen3-14B   1n TP=4 PP=1 DP=1**     |   **68.7 %** | 339.2 | ✓ |
| 9 | **Mixtral 8x7B 4n TP=4 PP=1 EP=2 DP=2** | **38.2 %** | 188.8 | ✓ |

---

## Recommended baselines

Two candidates, picked to span different optimization stories:

### Candidate A — **Mixtral 8x7B, 4 nodes, TP=4 PP=1 EP=2 DP=2, MBS=1, stock defaults**

| Property | Value |
|----------|-------|
| Architecture | Mixture of Experts (8 experts, top-2, no shared expert) |
| Total / active params | 47 B / 13 B |
| Megatron support | Official `examples/mixtral/train_mixtral_8x7b_distributed.sh` |
| Stock baseline | **188.8 TFLOP/s/GPU = 38.2 % MFU** |
| Job ID | 3383851 |

**Why pick A:** Lowest stock-default MFU in the entire sweep (38 %),
the only configuration with that headroom and reasonable parallelism
*and* a real model. Improvement axes are unusually rich because MoE
introduces optimization knobs that don't exist for dense models:

- MoE-specific levers: token-dispatcher choice (alltoall vs allgather),
  `--moe-permute-fusion`, grouped-GEMM kernel selection in TE, FP8
  expert GEMM (TE 2.11 supports it), token-drop / capacity-factor tuning,
  EP/DP rebalance.
- Dense-style levers that still apply: FA3 attention backend, MBS
  scaling (memory headroom is comfortable with EP=2 + DP=2 at 4 nodes),
  FP8 attention and attention-projection GEMMs.
- Improvement story spans multiple distinct axes — useful for the project
  to demonstrate breadth, not just stack FP8 on a dense model.

**Reproducible:**
```bash
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
  GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
  GIPFEL_EP=2 \
  ./launch-mp.sh throughput mixtral-8x7b 20 4
```

### Candidate B — **Qwen3-14B, 1 node, TP=4 PP=1 DP=1, MBS=1, stock defaults**

| Property | Value |
|----------|-------|
| Architecture | Dense decoder-only (RoPE, SwiGLU, RMSNorm, GQA 5:1) |
| Total params | 14 B |
| Megatron support | Generic GPT path with CLI-flag spec |
| Stock baseline | **339.2 TFLOP/s/GPU = 68.7 % MFU** |
| Job ID | 3383834 |

**Why pick B:** Single-node, the simplest possible infrastructure (no
cross-node Slingshot for any collective). TP=4 is *forced* by memory
(the alternative TP=1 PP=1 DP=4 OOMs); the resulting small per-shard
GEMM tile is what suppresses MFU to 68.7 %. Improvement story:

- Try TP=2 PP=1 DP=2 with `expandable_segments` — bigger GEMMs, expected
  ~78-82 % MFU, demonstrates that "more TP isn't always better".
- + MBS=2 if memory permits at TP=2 (more activation memory but bigger
  GEMMs).
- + FA3 (expected +2-4 %).
- + FP8 compute as the final memory-permitting step.

This baseline is a good fit if the project values infrastructure
simplicity (no MoE complexity, single-node training) and a pure dense
optimization narrative.

**Reproducible:**
```bash
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
  GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
  ./launch-mp.sh throughput qwen3-14b 20 1
```

### Comparison table

| Aspect | A: Mixtral 8x7B 4n | B: Qwen3-14B 1n |
|--------|--------------------|-----------------|
| Stock MFU | 38.2 % | 68.7 % |
| Headroom | very large; multi-axis | moderate; mostly TP-rebalance + FA3/FP8 |
| Model architecture | MoE | Dense |
| Hardware budget | 4 nodes (16 GPUs) | 1 node (4 GPUs) |
| Cross-node communication | EP all-to-all + DP grad-RS / param-AG | none |
| Real published model | ✓ Mixtral (Mistral AI 2024) | ✓ Qwen3 (Alibaba 2025) |
| Megatron-native MoE path | ✓ official `train_mixtral_8x7b_distributed.sh` | n/a (dense) |
| Optimization narrative | rich, varied (MoE + dense) | linear, well-trodden |
| Best for | demonstrating breadth of techniques | clean dense story, low resource cost |

---

## Why other candidates were rejected

| Config | MFU | Reason rejected |
|--------|------|-----------------|
| Qwen 2.5-8B 1n | 100 % | Saturated — zero optimization headroom |
| 140B-synth 8n | 90 % | Synthetic model, not a published architecture |
| Llama 3.1-70B 4n | 90 % | High MFU; ≤ 5 % headroom on dense |
| Qwen 2.5-72B 4n | 89 % | Same as Llama-70B class |
| Qwen 2.5-32B 2n | 82 % | Already studied extensively in this repo; only ~+6 % from FA3+MBS=2 |
| Qwen 2.5-32B 4n | 80 % | Slightly more headroom (FP8 fits) but still capped at ~+11 % |
| Qwen3-32B 2n | 78 % | Same headroom as Qwen 2.5-32B but slightly less efficient attention; net no advantage |
| Mixtral 8x22B 8n | OOM | Needs ≥ 16 nodes — out of practical scope |
| Mixtral 8x7B 2n | OOM (any MP) | Won't fit at any legal MP with stock defaults |

---

## launch-mp.sh additions made during this work

- New env var `GIPFEL_EP` (expert-model-parallel-size).
- Per-model-tier `DEFAULT_EP`.
- Per-model-tier `NUM_EXPERTS` and `MOE_TOPK` (default 0 for dense).
- Validation: `NUM_EXPERTS % EP == 0`; `world == TP × PP × EP × DP`.
- New `MOE_ARGS` array activated when `NUM_EXPERTS > 0`.
- New model cases: `qwen3-14b`, `qwen3-32b`, `llama3-70b`, `qwen2.5-72b`,
  `mixtral-8x7b`, `mixtral-8x22b`.

## TODO

- Lock one of the two candidates and verify with 3 reruns (target ±1 %
  to establish the noise floor).
- For Candidate A, evaluate one optimization step (e.g., FA3 or FP8
  expert GEMM) to publish an initial "+X %" number against the baseline.
- (Optional) Wire MoE shared-expert support into the launcher to enable
  Qwen3-30B-A3B as a third MoE baseline alternative.
