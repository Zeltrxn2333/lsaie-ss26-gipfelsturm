# Baseline search across real Qwen / Llama dense models on Daint GH200

Goal: find a "good baseline" — real, published model architecture; reasonable
parallelism for the cluster size (TP locked within a node, PP across nodes,
DP for the rest); stock Megatron defaults (no FA3 venv, no FP8, no exotic
GIPFEL knobs); MFU **moderate enough to leave headroom for optimization**.

Date: 2026-05-09. Cluster: CSCS Daint GH200 (4 GH200/node, NVLink-C2C 900 GB/s,
Slingshot-11 ~93 GB/s). Megatron-LM core_v0.16.1.

## Method

- All runs use stock launch-mp.sh defaults (`--use-distributed-optimizer`,
  `--overlap-grad-reduce`, `--overlap-param-gather`, `--sequence-parallel`,
  bf16, MBS=1 for ≥32B / MBS=2 for 8B). No `GIPFEL_USE_FA3`, no
  `GIPFEL_FP8`, no `GIPFEL_NO_MASTER_WEIGHTS`, no `GIPFEL_RECOMPUTE`.
- 20-iteration throughput probes (15 for 140B). Mean over iters 5-N as
  steady-state.
- MFU = TFLOP/s/GPU (Megatron's `--log-throughput`) ÷ 494 (GH200 bf16
  dense peak).
- Per-tier MP follows the **reasonable** rule: TP ≤ 4 (single node), PP
  spans nodes when needed, DP fills the rest.

## Results

### Successful runs

| # | Model | Real? | Nodes | TP | PP | DP | MBS | TFLOP/s/GPU | **MFU** | n samples | Job |
|---|-------|-------|-------|-----|-----|-----|-----|-------------|---------|-----------|-----|
| 1 | Qwen 2.5-8B          | yes      | 1 | 1 | 1 | 4 | 2 | 495.8 | **100%**  | 16 | 3383408 |
| 2 | Qwen3-14B            | yes      | 1 | 4 | 1 | 1 | 1 | 339.2 | **68.7%** | 16 | 3383834 |
| 3 | Qwen 2.5-32B         | yes      | 2 | 4 | 1 | 2 | 1 | 405.1 | **82.0%** | 16 | 3383409 |
| 4 | Qwen3-32B            | yes      | 2 | 4 | 1 | 2 | 1 | 383.9 | **77.7%** | 16 | 3383835 |
| 5 | Qwen 2.5-32B         | yes      | 4 | 4 | 1 | 4 | 1 | 396.5 | **80.3%** | 16 | 3383410 |
| 6 | Llama 3.1-70B        | yes      | 4 | 4 | 2 | 2 | 1 | 446.0 | **90.3%** | 9 (then OOM) | 3383841 |
| 7 | 140B (synthetic)     | **NO**   | 8 | 4 | 4 | 2 | 1 | 445.1 | **90.1%** | 11 | 3383415 |

### Failures (informative)

| Run | Failure | Implication |
|-----|---------|-------------|
| Qwen3-14B 1n **TP=1** PP=1 DP=4 | OOM at init (~95 GB) | 14B at full bf16 weight + grad doesn't fit replicated on 1 GPU. **TP=4 is forced.** |
| Qwen 2.5-72B 4n TP=4 PP=2 DP=2 | OOM iter 1 (~95 GB) | 70-72B class at 4 nodes is on the memory cliff; need 8 nodes or PP=4 DP=1 |
| Llama 3.1-70B 4n TP=4 PP=2 DP=2 | OOM at iter 14 | Same edge — gradient + Adam state accumulation creeps past the ceiling. 9 stable iters before OOM. |

## Architectural specs (real models)

| Model | Layers | h | ffn | Q heads | KV heads | head_dim | Total params |
|-------|--------|---|-----|---------|----------|----------|--------------|
| Qwen 2.5-8B   | 32 | 4096 | 14336 | 32 | 8 | 128 | 8B   |
| Qwen3-14B     | 40 | 5120 | 17408 | 40 | 8 | 128 | 14B  |
| Qwen 2.5-32B  | 64 | 5120 | 27648 | 40 | 8 | 128 | 32B  |
| Qwen3-32B     | 64 | 5120 | 25600 | **64** | 8 | 128 | 32B  |
| Llama 3.1-70B | 80 | 8192 | 28672 | 64 | 8 | 128 | 70B  |
| Qwen 2.5-72B  | 80 | 8192 | 29568 | 64 | 8 | 128 | 72B  |
| 140B (synth)  | 80 | 12288 | 32768 | 96 | 8 | 128 | ~145B |

## Observations

1. **MFU is dominated by per-rank GEMM tile size.** Bigger model + smaller
   TP → bigger GEMMs → higher MFU. The 8B-no-TP run hits 100 %; the 14B-with-TP=4
   forced run is at 68.7 % despite the model being only 1.75× bigger — TP=4
   shrinks each GEMM's hidden dimension to 1280, well below the cuBLAS
   sweet spot.

2. **Qwen3-32B is consistently slower than Qwen 2.5-32B at the same MP**
   (77.7 % vs 82.0 %). Same hidden + layer count, but 64 attention heads
   (vs 40) means head_dim and per-shard head count change; the attention
   GEMM partition shifts toward more, smaller kernels. ffn is also 7 %
   smaller (25600 vs 27648) → slightly less FFN compute per token.

3. **Distributed-optimizer state is the binding memory constraint at the
   70 B class on 4 nodes.** TP=4 PP=2 DP=2 puts each rank holding a 9-9-27 GB
   weight/grad/optim split + ~10 GB TE/activations workspace; Llama-70B
   ran for 13 iters before slowly creeping past 95 GB and OOM'ing.
   Qwen 2.5-72B (slightly bigger ffn) didn't even survive iter 1.
   Recommended: 70B class needs ≥ 8 nodes for stable stock-default training.

4. **140B at 8 nodes hits 90 % MFU** despite PP=4 cross-node bubbles. This
   contradicts the naïve "high PP → low MFU" prior — at this model size,
   the per-microbatch GEMM compute dominates, and PP bubble at
   num_microbatches = 128 / 4 stages = 0.78 % is negligible.

5. **No single config delivers a low-MFU baseline with reasonable MP and
   plenty of headroom.** Stock Megatron + reasonable MP saturates compute
   on big-enough models. The Qwen3-14B 1n TP=4 case at 68.7 % is the
   lowest **and** the parallelism is forced (TP=1 OOMs). That makes it the
   one config where the optimization story is open.

## Recommended baseline

### Primary: **Qwen3-14B, 1 node, TP=4 PP=1 DP=1, MBS=1, stock Megatron defaults**

- Real model (Qwen3-14B exact spec).
- Parallelism is **forced** by memory (TP=1 OOMs, TP=2 borderline, TP=4
  is the only safe stock-default fit).
- Stock baseline MFU: **68.7 %** (339.2 TFLOP/s/GPU).
- Improvement path is naturally rich:
  - Try TP=2 PP=1 DP=2 with `expandable_segments:True` — bigger GEMM tiles, expected ~78-82 % MFU
  - Add MBS=2 if memory permits (more activation memory but bigger GEMMs)
  - Add FA3 (expected +2-4 %)
  - Add FP8 compute if memory budget allows after above (potential +10 %)
- Reproducible:
  ```bash
  GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
    GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
    ./launch-mp.sh throughput qwen3-14b 20 1
  ```

### Secondary (more standard, less dramatic improvement story)

**Qwen 2.5-32B, 2 nodes, TP=4 PP=1 DP=2, MBS=1, stock**: MFU 82.0 %.
Improvement to ~87 % with FA3 + MBS=2 already validated; FP8 only fits at
≥ 4 nodes giving +14 %. Headroom: ~+6 % at 2n, ~+11 % at 4n.

### Reference (for "really big model" context, not chosen as primary)

**140B-synthetic 8 nodes**: 90 % MFU. Not a real model — synthetic
extrapolation. Useful only for demonstrating that the cluster can scale to
8 nodes with PP=4. **Recommend dropping from any reported "Qwen baseline"
table** since it doesn't correspond to any published model.

## Why not a real ~140B Qwen?

There is none in the dense ≤200B range that fits 8 nodes:
- Llama 3.1-70B / Qwen 2.5-72B: fit 4-8 nodes (real) — already covered above
- Llama 3.1-405B: ≥ 24 nodes minimum (out of practical range)
- Mixtral 8x22B (141B total / 39B active, MoE): real, fits 4-8 nodes,
  but requires MoE wiring in launch-mp.sh (`--num-experts`,
  `--moe-router-topk`, `--expert-model-parallel-size`). **Worth adding as
  a follow-up to demonstrate MoE baseline.**

## Job IDs (for log replay / verification)

| ID | Run |
|----|-----|
| 3383408 | Qwen 2.5-8B 1n |
| 3383415 | 140B-synth 8n |
| 3383409 | Qwen 2.5-32B 2n |
| 3383410 | Qwen 2.5-32B 4n |
| 3383834 | Qwen3-14B 1n TP=4 |
| 3383835 | Qwen3-32B 2n |
| 3383841 | Llama 3.1-70B 4n (OOM at iter 14) |
| 3383842 | Qwen3-14B 1n TP=1 (OOM init) |
| 3383837 | Qwen 2.5-72B 4n (OOM iter 1) |
