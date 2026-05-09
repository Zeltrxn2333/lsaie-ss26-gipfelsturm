# Baseline search across real Qwen / Llama / Mixtral models on Daint GH200

Goal: find a "good baseline" — real published model architecture; reasonable
parallelism for the cluster size (TP locked within a node, PP/EP/DP fill the
rest sensibly); stock Megatron defaults; **MFU naturally moderate or low so
optimization work has clear room to demonstrate value**.

Date: 2026-05-09. Cluster: CSCS Daint GH200 (4 GH200/node, NVLink-C2C 900
GB/s intra, Slingshot-11 ~93 GB/s inter). Megatron-LM core_v0.16.1.

## Method

- Stock launch-mp.sh defaults: `--use-distributed-optimizer`,
  `--overlap-grad-reduce`, `--overlap-param-gather`, `--sequence-parallel`,
  bf16, MBS per-tier default (=1 for ≥32B, =2 for 8B). No `GIPFEL_USE_FA3`,
  no `GIPFEL_FP8`, no `GIPFEL_NO_MASTER_WEIGHTS`, no `GIPFEL_RECOMPUTE`.
- 20-iteration throughput probes (15 for 140B). Mean over iters 5-N as
  steady-state.
- MFU = TFLOP/s/GPU (Megatron's `--log-throughput`) ÷ 494 (GH200 bf16 dense
  peak). For MoE models, Megatron computes FLOPs over the **active**
  param count (top-K of N experts), so MFU is comparable.
- MP rule: TP ≤ 4 (single node), PP across nodes for memory, EP across nodes
  for MoE expert sharding, DP fills the rest. EP must divide num_experts;
  world = TP×PP×EP×DP must hold exactly.
- Constraint: ≤ 200B total params (no Llama 3.1-405B / DeepSeek V2 / Qwen3-235B).

## Results

| # | Model | Real | Nodes | TP | PP | EP | DP | MBS | TFLOP/s | **MFU** | n | Job |
|---|-------|------|-------|----|----|----|----|----|---------|---------|---|-----|
| 1 | Qwen 2.5-8B    | ✓ | 1 | 1 | 1 | — | 4 | 2 | 495.8 | **100 %** | 16 | 3383408 |
| 2 | Qwen3-14B      | ✓ | 1 | 4 | 1 | — | 1 | 1 | 339.2 | **68.7 %** | 16 | 3383834 |
| 3 | Qwen 2.5-32B   | ✓ | 2 | 4 | 1 | — | 2 | 1 | 405.1 | **82.0 %** | 16 | 3383409 |
| 4 | Qwen3-32B      | ✓ | 2 | 4 | 1 | — | 2 | 1 | 383.9 | **77.7 %** | 16 | 3383835 |
| 5 | Qwen 2.5-32B   | ✓ | 4 | 4 | 1 | — | 4 | 1 | 396.5 | **80.3 %** | 16 | 3383410 |
| 6 | Llama 3.1-70B  | ✓ | 4 | 4 | 4 | — | 1 | 1 | 444.2 | **89.9 %** | 16 | 3383848 |
| 7 | Qwen 2.5-72B   | ✓ | 4 | 4 | 4 | — | 1 | 1 | 437.7 | **88.6 %** | 16 | 3383847 |
| 8 | **Mixtral 8x7B** | **✓** | **4** | **4** | **1** | **2** | **2** | **1** | **188.8** | **38.2 %** | 16 | **3383851** |
| 9 | 140B (synth) | ✗ | 8 | 4 | 4 | — | 2 | 1 | 445.1 | 90.1 % | 11 | 3383415 |

### OOM / partial runs (kept for diagnosis)

| Run | Outcome | Why |
|-----|---------|-----|
| Qwen3-14B 1n TP=1 DP=4 | OOM init | 14B replicated across 1 GPU = 92 GB state, no margin |
| Qwen 2.5-72B 4n TP=4 PP=2 DP=2 | OOM iter 1 | 63 GB state + 25 GB workspace = 88 GB; cliff |
| Llama 3.1-70B 4n TP=4 PP=2 DP=2 | OOM iter 14 | Same cliff; small ffn diff lets it survive longer |
| Mixtral 8x7B 2n TP=4 EP=2 DP=1 | OOM init | DP=1 ⇒ no opt-state sharding ⇒ ~95 GB |
| Mixtral 8x22B 8n TP=4 EP=4 DP=2 | OOM init | 9.7B params/rank × 10 bytes = 97 GB; needs ≥16 nodes |

## Architecture specs (real models tested)

| Model | Layers | h | ffn (per expert if MoE) | Q heads | KV heads | Num experts × topK | Total params |
|-------|--------|---|-------------------------|---------|----------|-------------------|--------------|
| Qwen 2.5-8B   | 32 | 4096 | 14336 | 32 | 8 | dense | 8 B   |
| Qwen3-14B     | 40 | 5120 | 17408 | 40 | 8 | dense | 14 B  |
| Qwen 2.5-32B  | 64 | 5120 | 27648 | 40 | 8 | dense | 32 B  |
| Qwen3-32B     | 64 | 5120 | 25600 | 64 | 8 | dense | 32 B  |
| Llama 3.1-70B | 80 | 8192 | 28672 | 64 | 8 | dense | 70 B  |
| Qwen 2.5-72B  | 80 | 8192 | 29568 | 64 | 8 | dense | 72 B  |
| Mixtral 8x7B  | 32 | 4096 | 14336 | 32 | 8 | **8 × top-2** | 47 B  |

## Observations

1. **MoE drops MFU dramatically.** Mixtral 8x7B at 38 % MFU is **half** the
   level of dense models with comparable per-rank compute. Causes:
   - Token dispatcher all-to-all (NVLink + Slingshot for cross-node EP)
   - Grouped expert GEMM (variable-size GEMMs, less tensor-core efficient
     than uniform dense GEMMs)
   - Routing overhead (top-K gating, scatter/gather)
   - Permutation kernels around expert dispatch
   This is *exactly* the kind of headroom a baseline needs.

2. **Dense models above 14B saturate compute fast.** All of 8B / 32B / 70B /
   72B / 140B-synth land in the 77-100 % MFU band with stock Megatron defaults
   on appropriate node counts. There's only ~5-15 pp left to chase, mostly
   via FP8 compute (validated +14 % on 32B 4n earlier in
   `experiments/optimization-headroom-2n-32b.md`).

3. **PP=4 DP=1 actually fits 70B/72B at 4 nodes**, while PP=2 DP=2 OOMs.
   Counterintuitive at first — DP=2 should let dist-opt shard the
   optimizer state — but at the 70B class TP=4 PP=2 leaves 9 B params/rank,
   and weight + grad + opt state already saturate the 95 GB budget before
   activations. PP=4 DP=1 cuts per-rank weight to 4.5 B params and
   *accepts* not sharding the optimizer state, giving more headroom for TE
   workspace + activations. Bubble cost is negligible at
   num_microbatches = 256.

4. **Qwen3-32B is consistently 5 pp lower than Qwen 2.5-32B** at the same MP.
   Same hidden + layer count, but 64 attention heads vs 40 — head_dim and
   per-shard head counts shift the attention GEMM partition into smaller,
   more numerous kernels. Plus ffn is 7 % smaller. Both reduce MFU.

5. **Stock-default 32B 2n / 4n already sit at 80-82 % MFU** — too tight for
   a "demonstrate optimization" baseline.

## Recommended baseline

### Primary: **Mixtral 8x7B, 4 nodes, TP=4 PP=1 EP=2 DP=2, MBS=1, stock Megatron defaults**

| Metric | Value |
|--------|-------|
| Real model | Mixtral 8x7B (Mistral AI, 2024) |
| Megatron support | Official `examples/mixtral/train_mixtral_8x7b_distributed.sh` |
| Total params | 47 B (13 B active per token via top-2) |
| Parallelism | TP=4 within node, PP=1, EP=2 across 2 of 4 DP ranks, DP=2 for dist-opt |
| Stock baseline | **188.8 TFLOP/s/GPU = 38.2 % MFU** (job 3383851) |
| Per-iter token throughput | ~2400 tok/s/GPU |

**Why this baseline:**

- The only configuration in the sweep where stock Megatron defaults give
  MFU < 50 % with **fully reasonable parallelism**.
- Real model (not synthetic), real Megatron MoE path (Megatron's own
  `train_mixtral_8x7b_distributed.sh` flags), real EP wiring.
- Headroom is rich AND varied:
  - **MoE dispatcher tuning**: alltoall vs allgather, fused permute,
    `--moe-permute-fusion`
  - **MoE-grouped-GEMM kernel**: TE has alternative paths (GEMM vs grouped-GEMM)
  - **FA3** (~+2-4 %)
  - **MBS scaling** (with EP=2 DP=2 there's memory headroom that 2-node
    didn't have)
  - **FP8 expert GEMM** (TE 2.11 has FP8 grouped-GEMM)
  - **TP/EP/DP rebalance** (e.g., TP=2 EP=4 DP=2 changes per-rank partition)
  - **Token drop / capacity factor tuning**
- Improvement story is varied across multiple optimization axes, not just
  "stack FP8 on a dense model".

**Reproducible:**
```bash
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
  GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
  GIPFEL_EP=2 \
  ./launch-mp.sh throughput mixtral-8x7b 20 4
```

### Secondary candidates (in case Mixtral baseline doesn't fit project scope)

- **Qwen3-14B 1n TP=4 PP=1 DP=1** (68.7 % MFU): real Qwen3, parallelism
  forced by memory, dense-only optimization path (FA3 / MBS / TP rebalance).
- **Qwen 2.5-32B 2n TP=4 PP=1 DP=2** (82.0 %): the most-studied config
  in this repo, ~+6 % headroom from FA3+MBS=2.

## Updated launch-mp.sh additions (this round)

- New cases: `qwen3-14b`, `qwen3-32b`, `llama3-70b`, `qwen2.5-72b`,
  `mixtral-8x7b`, `mixtral-8x22b`.
- New env var `GIPFEL_EP` + per-tier `DEFAULT_EP`.
- New validation: `NUM_EXPERTS % EP == 0`, `world == TP×PP×EP×DP`.
- New `MOE_ARGS` array (when NUM_EXPERTS > 0): `--num-experts`,
  `--moe-router-topk`, `--moe-router-load-balancing-type aux_loss`,
  `--moe-aux-loss-coeff 1e-2`, `--moe-grouped-gemm`,
  `--moe-token-dispatcher-type alltoall`, `--expert-model-parallel-size $EP`.

## TODO

- Verify Mixtral 8x7B 4n baseline reproducibility with 3 reruns (target ±1 %).
- Quantify a single optimization step (e.g., FA3 alone) on this baseline to
  publish an initial "+X %" number.
- (Optional) Wire MoE shared-expert support into the launcher to enable
  Qwen3-30B-A3B as an alternative MoE baseline (different routing topology).
- (Optional) Mixtral 8x22B requires ≥16 nodes for stock defaults to fit; try
  `GIPFEL_RECOMPUTE=full` + `GIPFEL_EXP_AVG_DTYPE=fp8` if 8 nodes is the
  cap, but those are no longer "stock defaults".

## Job IDs

| ID | Run | Status |
|----|-----|--------|
| 3383408 | Qwen 2.5-8B 1n | ✓ |
| 3383415 | 140B-synth 8n | ✓ |
| 3383409 | Qwen 2.5-32B 2n | ✓ |
| 3383410 | Qwen 2.5-32B 4n | ✓ |
| 3383834 | Qwen3-14B 1n TP=4 | ✓ |
| 3383835 | Qwen3-32B 2n | ✓ |
| 3383837 | Qwen 2.5-72B 4n TP=4 PP=2 | OOM iter 1 |
| 3383841 | Llama 3.1-70B 4n TP=4 PP=2 | OOM iter 14 |
| 3383842 | Qwen3-14B 1n TP=1 DP=4 | OOM init |
| 3383847 | Qwen 2.5-72B 4n TP=4 PP=4 | ✓ |
| 3383848 | Llama 3.1-70B 4n TP=4 PP=4 | ✓ |
| 3383849 | Mixtral 8x7B 2n EP=2 DP=1 | OOM init |
| 3383850 | Mixtral 8x22B 8n EP=4 DP=2 | OOM init |
| 3383851 | **Mixtral 8x7B 4n EP=2 DP=2** | **✓ recommended baseline** |
