# Brainstorm: Megatron modifications for 32B Qwen single-node

## Problem recap

32B Qwen (Qwen 2.5-32B: 64 layers, h=5120, ffn=27648, 40 heads, 8 kv, 64-layer)
on a **single GH200 node** (4 × 95 GB HBM) with TP=4, PP=1, DP=1 will not fit
under any combination of env-var knobs accessible in the launcher. 14 attempts
(A1–D7, E1–E2) all OOM at 93–95 GB per rank.

Observed per-rank baseline with the best env-var config
(`fp8 Adam m,v` + `fp8 param-gather` disabled + `no-overlap-param-gather`):

| Buffer | Bytes/param | Per-rank | Patchable? |
|--------|-------------|----------|------------|
| bf16 compute params                | 2 | 16.25 GB | Yes — fp8 param gather |
| int16 master remainder             | 2 | 16.25 GB | Yes — skip if redundant |
| bf16 main grads                    | 2 | 16.25 GB | Yes — smaller bucket / no overlap |
| fp8 Adam m (uint8)                 | 1 | 8.125 GB | Already done |
| fp8 Adam v (uint8)                 | 1 | 8.125 GB | Already done |
| **Sum (state)**                    |   | **65 GB** | |
| TE kernels + NCCL + CUDA + workspace |   | ~30 GB | Partial |
| **Total at OOM**                   |   | **~95 GB** | |

To fit with workspace, the state needs to drop by ~5-10 GB.

## Plans

### Plan 1 — Disable `--overlap-grad-reduce` (LAUNCHER ONLY, no Megatron patch)

Megatron's `--overlap-grad-reduce` keeps a full-size grad **double-buffer**
(one bucket being reduced while the next accumulates). On 32B TP=4 the grad
buffer is ~16 GB/rank; the double buffer doubles that transient to ~32 GB.
Disabling it keeps a single rolling buffer.

- **Estimated savings: 8–16 GB/rank** at cost of ~5–10 % throughput (less DP overlap).
- **Effort: 5 lines in `launch-mp.sh` + env var.**
- **Risk: low** — supported flag in Megatron, just not the default.

### Plan 2 — Shrink `--ddp-bucket-size` (LAUNCHER ONLY)

Default bucket is 40 MB. Each bucket needs matched buffers for allgather/
allreduce. Smaller buckets shave NCCL+Megatron workspace modestly.

- **Estimated savings: 2–4 GB/rank.**
- **Effort: 1 line in `launch-mp.sh`.**
- **Risk: lower NCCL effective bandwidth (~10 % collective regression on DP=1
  this doesn't matter, since no DP collectives happen).**

### Plan 3 — Megatron patch: skip duplicate `master_param` shard allocation

`megatron/core/optimizer/distrib_optimizer.py:800-806` allocates an int16
`master_param` tensor per-param in `state_dict_state`. TE FusedAdam **also**
owns the remainder internally when `store_param_remainders=True`. The outer
shard appears to be a duplicate kept for checkpoint save/load paths.

Patch plan: when `store_param_remainders=True` and no checkpoint load is
requested, skip allocating this tensor; reconstruct it only at checkpoint
save time from TE's internal state.

- **Estimated savings: 16 GB/rank if truly redundant.**
- **Effort: ~10 lines in `distrib_optimizer.py` + verification that checkpointing
  still works.**
- **Risk: medium** — if not redundant, breaks training; if checkpoint logic
  couples tightly, breaks save/load.

### Plan 4 — Megatron patch: smaller grad buffer via `grad_reduce_in_fp32=True` OFF + microbatch-wise grad accumulation

Megatron's gradient buffer holds one full-size accumulation tensor per rank
(~16 GB bf16). With careful restructuring, gradient can be reduced per
microbatch into a smaller buffer, trading bandwidth for space.

- **Estimated savings: 8–12 GB/rank.**
- **Effort: significant (~100+ lines), touches DDP internals.**
- **Risk: high.**

### Plan 5 — Megatron patch: implement bf16 master + bf16 Adam (drop fp32 master entirely)

Qualitatively different path: do Adam updates entirely in bf16 with no fp32
recovery. Risks training instability in long runs but fine for "does it fit"
benchmarks.

- **Estimated savings: 16 GB (master) + potential 16 GB more if exp_avg stays
  fp16 but recovered via low-rank compression.**
- **Effort: large (~200 LoC), touches TE FusedAdam integration.**
- **Risk: high (training stability).**

## Execution order

1. **Plan 1** first — zero patch, biggest free win. Try immediately.
2. **Plan 2** stacked on top — small win, trivial to add.
3. If still OOM: **Plan 3** — first actual patch. Test carefully (no checkpoint
   involved in throughput mode, so save/load isn't exercised; safe to try).
4. Plan 4 / 5 only if 1-3 don't fit.

## Success criterion

Run `./launch-mp.sh throughput 32b 15 1` with `GIPFEL_ACCOUNT=lp160` and
reach iter 5+ without OOM. Stretch: achieve reasonable throughput
(≥ 500 tok/s/GPU — bounded by 256 microbatches at DP=1).
