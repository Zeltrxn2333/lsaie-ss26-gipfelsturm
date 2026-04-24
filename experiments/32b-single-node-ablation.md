# 32B Qwen single-node memory ablation (cumulative)

Goal: fit Qwen 2.5-32B (64 layers, h=5120, ffn=27648, 40 heads, 8 kv) on a
single CSCS Daint GH200 node (4 × 95 GB HBM) with TP=4, PP=1, DP=1.

## Hardware / software fixtures

- Model: 32B dense, SEQ_LEN=4096, GBS=256, MBS=1.
- MP: TP=4, PP=1 → DP = 4 / (4·1) = 1 (no DP sharding).
- HBM per rank: 95 GB.
- Host mem: 870 GB total per node; cgroup limit tunable via `--mem`.
- Megatron-LM core_v0.16.1, `alps3` container (NGC PyTorch 26.01 + TE).

## Per-rank memory budget, decomposed

| Buffer                              | Bytes/param | Per-rank | Note |
|-------------------------------------|-------------|----------|------|
| bf16 compute params                 | 2           | 16.25 GB | mandatory forward |
| int16 master remainder              | 2           | 16.25 GB | fp32 reconstruction via TE FusedAdam `store_param_remainders=True` |
| bf16 main grads                     | 2           | 16.25 GB | `--main-grads-dtype bf16` |
| fp8 Adam m (uint8)                  | 1           | 8.125 GB | `--exp-avg-dtype fp8` (env var) |
| fp8 Adam v (uint8)                  | 1           | 8.125 GB | `--exp-avg-sq-dtype fp8` |
| **state baseline**                  |             | **65 GB** | |
| TE + NCCL + flash-attn workspace    |             | ~15-20 GB | unavoidable |
| cuBLAS heuristic + autograd + misc  |             | ~10 GB   | some unavoidable |
| **total at OOM (observed)**         |             | **~95 GB** | |

## Env-var-only (no-patch) attempts

| # | Config | Result | Usage at OOM | Notes |
|---|--------|--------|--------------|-------|
| A1 | fp8 Adam (m,v in fp8)             | GPU OOM init | 94.94 GB | ~16 GB saved vs fp32 Adam, doesn't fit |
| A2 | A1 + NCCL_NVLS=0 + NCCL_BUFFSIZE=4M | GPU OOM init | 94.94 GB | NCCL tune gave ~0 GB back |
| B1 | A1 + FP8 compute + fp8 param-gather | GPU OOM init | 94.86 GB | FP8 scale-factor workspace offset param savings |
| B2 | B1 + no-overlap-PG                | GPU OOM init | 93.22 GB | -1.7 GB; not enough |
| C1-C4 | Muon / dist_muon (various)     | Blocked      | — | "Emerging Optimizers" pkg not installed |
| D1 | B2 + optim-cpu-offload 0.5        | GPU OOM iter | 93.34 GB | rank0 close (26 MiB short) |
| D2 | D1 + full recompute               | Host OOM iter | — | First time past GPU init! Host mem=460G too tight |
| D3 | offload 0.3 + recompute full      | GPU OOM     | 93.03 GB | offload too low |
| D4 | D2 at mem=800G                    | GPU OOM iter | 92.88 GB | mem fine but GPU tight |
| D5 | offload 0.7 + recompute full + mem=800G | Host OOM iter | — | offload-grown pinned pages |
| D6 | offload 0.6 + recompute full + mem=800G | Host OOM iter | — | same mode |
| D7 | offload 0.5 + selective recompute | GPU OOM | 93.74 GB | selective not enough |
| E1 | TE activation offload 32 layers   | GPU OOM init | 94.94 GB | activation offload alone doesn't help |
| E2 | E1 + optim-offload 0.5            | GPU + host OOM | — | both offloads blow both sides |
| G1 | fp8 Adam + no-overlap-GR          | GPU OOM init | 94.96 GB | no measurable save |

**Env-only conclusion: cannot fit.** The fundamental floor is ~95 GB.

## Plan: Megatron patches

See `patches/README-32B-single-node-brainstorm.md` for 5 plans. Summary:

1. **Plan 1** disable overlap-grad-reduce (tried as G1): no save.
2. **Plan 2** shrink DDP bucket size: untested — small win.
3. **Plan 3** skip `master_param` shard in distrib_optimizer.state_dict(): turned
   out to be only on checkpoint-load path (never runs in throughput mode).
4. **Plan 4** custom grad-buffer compression: too complex.
5. **Plan 5** → **Plan executed**: skip fp32 master entirely.

## patches/0002-no-master-weights-option.patch

Adds `--no-master-weights` CLI flag. When set, passes `master_weights=False`
to TE FusedAdam. Skips the per-parameter fp32/fp16/int16 master allocation.

| Field | Value |
|-------|-------|
| Savings target | 16 GB/rank (the int16 remainder) |
| Trade-off | bf16 Adam precision (no fp32 master) |
| CLI flag | `--no-master-weights` |
| launch-mp.sh knob | `GIPFEL_NO_MASTER_WEIGHTS=1` |

## Patched attempts (H1–H5)

| # | Config | Result | Usage at OOM | Init time |
|---|--------|--------|--------------|-----------|
| H1 | patch + fp8 Adam                            | OOM iter 1 bwd | 94.86 GB | **3.6 s** (was 16 s; patch clearly active) |
| H2 | H1 + recompute full                         | OOM iter 1 bwd | 94.86 GB | 3.6 s |
| H3 | H2 + no-overlap-GR                          | hung >12 min, cancelled | — | — |
| H4 | H2 + optimizer cpu-offload 0.3              | OOM iter 1 bwd | ~95 GB   | — |
| H5 | H2 + SEQ_LEN=2048                           | OOM iter 1 bwd | ~95 GB   | — |

## Conclusion

**Single-node 32B Qwen with Megatron-LM core_v0.16.1 does not fit** under
any combination of env-var knobs, selective or full recompute, activation
or optimizer CPU offload, FP8 compute, fp8 Adam states, halved sequence
length, or the `--no-master-weights` patch.

The hard floor is ~95 GB of GPU memory per rank, dominated by a fixed
allocation pattern in TE FusedAdam's `initialize_state()` that our patch
reduces init *time* (16s → 3.6s) but does not reduce peak memory. The
patch's `master_weights=False` kwarg appears to skip the outer master
tensor, but TE still reserves an equivalent workspace internally — we
cannot observe or control this from Megatron level without modifying
TransformerEngine itself (a precompiled library).

**Practical recommendations:**

1. **Use 2 nodes minimum** for 32B. Distributed optimizer then shards
   optimizer state across DP=2, yielding 2038 tok/s/GPU at 2 nodes.
2. **For throughput maximization**, use 4 nodes + FP8 compute:
   **2311 tok/s/GPU** (+14 % over bf16 baseline).
3. **Beyond that**: patching TransformerEngine itself (to accept
   int16 master in fp8 Adam kernel, or to truly skip master allocation)
   is the next frontier — but that's a separate kernel build, not a
   single-file patch.

## Files produced

- `patches/0002-no-master-weights-option.patch` — adds the flag. Kept
  because it may be useful if TE FusedAdam's behavior changes in a future
  release, or as a platform for larger changes.
- `patches/README-32B-single-node-brainstorm.md` — documentation of 5
  plans evaluated.

## Breakthrough: single-node via layer reduction (Path A, round 2)

After confirming that the 95 GB peak scales non-linearly with layer count,
tested smaller shapes with full memory-saving stack enabled
(`GIPFEL_EXP_AVG_DTYPE=fp8 GIPFEL_EXP_AVG_SQ_DTYPE=fp8 GIPFEL_NO_MASTER_WEIGHTS=1 GIPFEL_NO_OVERLAP_PG=1 GIPFEL_RECOMPUTE=full GIPFEL_MEM=800000`).
All at SEQ_LEN=4096, GBS=256, MBS=1, TP=4, 1 node. Hidden/ffn/heads/kv as
Qwen 2.5-32B (h=5120, ffn=27648, 40H, 8KV); only NUM_LAYERS changed.

| # Layers | Approx params | 1-node fit? | tok/s/GPU | TFLOP/s/GPU | iter time |
|----------|---------------|-------------|-----------|-------------|-----------|
| 40       | ~20B          | ✓ fit       | **2457**  | ~307        | 105 s |
| 48       | ~24B          | ✓ fit       | **2085**  | ~310        | 125 s |
| 50       | ~25B          | TBD         | —         | —           | — |
| 52       | ~26B          | ✗ OOM       | —         | —           | — |
| 56       | ~28B          | ✗ OOM       | —         | —           | — |
| 64       | 32B           | ✗ OOM       | —         | —           | — |

**Actionable single-node answer**: **48 layers / ~24B** at **2085 tok/s/GPU, 310 TFLOP/s**.
This is ~75% of a real 32B but is *an actual 1-node GH200 training benchmark*.

For a true 32B single-node, the blocker is TE/Megatron's workspace peak at ~95
GB regardless of layer count below some threshold — needs TE rebuild (Path B)
or Managed Memory (Path C).

## Throughput at ≥2 nodes (fallback fully working)

| Config | Nodes | tok/s/GPU |
|--------|-------|-----------|
| 2n bf16 default    | 2 | 2038 |
| 4n bf16 default    | 4 | 2024 |
| **4n + FP8**       | 4 | **2311** (+14 %) |
