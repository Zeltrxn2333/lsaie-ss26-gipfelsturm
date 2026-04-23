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

## Patched attempts (H1–H3)

| # | Config | Result | Usage at OOM | Init time |
|---|--------|--------|--------------|-----------|
| H1 | patch + fp8 Adam                          | OOM iter 1 bwd | 94.86 GB | **3.6 s** (was 16 s; patch works) |
| H2 | H1 + recompute full                       | OOM iter 1 bwd | 94.86 GB | 3.6 s |
| H3 | H2 + no-overlap-GR                        | in progress   | —        | — |

**Patch is confirmed active** (`no_master_weights=True` in optimizer config
log, model-and-optimizer-setup went from 16 s → 3.6 s). Peak memory at OOM
didn't drop, suggesting the init savings are consumed by training-time
buffers during the first iteration. Still under investigation.

## Open questions (next iterations)

- Verify whether TE FusedAdam actually skipped allocating master when
  `master_weights=False` — the init time dropped but OOM cliff didn't.
  Possibly TE lazily allocates master on first step.
- Test with `GIPFEL_CPU_OFFLOAD=0.3` on top of the patch (move remaining
  optimizer state to CPU, since the patch saved master but kept m, v).
- Reduce `SEQ_LEN` from 4096 to 2048 — halves activation memory.

## Throughput at ≥2 nodes (fallback fully working)

| Config | Nodes | tok/s/GPU |
|--------|-------|-----------|
| 2n bf16 default    | 2 | 2038 |
| 4n bf16 default    | 4 | 2024 |
| **4n + FP8**       | 4 | **2311** (+14 %) |
