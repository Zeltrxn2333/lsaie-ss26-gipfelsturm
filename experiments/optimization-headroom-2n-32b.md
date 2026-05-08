# 2-node 32B Qwen-2.5 — remaining optimization headroom

After the FA2 vs FA3 verification, this captures where we currently sit and
what's actually left worth trying. Written 2026-05-08.

## Where we are

| Metric | Value |
|--------|-------|
| Throughput | **2190 tok/s/GPU** (FA3, MBS=2, bf16, 2 nodes) |
| Compute | **432 TFLOP/s/GPU** ≈ **87 % of GH200 bf16 dense peak (494)** |
| Compute share of iter time | 99.7 % (from `GIPFEL_TIMING=2` profile) |
| Comm | DP grad RS + param AG fully hidden behind compute; TP RS+AG sync but on NVLink-C2C, ns-scale |

87 % MFU on bf16 is close to the practical ceiling (industry top-line is
89-92 %). The reason "compute, especially GEMM, is the only thing that
matters" is mechanical: 99.7 % of iter time is in `forward-compute +
backward-compute`, and within that the FFN + attention GEMMs are the bulk.
Everything else (TP collective, DP grad/param sync, optimizer step,
data-loader) sums to <0.5 % already-hidden time.

## Remaining levers, ranked by realistic gain

| Lever | Expected gain | Effort | Status |
|-------|---------------|--------|--------|
| **FP8 compute @ 2-node** | +5 ~ +14 %    | medium | untried at 2n; 4n result was +14 %, blocked at 2n by MBS=2+FP8 workspace OOM. Try **MBS=1 + FP8** specifically. |
| **CUDA graph capture** | +1 ~ +3 %     | low | `--enable-cuda-graph` (Megatron) untried; throughput-mode shapes are stable, ideal fit |
| **`--attention-backend auto`** | ~+2 %       | trivial | Switch from FA3 (`flash`) back to cuDNN FusedAttention; old data suggested cuDNN slightly beats FA3 on this shape |
| **`--tp-comm-overlap`** | +1 ~ +2 % | high | userbuffers config + TE injection; absolute ceiling small because NVLink-C2C is ns-scale already |
| **Rebuilt TE with smaller workspace** | indirect (frees HBM ⇒ enables FP8/larger MBS) | high (~1 day) | only worth it if FP8 still doesn't fit at 2n+MBS=1 |

### Confirmed non-levers (already tried or analysed away)

- **PP=2/4** — `num_microbatches / PP_stages` insufficient at 2 nodes, bubble
  ≥ 10 %. Only worth at ≥ 4 nodes.
- **MBS scaling** — M4/M8R explored earlier; MBS=2 sits at the GEMM-tile vs
  activation-memory sweet spot.
- **FSDP-style param sharding** — distributed-optimizer already does ZeRO-1.
  core_v0.16.1 has no clean param-shard mode; comm cost would offset.
- **Context parallelism** — seq_len=4096 too short; CP is for ≥ 32k.
- **Optimizer-step micro-opt** — 0.2 % of iter time, no point.
- **Attention algorithm** — FA3 vs cuDNN both saturate this shape; attention
  is one slice of the 99.7 % compute, not a primary lever.

## TODO — pick a defensible baseline

The next concrete step is **fix a single configuration as the reported
baseline** for the 2-node 32B Challenge-2 number, rather than continuing to
chase incremental knobs. Decision criteria:

1. **Reproducible**: documented sbatch + commit hash.
2. **Stable**: ≥ 3 reruns within ±1 % to rule out node-placement noise.
3. **Honest**: only knobs whose presence is justified by a measurable gain
   in our own numbers, not knobs cargo-culted from other reports.

Candidate baseline today: `MBS=2 + FA3 + bf16 + GIPFEL_USE_FA3=1`,
`./launch-mp.sh throughput 32b 20 2`. Verify with 3 same-config reruns to
establish noise floor, then lock that as the reference.

The three "worth-trying" levers above (FP8 @ MBS=1, CUDA graph, cuDNN
backend) can be evaluated once vs the locked baseline. If any gives a clean
≥ 3 % uplift across 3 reruns, fold it in; otherwise ship the bf16 baseline
and call it done.
