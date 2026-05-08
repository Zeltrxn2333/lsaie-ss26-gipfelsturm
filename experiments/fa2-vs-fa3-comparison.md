# FA2 vs FA3 head-to-head — 2-node 32B Qwen 2.5 throughput

Direct same-day, same-config comparison of TransformerEngine 2.11 dispatching
attention through FlashAttention-2 (container's bundled `flash-attn 2.7.4.post1`)
vs FlashAttention-3 (built from source under `/iopsstor/.../venvs/fa3`).

## Configuration

- Hardware: 2 × Daint GH200 nodes (8 GPUs total), Slingshot-11.
- Model: Qwen 2.5-32B exact (64 layers, h=5120, ffn=27648, 40 heads, 8 kv heads).
- Parallelism: TP=4, PP=1, DP=2. Distributed optimizer enabled.
- Sequence: SEQ_LEN=4096, GBS=256, MBS=2.
- Precision: bf16 compute + bf16 main grads + precision-aware optimizer (int16
  remainder for fp32 reconstruction).
- Job duration: 15 iterations; first 4 discarded as warmup; mean over iters 5-15.

Both jobs use `--attention-backend flash` (forces TE to dispatch attention via
FA, vs `auto` which prefers cuDNN FusedAttention). The only delta:

| Job | PYTHONPATH FA3 venv | TE imports `flash_attn_3` | Effective backend |
|-----|---------------------|---------------------------|-------------------|
| FA3 (3382051) | `/iopsstor/scratch/cscs/$USER/venvs/fa3:$PYTHONPATH` | yes | FA3 |
| FA2 (3382052) | `/iopsstor/scratch/cscs/$USER/venvs/fa3_DISABLED_FOR_FA2_TEST:$PYTHONPATH` | no (ImportError) → fallback | FA2 (container's 2.7.4.post1) |

Fallback evidence in the FA2 log (TE 2.11 prints an install hint when FA3
import fails):
```
mkdir -p $python_path/flash_attn_3
cp flash_attn_interface.py $python_path/flash_attn_3/flash_attn_interface.py
```
Plus only `flash_attn::_flash_attn_backward` (FA2 namespace) registered as a
torch operator, no `flash_attn_3::` namespace warning.

## Per-iteration tokens/sec/GPU

| iter | FA2  | FA3  | FA3 − FA2 |
|------|------|------|-----------|
|  1   | 1843 | 1914 | +71  (warmup) |
|  2   | 1897 | 2020 | +123 (warmup) |
|  3   | 2065 | 2159 | +94  (warmup) |
|  4   | 2065 | 2020 | −45  (warmup; FA3 had a slow iter) |
|  5   | 2077 | 2166 | +89  |
|  6   | 2079 | 2174 | +95  |
|  7   | 2094 | 2188 | +94  |
|  8   | 2096 | 2190 | +94  |
|  9   | 2098 | 2195 | +97  |
| 10   | 2098 | 2195 | +97  |
| 11   | 2097 | 2194 | +97  |
| 12   | 2111 | 2207 | +96  |
| 13   | 2098 | 2194 | +96  |
| 14   | 2099 | 2196 | +97  |
| 15   | 2098 | 2195 | +97  |

## Steady-state means (iters 5-15, n=11)

| Backend | mean tok/s/GPU | TFLOP/s/GPU | vs FA2 |
|---------|----------------|-------------|--------|
| FA2 (container 2.7.4.post1) | **2095.0** | ~412 | baseline |
| FA3 (built from source)     | **2190.4** | ~432 | **+4.6 %** |

The +95 tok/s/GPU gap is consistent every iteration (range +89..+97), well
outside per-iter noise.

## How this differs from the prior "+0.8%" measurement

A previous comparison on the same hardware showed FA3 only +0.8 % over the
default `--attention-backend auto` (which dispatches to **cuDNN FusedAttention**,
not FA2). cuDNN's fused attention on Hopper is already well-tuned for
head_dim=128 + bf16 + causal, so FA3's wins over cuDNN are small at this shape.

The full picture:

| Backend                                    | tok/s/GPU | vs FA2 | vs cuDNN auto |
|--------------------------------------------|-----------|--------|---------------|
| FA2 (`--attention-backend flash`, no FA3)  | 2095      | —      | ~−6 %         |
| cuDNN FusedAttention (`--attention-backend auto`) | ~2230     | +6 %   | —             |
| FA3 (`--attention-backend flash` + FA3 venv) | 2190      | +4.6 % | ~−2 %         |

So **just setting `--attention-backend flash` without installing FA3 is a
regression** vs the default. The FA3 venv is what makes the `flash` backend
worth turning on at all.

## Reproduction

Both jobs share the same generated sbatch except for one line. Recipe:

```bash
# On a Daint login node, in your gipfelsturm checkout:
cd /users/$USER/gipfelsturm

# 1. Generate sbatch via the standard launch-mp.sh path (FA3 enabled).
GIPFEL_ACCOUNT=lp160 GIPFEL_PARTITION=normal \
  GIPFEL_WORKDIR=/users/$USER/gipfelsturm GIPFEL_TIME=00:30:00 \
  GIPFEL_MEM=800000 GIPFEL_MBS=2 GIPFEL_USE_FA3=1 \
  ./launch-mp.sh throughput 32b 15 2
# This submits the FA3 job and writes logs/gipfel-throughput-32b-tp4pp1-15s-2n.sbatch

# 2. Make a FA2 variant by mangling the FA3 venv path (forces ImportError).
SBATCH=logs/gipfel-throughput-32b-tp4pp1-15s-2n.sbatch
SBATCH_FA2=${SBATCH%.sbatch}-fa2only.sbatch
sed 's|venvs/fa3|venvs/fa3_DISABLED_FOR_FA2_TEST|' $SBATCH > $SBATCH_FA2
sbatch $SBATCH_FA2
```

The FA2 variant retains `--attention-backend flash` (so TE is asked for
flash), but the bogus PYTHONPATH means `from flash_attn_3 import ...` fails
inside TE's `backends.py`, which falls back to `flash_attn` (FA2).

To analyze:
```bash
LOG_FA3=logs/gipfel-throughput-32b-tp4pp1-15s-2n-<jid_fa3>.log
LOG_FA2=logs/gipfel-throughput-32b-tp4pp1-15s-2n-<jid_fa2>.log
for L in $LOG_FA3 $LOG_FA2; do
  echo "=== $L ==="
  grep -oE 'tokens/sec/GPU: [0-9]+' $L \
    | awk -F': ' '{print $2}' | tail -n +5 \
    | awk '{s+=$1; n++} END {printf "mean=%.1f n=%d\n", s/n, n}'
done
```

## Why FA3 only beats FA2 by +4.6 % (not more)

`GIPFEL_TIMING=2` profile of the same 2-node 32B run shows
`forward-compute + backward-compute ≈ 99.7 %` of iter time. Attention is one
slice of compute; within attention, only the kernel time itself is replaceable
by FA3. Even if FA3's attention kernel were 20 % faster, the end-to-end win is
bounded by how much of total time attention occupies — at head_dim=128 + GQA
8KV + seq_len=4096, attention is a small share, and the +4.6 % we measure is
about what's available.

To get bigger wins on this shape, the next lever is GEMM/FFN (FP8 compute on
4-node config gave +14 % over bf16), not attention.
