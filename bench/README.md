# Attention Backend Benchmark — Qwen3-14B 1 node × 4 GPU

End-to-end throughput comparison of 6 attention kernels on Qwen3-14B across
sequence lengths 512 → 16384 on CSCS Daint GH200.


## Backends compared

All 6 backends run inside `--transformer-impl transformer_engine`, so
LayerNorm / sequence-parallel / fused softmax / MLP are identical across
runs — **only the attention kernel changes**. This is the fair attention-only
A/B comparison.

| Backend | Megatron flag | Kernel source |
|---------|---------------|---------------|
| Standard (math SDPA) | `--attention-backend unfused`        | PyTorch math SDPA (`torch.baddbmm` + softmax + `torch.bmm`) |
| FA2                  | `--attention-backend flash`          | `flash-attn 2.7.4.post1` shipped in the alps3 container |
| Triton               | `GIPFEL_ATTN_KERNEL=triton`          | OpenAI Triton tutorial 06-fused-attention (F+B), GQA expanded |
| cuDNN FusedAttention | `--attention-backend fused`          | cuDNN FusedAttention via TE |
| FA3                  | `--attention-backend flash` + `GIPFEL_USE_FA3=1` | `flash-attn-3` built from source under `/iopsstor/.../venvs/fa3` |
| TileLang             | `GIPFEL_ATTN_KERNEL=tilelang`        | TileLang 0.1.9 GQA flash-attn fwd (backward falls back to PyTorch math SDPA) |

Triton and TileLang are wired in by `patches/0003-custom-attention-kernels.patch`
which patches `TEDotProductAttention.forward` to short-circuit through our
`kernels/` package when `GIPFEL_ATTN_KERNEL` is set.

## How to reproduce

### One-time setup (on Daint login node)

1. Clone the repo to `/users/$USER/gipfelsturm` and pull submodules:
   ```bash
   git clone <repo> /users/$USER/gipfelsturm
   cd /users/$USER/gipfelsturm
   git submodule update --init --recursive
   ```
2. Build FA3 venv (see `CLAUDE.md` § "Flash-Attention 3" for the recipe).
   Lands at `/iopsstor/scratch/cscs/$USER/venvs/fa3`.
3. Install TileLang into a separate venv:
   ```bash
   sbatch <<'EOF'
   #!/bin/bash
   #SBATCH --account=lp160 --partition=debug --time=00:15:00
   #SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=50G
   #SBATCH --output=install-tilelang.log
   VENV=/iopsstor/scratch/cscs/$USER/venvs/tilelang
   mkdir -p $VENV
   srun --environment=alps3-mp pip install --target=$VENV tilelang
   # Strip duplicated torch/nvidia/etc from the venv — they shadow the
   # container's torch. Same fix as the FA3 venv.
   cd $VENV
   rm -rf torch torch-*.dist-info functorch torchgen pkg_resources nvidia* \
          networkx* sympy* mpmath* triton* setuptools* jinja2* \
          MarkupSafe* markupsafe* filelock* fsspec* typing_extensions* \
          ninja* packaging* bin _distutils_hack distutils-precedence.pth wheel*
   EOF
   ```
4. Make sure the alps3-mp EDF is in `~/.edf/alps3-mp.toml` (see top-level
   `alps3-mp.toml`).

### Submit the sweep

```bash
cd /users/$USER/gipfelsturm
bash bench/attention_sweep.sh --dry-run    # preview 36 jobs
bash bench/attention_sweep.sh              # submit all
```

Each job runs 15 iterations of Qwen3-14B training (TP=4 PP=1 DP=1, GBS=256,
MBS=1, bf16 + distributed-optimizer + sequence-parallel + overlap-grad-reduce
+ overlap-param-gather, stock Megatron defaults otherwise). Submitted to
`normal` partition; first 5 iters are warmup, next 10-11 averaged.

To submit only one backend:
```bash
bash bench/attention_sweep.sh --only triton
```

### Collect + plot

After all jobs finish (monitor with `squeue -u $USER`):

```bash
# 1. Parse every log into the raw CSV
python3 bench/collect_results.py > bench/results/sweep.csv

# 2. Filter to the 36-row deliverable for this Qwen3-14B comparison
python3 bench/filter_deliverable.py bench/results/sweep.csv \
    > bench/results/qwen3-14b_deliverable.csv

# 3. Plot (matplotlib needed; if not on Daint, scp the CSV and plot locally)
python3 bench/plot_results.py bench/results/qwen3-14b_deliverable.csv bench/results/
```

The plotter needs `matplotlib`; on Daint login nodes it's not installed by
default (do it locally if needed: `python3 -m pip install --user 'matplotlib==3.3.0' --only-binary=:all:`).

## Results

`results/qwen3-14b_cp1.png` — grouped bar chart, x-axis seq_len, y-axis
tokens/sec/GPU, 6 bars per seq_len.

Headline numbers (tokens/sec/GPU, mean of iters 5-15):

| seq    | Std  | FA2  | Triton | cuDNN | FA3  | TileLang |
|--------|------|------|--------|-------|------|----------|
| 512    | 515  | 611  | 603    | 611   | 591  | 580  |
| 1024   | 1035 | 1180 | 1211   | 1227  | 1210 | 1135 |
| 2048   | 1998 | 2430 | 2340   | 2429  | 2421 | 2293 |
| 4096   | 2293 | 3652 | 3434   | 3943  | 3943 | 3615 |
| 8192   | OOM  | 3922 | 3216   | 4215  | **4301** | 3734 |
| 16384  | OOM  | 3467 | 2620   | 3956  | **4101** | 3307 |

- FA3 leads at every long seq_len.
- TileLang holds ~80-85 % of FA3 at long seq.
- Triton (OpenAI tutorial) is competitive at short seq but drops sharply
  past 4k — its kernel is not Hopper-tuned.
- Standard math SDPA OOMs at seq ≥ 8k (O(seq²) attention matrix).
