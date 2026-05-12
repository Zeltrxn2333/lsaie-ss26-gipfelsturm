# Qwen3-14B 1 node × 4 GPU optimization benchmarks

Two end-to-end training-throughput ablations on CSCS Daint GH200 plus one
GEMM-kernel microbenchmark. All measurements use Qwen3-14B at TP=4 PP=1 DP=1,
GBS=256, FA3 attention (except where attention itself is being swept), stock
Megatron defaults otherwise.

## Folder layout

```
bench/
├── README.md                          ← this file
├── attention/                         ← attention-backend ablation
│   ├── sweep.sh                       ← submit 6 backends × 6 seq_lens (36 jobs)
│   ├── plot.py                        ← CSV → grouped bar chart
│   ├── data.csv                       ← 36 rows: mean throughput per cell
│   └── chart.png                      ← final chart
├── ffn/                               ← FFN ablation
│   ├── sweep.sh                       ← submit MBS×precision×SwiGLU-kernel (8 jobs)
│   ├── plot.py                        ← CSV → grouped bar chart
│   ├── data.csv                       ← 8 rows: tflops + tok/s per cell
│   └── chart.png                      ← final chart
└── microbench/                        ← GEMM kernel-level comparisons
    ├── deepgemm_vs_cublas.py          ← cuBLAS FP8 vs DeepGEMM FP8 standalone bench
    └── results.md                     ← findings (DeepGEMM is slower on FC1/FC2)
```

## 1. Attention ablation

6 attention kernels × 6 sequence lengths (512 → 16384). All backends run under
`--transformer-impl transformer_engine`, so LayerNorm / SP / fused softmax /
MLP are identical across runs — only the attention kernel changes.

| Backend | Megatron flag | Kernel source |
|---------|---------------|---------------|
| Standard (math SDPA) | `--attention-backend unfused` | PyTorch math (`baddbmm` + softmax + `bmm`) |
| FA2                  | `--attention-backend flash`   | container's `flash-attn 2.7.4.post1` |
| Triton               | `GIPFEL_ATTN_KERNEL=triton`   | OpenAI Triton tutorial 06-fused-attention (F+B) |
| cuDNN FusedAttention | `--attention-backend fused`   | cuDNN via TE |
| FA3                  | `--attention-backend flash` + `GIPFEL_USE_FA3=1` | FA3 built from source in `/iopsstor/.../venvs/fa3` |
| TileLang             | `GIPFEL_ATTN_KERNEL=tilelang` | TileLang 0.1.9 (fwd kernel; bwd falls back to math SDPA) |

Triton + TileLang are wired in via `patches/0003-custom-attention-kernels.patch`
which short-circuits `TEDotProductAttention.forward` to our `kernels/` package.

**Headline result (tokens/sec/GPU, MBS=1):**

| seq    | Std  | FA2  | Triton | cuDNN | FA3      | TileLang |
|--------|------|------|--------|-------|----------|----------|
| 512    | 515  | 611  | 603    | 611   | 591      | 580      |
| 1024   | 1035 | 1180 | 1211   | 1227  | 1210     | 1135     |
| 2048   | 1998 | 2430 | 2340   | 2429  | 2421     | 2293     |
| 4096   | 2293 | 3652 | 3434   | 3943  | 3943     | 3615     |
| 8192   | OOM  | 3922 | 3216   | 4215  | **4301** | 3734     |
| 16384  | OOM  | 3467 | 2620   | 3956  | **4101** | 3307     |

FA3 wins at long seq. TileLang holds ~80-85 % of FA3. Triton drops sharply
past 4k. Standard SDPA OOMs at seq ≥ 8k (O(seq²) attention matrix).

## 2. FFN ablation

8 configurations on Qwen3-14B 1n4g, seq=4096, FA3 attention:
`{MBS=1, MBS=2} × {BF16, FP8} × {Megatron-native SwiGLU, Liger Triton SwiGLU}`.

Liger SwiGLU integration is via `patches/0004-liger-swiglu-kernel.patch` which
swaps Megatron's `bias_swiglu_impl` for `LigerSiLUMulFunction` when
`GIPFEL_SWIGLU_KERNEL=liger`. FC1 and FC2 cuBLAS GEMMs are unchanged.

**Headline result (tokens/sec/GPU, mean of stable iters):**

| MBS | precision | Megatron native SwiGLU | Liger Triton SwiGLU |
|-----|-----------|------------------------|---------------------|
| 1   | BF16      | 3943 (baseline)        | 3985 (+1.1 %)       |
| 1   | FP8       | 3781 (-4.1 %)          | 3757 (-4.7 %)       |
| 2   | BF16      | 4598 (+16.6 %)         | 4560 (+15.6 %)      |
| 2   | FP8       | **6225 (+57.9 %)**     | 6038 (+53.1 %)      |

- **Winner: MBS=2 + cuBLAS FP8 + Megatron native SwiGLU = +57.9 % vs MBS=1 BF16.**
- FP8 only helps at large M (M = MBS·seq/TP). At MBS=1 → M=1024, FP8 *loses* 4 %; at MBS=2 → M=2048, FP8 *wins* 35 %.
- Liger Triton activation is within ±3 % of Megatron native — no systematic win because activation is only 2-5 % of MLP time.

## 3. DeepGEMM microbench

Standalone GEMM benchmark (no Megatron) at the exact MLP shapes. See
`microbench/results.md` — DeepGEMM is **15-18 % slower than cuBLAS FP8** on
FC1/FC2 (the dominant GEMMs), only marginally faster on small Q/O projections.
**Net: DeepGEMM does not help on this stack.**

## How to reproduce

### One-time setup (Daint login node)

1. Clone + pull submodules into `/users/$USER/gipfelsturm`.
2. Build the **FA3** venv at `/iopsstor/.../venvs/fa3` (see top-level
   `CLAUDE.md` § "Flash-Attention 3" for the recipe).
3. Install **TileLang** venv at `/iopsstor/.../venvs/tilelang` (sbatch
   recipe in earlier bench commits; pip install + strip duplicate
   torch/nvidia deps).
4. Install **Liger** venv at `/iopsstor/.../venvs/liger`:
   ```bash
   sbatch <<'EOF'
   #!/bin/bash
   #SBATCH --account=lp160 --partition=debug --time=00:10:00
   #SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=50G
   VENV=/iopsstor/scratch/cscs/$USER/venvs/liger
   mkdir -p $VENV
   srun --environment=alps3-mp pip install --target=$VENV liger-kernel
   cd $VENV
   rm -rf torch torch-*.dist-info functorch torchgen pkg_resources nvidia* \
          networkx* sympy* mpmath* triton* setuptools* jinja2* \
          MarkupSafe* markupsafe* filelock* fsspec* typing_extensions* \
          ninja* packaging* bin _distutils_hack distutils-precedence.pth \
          wheel* transformers* tokenizers* huggingface_hub* regex* tqdm* \
          pyyaml* safetensors* requests* certifi* charset* idna* urllib3* \
          numpy* sentencepiece* protobuf* hf_xet
   EOF
   ```
5. (Optional, for microbench) Install **DeepGEMM** at `/iopsstor/.../venvs/deepgemm`:
   ```bash
   sbatch <<'EOF'
   #!/bin/bash
   #SBATCH --account=lp160 --partition=debug --time=00:25:00
   #SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=100G
   VENV=/iopsstor/scratch/cscs/$USER/venvs/deepgemm
   BUILD=/iopsstor/scratch/cscs/$USER/build
   mkdir -p $VENV $BUILD
   srun --environment=alps3-mp bash -c '
     set -e
     cd '"$BUILD"'
     [ -d DeepGEMM ] || git clone --depth=1 --recurse-submodules \
       https://github.com/deepseek-ai/DeepGEMM.git
     cd DeepGEMM
     pip install --no-build-isolation --target='"$VENV"' .
   '
   EOF
   ```

### Run the sweeps

```bash
cd /users/$USER/gipfelsturm

# Attention: 36 jobs (6 backends × 6 seq_lens)
bash bench/attention/sweep.sh --dry-run     # preview
bash bench/attention/sweep.sh               # submit all

# FFN: 8 jobs (MBS × precision × SwiGLU kernel)
bash bench/ffn/sweep.sh --dry-run
bash bench/ffn/sweep.sh
```

Each job is 15 iterations (~5-15 min); first 5 are warmup, rest averaged.

### Refresh the charts

`data.csv` files are checked into the repo with the current measurements.
To regenerate the chart from CSV:

```bash
python3 bench/attention/plot.py bench/attention/data.csv bench/attention/
python3 bench/ffn/plot.py       bench/ffn/data.csv       bench/ffn/
```

(matplotlib needed; on Daint it's not pre-installed — plot locally if necessary.)

### Configurable env vars (both sweeps)

| var | default | meaning |
|-----|---------|---------|
| `GIPFEL_ACCOUNT`   | `lp160`   | SLURM account |
| `GIPFEL_PARTITION` | `normal`  | SLURM partition |
| `GIPFEL_WORKDIR`   | `/users/$USER/gipfelsturm` | repo path on Daint |
| `GIPFEL_TIME`      | `00:25:00` | SLURM time limit |
| `MODEL`            | `qwen3-14b` | model size (see launch-mp.sh) |
| `NODES`            | `1`       | node count |
| `ITERS`            | `15`      | training iters per job |
