# DeepGEMM FP8 vs cuBLAS FP8 microbench

Standalone Python-level GEMM benchmark (no Megatron) at the exact M/K/N shapes
Qwen3-14B 1n4g MBS=2 TP=4 actually uses for MLP and attention projections.

## Setup

- GH200 GPU (sm_90a), CUDA 13.1, PyTorch 2.x (alps3-mp container)
- DeepGEMM 2.5.0 built from source, installed at
  `/iopsstor/scratch/cscs/$USER/venvs/deepgemm`
- cuBLAS FP8 via `torch._scaled_mm`
- Scales pre-computed outside the timed loop (only the GEMM kernel is timed)

## Result

| GEMM            | shape (M K N)        | cuBLAS FP8 TFLOP/s | DeepGEMM FP8 TFLOP/s | DG / cuBLAS |
|-----------------|----------------------|--------------------|----------------------|-------------|
| FC1 (gate+up)   | 2048 × 5120 × 8704   | **1242**           | 1048                 | **0.84×** (-15.7%) |
| FC2 (down)      | 2048 × 4352 × 5120   | **1199**           | 985                  | **0.82×** (-17.9%) |
| Q proj          | 2048 × 5120 × 1280   | 832                | 848                  | 1.02× (+1.9%)  |
| O proj          | 2048 × 1280 × 5120   | 734                | 764                  | 1.04× (+4.0%)  |

## Conclusion

**DeepGEMM does not help on this stack** for Qwen3-14B MLP shapes:
- On the two biggest GEMMs (FC1 and FC2, ~70% of MLP time) DeepGEMM is
  **15-18 % slower** than cuBLAS FP8.
- On the smaller projections (Q/O) DeepGEMM is marginally faster (1-4 %),
  but their total time is small.
- Net end-to-end MLP impact of swapping cuBLAS → DeepGEMM would be **negative**.

Likely reasons:
1. cuBLAS Hopper kernels are very well tuned for "round" shapes at M=2048.
2. DeepGEMM is primarily targeted at H100 + CUDA 12.x; GH200 + CUDA 13.1 may
   not hit its best paths.
3. DeepGEMM uses per-token × per-128-K-block scales, adding a small K-axis
   accumulation overhead vs. cuBLAS's per-tensor scale.

## Reproduction

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --account=lp160 --partition=debug --time=00:10:00
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --mem=50G
srun --environment=alps3-mp bash -c '
  export PYTHONPATH=/iopsstor/scratch/cscs/$USER/venvs/deepgemm:$PYTHONPATH
  cd /users/$USER/gipfelsturm
  python3 bench/microbench/deepgemm_vs_cublas.py
'
EOF
```
