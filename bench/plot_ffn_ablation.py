#!/usr/bin/env python3
"""Plot FFN ablation chart for Qwen3-14B 1n4g, seq=4096.

Compares: 2 MBS values × 2 precisions × 2 SwiGLU kernels = 8 configs.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data: tokens/sec/GPU, mean of last 5 stable iters
DATA = [
    # (MBS, precision, kernel, tok/s/GPU, tflops)
    (1, "BF16", "Megatron native", 3943, 339),
    (1, "BF16", "Liger Triton",    3985, 342),
    (1, "FP8",  "Megatron native", 3781, 325),
    (1, "FP8",  "Liger Triton",    3757, 322),
    (2, "BF16", "Megatron native", 4598, 395),
    (2, "BF16", "Liger Triton",    4560, 391),
    (2, "FP8",  "Megatron native", 6225, 534),
    (2, "FP8",  "Liger Triton",    6038, 518),
]

CONFIGS = ["MBS=1\nBF16", "MBS=1\nFP8", "MBS=2\nBF16", "MBS=2\nFP8"]
KERNELS = ["Megatron native", "Liger Triton"]
COLORS = {"Megatron native": "#1f77b4", "Liger Triton": "#ff7f0e"}


def main():
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CONFIGS))
    width = 0.35

    for i, kernel in enumerate(KERNELS):
        heights = []
        for cfg in [(1, "BF16"), (1, "FP8"), (2, "BF16"), (2, "FP8")]:
            mbs, prec = cfg
            tok = next(d[3] for d in DATA if d[0] == mbs and d[1] == prec and d[2] == kernel)
            heights.append(tok)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, heights, width, label=kernel, color=COLORS[kernel])
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 60,
                    f"{h}", ha="center", va="bottom", fontsize=9)

    # Annotate baseline (MBS=1 BF16 Megatron native)
    baseline = 3943
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.7, baseline + 60, "baseline = 3943 tok/s/GPU\n(MBS=1 BF16 Megatron)",
            ha="right", va="bottom", color="gray", fontsize=9)

    # Annotate winner
    winner = 6225
    ax.annotate(
        f"+57.9% vs baseline",
        xy=(3 - 0.5 * width, winner), xytext=(3 - 0.5 * width, winner + 400),
        ha="center", color="darkgreen", fontsize=10, weight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=11)
    ax.set_xlabel("(micro batch size, precision)", fontsize=11)
    ax.set_ylabel("Tokens per second per GPU", fontsize=11)
    ax.set_title("FFN ablation: Qwen3-14B 1 node × 4 GPU, seq=4096, FA3 attention\n"
                 "(MBS × precision × SwiGLU kernel)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 7300)

    out = Path("bench/results/qwen3-14b_ffn_ablation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
