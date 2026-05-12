#!/usr/bin/env python3
"""Plot FFN ablation chart from CSV.

Usage:
    python3 bench/ffn/plot.py bench/ffn/data.csv bench/ffn/

Produces:
    chart.png (overwrites)

CSV columns: mbs, precision, swiglu_kernel, tflops_per_gpu, tokens_per_sec_per_gpu
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONFIG_ORDER = [(1, "bf16"), (1, "fp8"), (2, "bf16"), (2, "fp8")]
CONFIG_LABEL = {
    (1, "bf16"): "MBS=1\nBF16",
    (1, "fp8"):  "MBS=1\nFP8",
    (2, "bf16"): "MBS=2\nBF16",
    (2, "fp8"):  "MBS=2\nFP8",
}
KERNEL_ORDER = ["megatron", "liger"]
KERNEL_LABEL = {"megatron": "Megatron native", "liger": "Liger Triton"}
COLORS = {"megatron": "#1f77b4", "liger": "#ff7f0e"}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    rows = list(csv.DictReader(open(csv_path)))

    data = {}
    for r in rows:
        k = (int(r["mbs"]), r["precision"].lower(), r["swiglu_kernel"].lower())
        data[k] = float(r["tokens_per_sec_per_gpu"])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CONFIG_ORDER))
    width = 0.35

    for i, kernel in enumerate(KERNEL_ORDER):
        heights = [data[(mbs, prec, kernel)] for mbs, prec in CONFIG_ORDER]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, heights, width,
                      label=KERNEL_LABEL[kernel], color=COLORS[kernel])
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 60,
                    f"{int(h)}", ha="center", va="bottom", fontsize=9)

    baseline = data[(1, "bf16", "megatron")]
    winner = data[(2, "fp8", "megatron")]
    speedup = (winner / baseline - 1) * 100

    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(3.7, baseline + 60,
            f"baseline = {int(baseline)} tok/s/GPU\n(MBS=1 BF16 Megatron)",
            ha="right", va="bottom", color="gray", fontsize=9)

    ax.annotate(
        f"+{speedup:.1f}% vs baseline",
        xy=(3 - 0.5 * width, winner), xytext=(3 - 0.5 * width, winner + 400),
        ha="center", color="darkgreen", fontsize=10, weight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABEL[c] for c in CONFIG_ORDER], fontsize=11)
    ax.set_xlabel("(micro batch size, precision)", fontsize=11)
    ax.set_ylabel("Tokens per second per GPU", fontsize=11)
    ax.set_title("FFN ablation: Qwen3-14B 1 node × 4 GPU, seq=4096, FA3 attention\n"
                 "(MBS × precision × SwiGLU kernel)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(data.values()) * 1.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chart.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
