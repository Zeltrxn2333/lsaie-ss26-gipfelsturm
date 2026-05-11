#!/usr/bin/env python3
"""Plot grouped-bar attention backend chart from sweep CSV.

Usage:
    python3 bench/plot_results.py bench/results/sweep.csv bench/results/

Produces one PNG per (model, cp):
    <model>_cp<cp>.png
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKEND_ORDER = ["unfused", "flash", "triton", "fused", "flash+fa3", "tilelang"]
BACKEND_LABELS = {
    "unfused": "Standard (math SDPA)",
    "flash": "FA2",
    "triton": "Triton (OpenAI tutorial)",
    "fused": "cuDNN FusedAttention",
    "flash+fa3": "FA3",
    "tilelang": "TileLang",
}
BACKEND_COLORS = {
    "unfused": "#1f77b4",
    "flash": "#ff7f0e",
    "triton": "#2ca02c",
    "fused": "#d62728",
    "flash+fa3": "#9467bd",
    "tilelang": "#8c564b",
}
SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384]


def load(csv_path: Path):
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def plot_panel(rows, model, cp, out_dir: Path) -> None:
    by_seq_be = {}
    for r in rows:
        if r["model"] != model or int(r["cp"]) != cp:
            continue
        seq = int(r["seq_len"])
        if seq == 0:
            continue
        be = r["backend"]
        tflops = float(r["mean_tokens_per_gpu"])
        status = r["status"]
        by_seq_be[(seq, be)] = (tflops, status)

    if not by_seq_be:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(SEQ_LENS))
    width = 0.20

    for i, backend in enumerate(BACKEND_ORDER):
        heights = []
        statuses = []
        for seq in SEQ_LENS:
            tflops, status = by_seq_be.get((seq, backend), (0.0, "MISSING"))
            heights.append(tflops)
            statuses.append(status)
        bars = ax.bar(
            x + (i - (len(BACKEND_ORDER) - 1) / 2) * width,
            heights, width,
            label=BACKEND_LABELS[backend],
            color=BACKEND_COLORS[backend],
        )
        for bar, h, status in zip(bars, heights, statuses):
            if status == "OOM":
                ax.text(bar.get_x() + bar.get_width() / 2, 5, "OOM",
                        ha="center", va="bottom", fontsize=7, color="gray")
            elif status == "MISSING":
                pass  # skip empty
            elif h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEQ_LENS])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Tokens per second per GPU")
    ax.set_title(f"End-to-end attention backend throughput: {model}, CP={cp}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    max_h = max((h for h in [b[0] for b in by_seq_be.values()]), default=1.0)
    ax.set_ylim(0, max_h * 1.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_cp{cp}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    rows = load(csv_path)

    models = sorted({r["model"] for r in rows})
    cps = sorted({int(r["cp"]) for r in rows})
    for model in models:
        for cp in cps:
            plot_panel(rows, model, cp, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
