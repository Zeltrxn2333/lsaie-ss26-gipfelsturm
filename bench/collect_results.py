#!/usr/bin/env python3
"""Parse sweep log files into a CSV.

Run on the Daint login node from /users/$USER/gipfelsturm:
    python3 bench/collect_results.py > bench/results/sweep.csv

Walks /users/$USER/gipfelsturm/logs/, extracts per-iter
`throughput per GPU (TFLOP/s/GPU)` values, computes the mean of iters 5-N
(skipping warmup), and detects OOM / failure modes.
"""
import csv
import os
import re
import sys
from pathlib import Path

LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))

# Job-name pattern, e.g.
#   gipfel-throughput-qwen3-14b-tp4pp1cp2-flashfa3-15s-2n-<jid>.log
#   gipfel-throughput-qwen3-8b-tp4pp1-unfused-15s-1n-<jid>.log
NAME_RE = re.compile(
    r"gipfel-throughput-"
    r"(?P<model>qwen3-8b|qwen3-14b|qwen3-32b|qwen2\.5-72b|llama3-70b|"
    r"mixtral-8x7b|mixtral-8x22b|32b|8b|140b)"
    r"-tp(?P<tp>\d+)pp(?P<pp>\d+)(?:cp(?P<cp>\d+))?"
    r"(?:-(?P<backend>unfused|flash|fused|auto|local)(?P<fa3>fa3)?)?"
    r"-(?P<steps>\d+)s-(?P<nodes>\d+)n-(?P<jid>\d+)\.log"
)

TFLOPS_RE = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): ([0-9.]+)")
TOKENS_RE = re.compile(r"tokens/sec/GPU: ([0-9]+)")
OOM_RE = re.compile(r"OutOfMemoryError|out of memory")
SEQLEN_RE = re.compile(r"--seq-length (\d+)")

GH200_PEAK = 494.0  # bf16 dense TFLOP/s peak per GPU


def classify(log: Path) -> dict:
    text = log.read_text(errors="ignore")
    tflops = [float(m) for m in TFLOPS_RE.findall(text)]
    tokens = [int(m) for m in TOKENS_RE.findall(text)]
    has_oom = bool(OOM_RE.search(text))
    seqlen_match = SEQLEN_RE.search(text)
    seq_len = int(seqlen_match.group(1)) if seqlen_match else 0
    stable = tflops[4:] if len(tflops) >= 5 else []
    if stable:
        mean_tflops = sum(stable) / len(stable)
        mean_tokens = sum(tokens[4:][: len(stable)]) / len(stable) if tokens else 0
        status = "OK"
    elif has_oom:
        mean_tflops = 0.0
        mean_tokens = 0
        status = "OOM"
    elif tflops:
        mean_tflops = sum(tflops) / len(tflops)
        mean_tokens = sum(tokens) / len(tokens) if tokens else 0
        status = "PARTIAL"
    else:
        mean_tflops = 0.0
        mean_tokens = 0
        status = "FAIL"
    return {
        "seq_len": seq_len,
        "iters_recorded": len(tflops),
        "iters_stable": len(stable),
        "mean_tflops": round(mean_tflops, 2),
        "mean_tokens_per_gpu": int(mean_tokens),
        "mfu_pct": round(100.0 * mean_tflops / GH200_PEAK, 2),
        "status": status,
    }


def main() -> int:
    rows = []
    for log in sorted(LOG_DIR.glob("gipfel-throughput-*.log")):
        m = NAME_RE.match(log.name)
        if not m:
            continue
        meta = m.groupdict()
        backend = meta.get("backend") or "default"
        if meta.get("fa3"):
            backend = f"{backend}+fa3"
        row = {
            "jid": meta["jid"],
            "model": meta["model"],
            "tp": int(meta["tp"]),
            "pp": int(meta["pp"]),
            "cp": int(meta["cp"] or 1),
            "backend": backend,
            "nodes": int(meta["nodes"]),
        }
        row.update(classify(log))
        rows.append(row)

    if not rows:
        print("No matching logs found", file=sys.stderr)
        return 1

    fieldnames = [
        "jid", "model", "tp", "pp", "cp", "backend", "nodes", "seq_len",
        "status", "iters_recorded", "iters_stable",
        "mean_tflops", "mean_tokens_per_gpu", "mfu_pct",
    ]
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
