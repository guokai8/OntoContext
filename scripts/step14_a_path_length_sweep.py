"""
step14_a_path_length_sweep.py — path-length ablation for multi-hop.

Sweep the path length in the multi-hop benchmark. For each target
length L ∈ {1, 2, 3, 4, 5}, generate a question with a path of
length L, and report accuracy.

Output:
    tables/step14_a_path_length_sweep.csv
    figures/step14_a_path_length_sweep.png/.pdf

Usage:
    python3 scripts/step14_a_path_length_sweep.py --synthetic synth_d02_b02_s017
    python3 scripts/step14_a_path_length_sweep.py --snapshot go_2024-06-10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step14_bench_multihop import _bfs_path, _load_graph, _score_multihop
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)


def _path_with_length(adj, length: int) -> tuple[str, str, list[str]] | None:
    """Find a path with at least `length` nodes (length-1 edges)."""
    nodes = list(adj.keys())
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            path = _bfs_path(adj, src, dst)
            if path and len(path) - 1 == length - 1:
                return src, dst, path
    return None


def _make_q_for_length(adj, length: int) -> dict | None:
    r = _path_with_length(adj, length)
    if not r:
        return None
    src, dst, path = r
    return {
        "qid": f"path_len_{length}",
        "question": f"Find a {length}-hop path from {src} to {dst}.",
        "src": src, "dst": dst,
        "gold_path": path,
        "gold_intermediates": path[1:-1],
        "gold_term_ids": path,
        "family": "multi-hop",
    }


def _sweep(snapshot: str, ontology: str, budget: int) -> list[dict]:
    proc = Path("data/processed") / snapshot
    adj = _load_graph(proc)
    rows: list[dict] = []
    for length in [1, 2, 3, 4, 5]:
        q = _make_q_for_length(adj, length)
        if not q:
            rows.append({
                "path_length": length, "n": 0, "n_correct": 0,
                "accuracy": 0.0, "no_path": True,
            })
            continue
        rc = render(q["question"], snapshot, ontology, budget)
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}\n\n{rc['text']}"
        r = _score_multihop(answer, q.get("gold_intermediates", []))
        rows.append({
            "path_length": length, "n": 1, "n_correct": int(r["correct"]),
            "accuracy": float(r["correct"]), "no_path": False,
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path_length", "n", "n_correct", "accuracy", "no_path"])
        for r in rows:
            w.writerow([r["path_length"], r["n"], r["n_correct"],
                        f"{r['accuracy']:.4f}", r["no_path"]])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lengths = [r["path_length"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    no_paths = [r["no_path"] for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    color = ["#bbbbbb" if n else "#3b6fb6" for n in no_paths]
    ax.bar(lengths, accs, color=color)
    ax.set_xlabel("path length (nodes)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Step 14a — path-length sweep ({rows[0].get('path_length')}-{rows[-1].get('path_length')})")
    for i, (l, n) in enumerate(zip(lengths, no_paths)):
        if n:
            ax.text(l, 0.5, "no path", ha="center", fontsize=8, rotation=90)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--budget", type=int, default=512)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR: pass --synthetic or --snapshot", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    rows = _sweep(snapshot, ontology, args.budget)
    basename = f"step14_a_path_length_sweep_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} lengths; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
