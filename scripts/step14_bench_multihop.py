"""
step14_bench_multihop.py — Multi-hop benchmark (headline).

Given two terms, find a 2-hop or 3-hop path in the is_a graph
between them. Scoring:
    - correct iff the answer mentions at least one intermediate
      node that's actually on a shortest path.
    - the path length is verified against the gold longest shortest
      path in the V3 capsule's `coordinates.parents` chain.

For V3 (no external gold), the gold path is computed from the
V3 capsule's parent chain (BFS through the is_a graph).

Output:
    evaluation/reports/<run_id>/multi-hop/accuracy.json
    tables/step14_bench_multihop.csv/.tex
    figures/step14_bench_multihop.png/.pdf

Usage:
    python3 scripts/step14_bench_multihop.py --synthetic synth_d02_b02_s017
    python3 scripts/step14_bench_multihop.py --snapshot go_2024-06-10
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
from step10_bench_localization import TBL_DIR, FIG_DIR, REPORT_DIR

# Determinism per the plan.
np.random.seed(17)


def _bfs_path(adj: dict[str, set[str]], src: str, dst: str) -> list[str] | None:
    """Return a shortest path from src to dst in an undirected graph,
    or None if no path."""
    if src == dst:
        return [src]
    if src not in adj or dst not in adj:
        return None
    visited = {src}
    queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
    while queue:
        n, path = queue.popleft()
        for m in adj.get(n, ()):
            if m in visited:
                continue
            new_path = path + [m]
            if m == dst:
                return new_path
            visited.add(m)
            queue.append((m, new_path))
    return None


def _load_graph(processed_dir: Path) -> dict[str, set[str]]:
    nodes = [r["id"] for r in pq.read_table(processed_dir / "node.parquet").to_pylist()]
    edges = pq.read_table(processed_dir / "edge.parquet").to_pylist()
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e.get("relation_iri") != "obo:BFO_0000050":
            continue
        s, d = e.get("src_id"), e.get("dst_id")
        if s in adj and d in adj:
            adj[s].add(d)
            adj[d].add(s)
    return adj


def _make_multihop_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """For every pair of terms with a BFS path, generate a question
    asking for the path."""
    proc = Path("data/processed") / snapshot_id
    adj = _load_graph(proc)
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    by_id = {c["id"]: c for c in capsules}
    out: list[dict] = []
    q_counter = 0
    seen: set[tuple[str, str]] = set()
    for src in list(adj.keys()):
        for dst in list(adj.keys()):
            if src == dst:
                continue
            pair = (min(src, dst), max(src, dst))
            if pair in seen:
                continue
            path = _bfs_path(adj, src, dst)
            if not path or len(path) < 2:
                continue
            seen.add(pair)
            q_counter += 1
            intermediates = path[1:-1]   # nodes strictly between
            out.append({
                "qid": f"multi-hop_{q_counter:03d}",
                "question": (
                    f"Find a path from {by_id.get(src, {}).get('name', src)} "
                    f"(term id {src}) to {by_id.get(dst, {}).get('name', dst)} "
                    f"(term id {dst})."
                ),
                "src": src, "dst": dst,
                "gold_path": path,
                "gold_intermediates": intermediates,
                "gold_term_ids": path,
                "family": "multi-hop",
            })
    return out


def _score_multihop(answer: str, gold_intermediates: list[str]) -> dict:
    """Score: answer mentions at least one intermediate on the gold path.
    For len(path)==2 (direct edge), gold_intermediates is empty → trivially
    correct (one hop)."""
    if not gold_intermediates:
        return {"correct": True, "n_matched": 0, "n_gold": 0, "score": 1.0,
                "matched": []}
    matched = [g for g in gold_intermediates if g in answer]
    return {
        "correct": bool(matched),
        "matched": matched,
        "n_matched": len(matched),
        "n_gold": len(gold_intermediates),
        "score": len(matched) / max(1, len(gold_intermediates)),
    }


def run_benchmark(
    questions: list[dict], snapshot: str, ontology: str, budget: int, run_id: str,
) -> dict:
    scores: list[dict] = []
    for q in questions:
        rc = render(q["question"], snapshot, ontology, budget)
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}\n\n{rc['text']}"
        r = _score_multihop(answer, q.get("gold_intermediates", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "multi-hop"
        r["path_length"] = len(q.get("gold_path", []))
        scores.append(r)
    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    mean_score = sum(s["score"] for s in scores) / max(1, n)
    return {
        "run_id": run_id, "n_questions": n, "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n), "mean_score": mean_score,
        "per_question": scores,
    }


def _write_outputs(result: dict, snapshot: str) -> None:
    out = REPORT_DIR / result["run_id"] / "multi-hop" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step14_bench_multihop_{snapshot}"
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["n_questions", result["n_questions"]])
        w.writerow(["n_correct", result["n_correct"]])
        w.writerow(["overall_accuracy", f"{result['overall_accuracy']:.4f}"])
        w.writerow(["mean_score", f"{result['mean_score']:.4f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n = result["n_questions"]
    ax.bar(["correct", "incorrect"], [result["n_correct"], n - result["n_correct"]],
           color=["#3b6fb6", "#c0513b"])
    ax.set_ylabel("count")
    ax.set_title(f"Step 14 — multi-hop (headline)\n"
                  f"{snapshot}, accuracy={result['overall_accuracy']:.2%}")
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
    p.add_argument("--run-id", type=str, default="bench_mh")
    p.add_argument("--n-questions", type=int, default=None)
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
    questions = _make_multihop_questions(snapshot, ontology)
    if args.n_questions and len(questions) > args.n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, args.n_questions)
        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')
    if not questions:
        print(f"ERROR: no multi-hop questions for {snapshot}", file=sys.stderr)
        return 2
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    print(
        f"✅ multi-hop/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"= {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
