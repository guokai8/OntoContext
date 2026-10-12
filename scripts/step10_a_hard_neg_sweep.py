"""
step10_a_hard_neg_sweep.py — hard-negative ablation.

For the localization benchmark, sweep a confusion-tolerance
threshold: the scoring method only counts a question as correct
if the agent's answer mentions the gold term AND does NOT
mention any confusable term (sibling or near_term from V2
disambiguation) with rank ≤ K.

The point of this ablation: in a real ontology, two leaves of
the same parent can both be "valid" answers for a fuzzy
question; the hard-negative sweep measures how often the
agent confuses the gold with a confusable.

Output:
    tables/step10_a_hard_neg_sweep.csv
    figures/step10_a_hard_neg_sweep.png/.pdf

Usage:
    python3 scripts/step10_a_hard_neg_sweep.py --synthetic synth_d02_b02_s017
    python3 scripts/step10_a_hard_neg_sweep.py --snapshot go_2024-06-10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step10_bench_localization import (
    _extract_curies, _load_gold, _make_localization_questions, _load_agent_results,
    REPORT_DIR, TBL_DIR, FIG_DIR,
)
from step08_render import render, _infer_ontology

# Determinism per the plan.
np.random.seed(17)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 16]


def _load_confusables(snapshot_id: str) -> dict[str, set[str]]:
    """Load the V2 confusables table for a snapshot, return term → set of
    hard-negative IDs (rank ≤ 16)."""
    cf = Path("cache/confusables") / f"{snapshot_id}.parquet"
    if not cf.exists():
        return {}
    t = pq.read_table(cf).to_pylist()
    out: dict[str, set[str]] = defaultdict(set)
    for r in t:
        if r["rank"] <= 16:
            out[r["term_id"]].add(r["near_term_id"])
    return dict(out)


def _score_with_k(
    answer: str, context_text: str, gold_ids: list[str], hard_negs: set[str], k: int,
) -> dict:
    """Score an answer with hard-negative threshold K.

    A question is "correct" iff the answer mentions at least one gold
    term AND does not mention any hard-negative of rank ≤ K.
    """
    mentioned = set(_extract_curies(answer))
    gold = set(gold_ids)
    context_ids = set(_extract_curies(context_text))
    has_gold = bool(gold & mentioned)
    # Only consider hard-negs that are in the context (otherwise the
    # agent has no way to know about them)
    hard_negs_in_context = hard_negs & context_ids
    has_hard_neg = bool(mentioned & hard_negs_in_context)
    return {
        "correct": has_gold and not has_hard_neg,
        "has_gold": has_gold,
        "has_hard_neg": has_hard_neg,
        "gold_count": len(gold & mentioned),
        "hard_neg_count": len(mentioned & hard_negs_in_context),
    }


def run_sweep(
    questions: list[dict], snapshot: str, ontology: str,
    hard_negs_by_term: dict[str, set[str]], budget: int,
) -> list[dict]:
    """For each K, score every question and report the accuracy."""
    rows: list[dict] = []
    for k in K_VALUES:
        scores: list[dict] = []
        for q in questions:
            tid = q.get("term_id") or (q.get("gold_term_ids") or [""])[0]
            hard_negs = hard_negs_by_term.get(tid, set())
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}"
            r = _score_with_k(answer, rc["text"], q.get("gold_term_ids", []),
                              hard_negs, k)
            scores.append(r)
        n = len(scores)
        n_correct = sum(1 for s in scores if s["correct"])
        n_with_hard_neg = sum(1 for s in scores if s["has_hard_neg"])
        rows.append({
            "k": k,
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
            "n_with_hard_neg": n_with_hard_neg,
            "hard_neg_rate": n_with_hard_neg / max(1, n),
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "n", "n_correct", "accuracy", "n_with_hard_neg", "hard_neg_rate"])
        for r in rows:
            w.writerow([r["k"], r["n"], r["n_correct"], f"{r['accuracy']:.4f}",
                        r["n_with_hard_neg"], f"{r['hard_neg_rate']:.4f}"])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = [r["k"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    rates = [r["hard_neg_rate"] for r in rows]
    axes[0].plot(ks, accs, marker="o", color="#3b6fb6")
    axes[0].set_xlabel("K (max confusable rank considered)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Accuracy vs hard-negative threshold K")
    axes[0].set_ylim(0, 1)
    axes[1].plot(ks, rates, marker="s", color="#c0513b")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("hard-neg rate")
    axes[1].set_title("Hard-negative rate vs K")
    axes[1].set_ylim(0, max(rates + [0.1]) * 1.1)
    fig.suptitle(f"Step 10a — hard-neg sweep", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--family", type=str, default="localization")
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

    gold = _load_gold(args.family)
    questions = gold if gold else _make_localization_questions(snapshot, ontology)
    hard_negs = _load_confusables(snapshot)

    rows = run_sweep(questions, snapshot, ontology, hard_negs, args.budget)
    basename = f"step10_a_hard_neg_sweep_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} K values; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
