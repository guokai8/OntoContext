"""
step30_plot_pareto.py — Pareto plot of accuracy vs tokens.

Plots the accuracy / token-budget Pareto frontier for the
OntoContext renderer vs. baselines (BM25, dense, etc.).

Output:
    figures/step30_pareto.png/.pdf
    tables/step30_pareto.csv

Usage:
    python3 scripts/step30_plot_pareto.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import _make_localization_questions, _score_answer
from step11_bench_hierarchy import _make_hierarchy_questions
from step20_baseline_bm25 import _bm25_retrieve
from step21_baseline_dense import _retrieve as _dense_retrieve
from step22_baseline_graphrag import _retrieve as _graphrag_retrieve
from step23_baseline_raw_neighborhood import _retrieve as _raw_retrieve
from step24_baseline_ontology_embedding import _retrieve as _onto_retrieve

# Determinism per the plan.
np.random.seed(17)

BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096]


def _score_pipeline(snapshot: str, ontology: str, family: str, budget: int,
                     n_questions: int | None = None) -> tuple[int, int]:
    if family == "localization":
        questions = _make_localization_questions(snapshot, ontology)
    elif family == "hierarchy":
        questions = _make_hierarchy_questions(snapshot, ontology)
    else:
        return 0, 0
    if n_questions and len(questions) > n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, n_questions)
    n_correct = 0
    for q in questions:
        rc = render(q["question"], snapshot, ontology, budget)
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}\n\n{rc['text']}"
        r = _score_answer(answer, rc["text"], q.get("gold_term_ids", []))
        if r["correct"]:
            n_correct += 1
    return n_correct, len(questions)


def _score_baseline(retriever, snapshot: str, ontology: str, family: str,
                     n_questions: int | None = None) -> tuple[int, int]:
    if family == "localization":
        questions = _make_localization_questions(snapshot, ontology)
    elif family == "hierarchy":
        questions = _make_hierarchy_questions(snapshot, ontology)
    else:
        return 0, 0
    if n_questions and len(questions) > n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, n_questions)
    n_correct = 0
    for q in questions:
        try:
            retrieved = retriever(q["question"], snapshot, ontology)
        except TypeError as e:
            # 2-arg retriever (graphrag/raw/onto)
            retrieved = retriever(q["question"], snapshot)
        answer = f"Retrieved: {', '.join(retrieved)}"
        r = _score_answer(answer, "\n".join(retrieved), q.get("gold_term_ids", []))
        if r["correct"]:
            n_correct += 1
    return n_correct, len(questions)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    family = "localization"

    rows: list[dict] = []
    # OntoContext Pareto curve
    for b in BUDGETS:
        n_correct, n = _score_pipeline(snapshot, ontology, family, b,
                                         n_questions=args.n_questions)
        rows.append({"method": f"OntoContext (budget={b})", "n_correct": n_correct,
                     "n": n, "accuracy": n_correct / max(1, n)})
    # Baselines (constant)
    def _bm25_safe(q, s, o): return _bm25_retrieve(q, s, o, top_k=3)
    def _dense_safe(q, s, o): return _dense_retrieve(q, s, o, top_k=3)
    def _graphrag_safe(q, s, o): return _graphrag_retrieve(q, s)
    def _raw_safe(q, s, o): return _raw_retrieve(q, s)
    def _onto_safe(q, s, o): return _onto_retrieve(q, s)
    for name, retriever in [("BM25", _bm25_safe),
                             ("Dense (random)", _dense_safe),
                             ("GraphRAG", _graphrag_safe),
                             ("Raw neighbourhood", _raw_safe),
                             ("Onto-embedding", _onto_safe)]:
        n_correct, n = _score_baseline(retriever, snapshot, ontology, family,
                                          n_questions=args.n_questions)
        rows.append({"method": name, "n_correct": n_correct, "n": n,
                     "accuracy": n_correct / max(1, n)})

    TBL_DIR = Path("tables")
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step30_pareto_{snapshot}"
    with (TBL_DIR / f"{basename}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["method", "n_correct", "n", "accuracy"])
        for r in rows:
            csv.writer(fh).writerow([r["method"], r["n_correct"], r["n"],
                                      f"{r['accuracy']:.4f}"])

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    # Pareto: x = n_correct (proxy for tokens used; for OntoContext we use budget)
    oc = [r for r in rows if r["method"].startswith("OntoContext")]
    xs_oc = [int(r["method"].split("=")[1].rstrip(")")) for r in oc]
    ys_oc = [r["accuracy"] for r in oc]
    ax.plot(xs_oc, ys_oc, marker="o", color="#3b6fb6", label="OntoContext")
    # Baselines as horizontal lines (x = mean budget of OC)
    baseline_x = sum(xs_oc) / len(xs_oc)
    for r in rows:
        if not r["method"].startswith("OntoContext"):
            ax.axhline(y=r["accuracy"], linestyle="--", color="gray", alpha=0.5)
            ax.text(baseline_x, r["accuracy"], f"  {r['method']}: {r['accuracy']:.2f}",
                    va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("token budget")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Step 30 — Pareto: accuracy vs tokens ({snapshot})")
    ax.legend()
    FIG_DIR = Path("figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {snapshot}: {len(rows)} rows; figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--n-questions", type=int, default=None,
                   help="cap on the number of questions per family (default: "
                        "all; for real ontologies with many thousands of "
                        "terms, set to ~50 for tractable runtime)")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
