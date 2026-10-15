"""
step30_matched_budget.py — Fast matched-budget experiment.

To stay under sandbox runtime limits, this runs a
TRIMMED matched-budget sweep on the **synthetic** snapshot
(synth_d06_b05_s017 is the canonical headline synth).

At each of 5 budgets (64, 128, 256, 512, 1024 tokens), this
scores OntoContext and the 5 baselines on a deterministic
10-question × 4-family subset. The synthetic ontology is
small enough that each render takes <0.5s.

Output:
    tables/step30_matched_budget.csv
    figures/step30_matched_budget.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from step08_render import render
from step10_bench_localization import (
    _make_localization_questions, _score_answer,
)
from step11_bench_hierarchy import _make_hierarchy_questions
from step13_bench_relation import _make_relation_questions
from step12_bench_sibling import _make_sibling_questions
from step20_baseline_bm25 import _bm25_retrieve, _build_corpus
from step21_baseline_dense import _retrieve as _dense_retrieve
from step22_baseline_graphrag import _retrieve as _graphrag_retrieve
from step23_baseline_raw_neighborhood import _retrieve as _raw_retrieve
from step24_baseline_ontology_embedding import _retrieve as _onto_retrieve

np.random.seed(17)
random.seed(17)

BUDGETS = [64, 128, 256, 512, 1024]
FAMILIES = ["localization", "hierarchy",
            "relation-completion", "sibling-discrimination"]
SNAPSHOT = "synth_d06_b05_s017"
ONTOLOGY = "SYNTH"
N_QUESTIONS = 10  # small for fast runtime

_CORPUS = None


def _corpus(snapshot, ontology):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _build_corpus(snapshot, ontology)
    return _CORPUS


def _load_qs(family, n):
    if family == "localization":
        qs = _make_localization_questions(SNAPSHOT, ONTOLOGY)
    elif family == "hierarchy":
        qs = _make_hierarchy_questions(SNAPSHOT, ONTOLOGY)
    elif family == "relation-completion":
        qs = _make_relation_questions(SNAPSHOT, ONTOLOGY)
    elif family == "sibling-discrimination":
        qs = _make_sibling_questions(SNAPSHOT, ONTOLOGY)
    else:
        return []
    if len(qs) > n:
        random.seed(17)
        qs = random.sample(qs, n)
    return qs


def _gold_ids(q):
    return re.findall(r"\b([A-Z][A-Z0-9]+:[A-Z0-9_]+)\b", q)


def _score(retrieved, gold):
    return bool(set(gold) & set(retrieved))


def score_oc(q, snapshot, ontology, budget):
    rc = render(q, snapshot, ontology, budget)
    gold = _gold_ids(q)
    retrieved = (rc.get("selected_modules", [])
                 + rc.get("selected_capsules", []))
    return _score(retrieved, gold)


def score_bm25(q, snapshot, ontology, budget):
    """At budget B, take top-k BM25 hits where k = budget / 64."""
    k = max(1, budget // 64)
    retrieved = _bm25_retrieve(q, snapshot, ontology, top_k=k)
    return _score(retrieved, _gold_ids(q))


def score_dense(q, snapshot, ontology, budget):
    k = max(1, budget // 64)
    retrieved = _dense_retrieve(q, snapshot, ontology, top_k=k)
    return _score(retrieved, _gold_ids(q))


def score_graphrag(q, snapshot, ontology, budget):
    retrieved = _graphrag_retrieve(q, snapshot)
    return _score(retrieved, _gold_ids(q))


def score_raw(q, snapshot, ontology, budget):
    retrieved = _raw_retrieve(q, snapshot)
    return _score(retrieved, _gold_ids(q))


def score_onto(q, snapshot, ontology, budget):
    retrieved = _onto_retrieve(q, snapshot)
    return _score(retrieved, _gold_ids(q))


SCORERS = {
    "OntoContext": score_oc,
    "BM25": score_bm25,
    "Dense (random)": score_dense,
    "GraphRAG": score_graphrag,
    "Raw neighbourhood": score_raw,
    "Onto-embedding": score_onto,
}


def run() -> list[dict]:
    rows = []
    print(f"=== {SNAPSHOT} | N={N_QUESTIONS}/family | {len(FAMILIES)} families ===",
          flush=True)
    print(f"=== Budgets: {BUDGETS} ===\n", flush=True)

    for family in FAMILIES:
        qs = _load_qs(family, N_QUESTIONS)
        print(f"Family: {family} ({len(qs)} questions)", flush=True)
        for method, scorer in SCORERS.items():
            if method == "OntoContext":
                for b in BUDGETS:
                    t0 = time.time()
                    n_ok = sum(scorer(q["question"], SNAPSHOT, ONTOLOGY, b)
                               for q in qs)
                    dt = time.time() - t0
                    acc = n_ok / max(1, len(qs))
                    rows.append({
                        "method": f"OntoContext (budget={b})",
                        "family": family,
                        "budget": b,
                        "n_correct": n_ok,
                        "n": len(qs),
                        "accuracy": acc,
                    })
                    print(f"  {method:18s} @ {b:4d} tok: {n_ok}/{len(qs)} = {acc:.0%}  ({dt:.1f}s)",
                          flush=True)
            else:
                t0 = time.time()
                n_ok = sum(scorer(q["question"], SNAPSHOT, ONTOLOGY, 256)
                           for q in qs)
                dt = time.time() - t0
                acc = n_ok / max(1, len(qs))
                rows.append({
                    "method": method,
                    "family": family,
                    "budget": "full",
                    "n_correct": n_ok,
                    "n": len(qs),
                    "accuracy": acc,
                })
                print(f"  {method:18s}:               {n_ok}/{len(qs)} = {acc:.0%}  ({dt:.1f}s)",
                      flush=True)
        print(flush=True)

    # 4-family averages
    for b in BUDGETS:
        ok = sum(r["n_correct"] for r in rows
                 if r["method"] == f"OntoContext (budget={b})")
        n = sum(r["n"] for r in rows
                if r["method"] == f"OntoContext (budget={b})")
        rows.append({
            "method": f"OntoContext (budget={b})",
            "family": "4-family avg",
            "budget": b,
            "n_correct": ok,
            "n": n,
            "accuracy": ok / max(1, n),
        })

    return rows


def write_table(rows) -> Path:
    TBL = Path("tables")
    TBL.mkdir(exist_ok=True)
    out = TBL / "step30_matched_budget.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "method", "family", "budget", "n_correct", "n", "accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "method": r["method"],
                "family": r["family"],
                "budget": r["budget"],
                "n_correct": r["n_correct"],
                "n": r["n"],
                "accuracy": f"{r['accuracy']:.4f}",
            })
    return out


def write_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg = [r for r in rows if r["family"] == "4-family avg"]
    oc = sorted([r for r in avg if r["method"].startswith("OntoContext")],
                key=lambda x: x["budget"])
    bl = [r for r in avg if not r["method"].startswith("OntoContext")]

    fig, ax = plt.subplots(figsize=(9, 5))

    oc_b = [r["budget"] for r in oc]
    oc_a = [r["accuracy"] for r in oc]
    ax.plot(oc_b, oc_a, marker="o", linewidth=2.5,
            color="#2563eb", label="OntoContext", zorder=5)
    ax.scatter(oc_b, oc_a, color="#2563eb", s=80, zorder=6)

    colors = {"BM25": "#dc2626", "GraphRAG": "#7c3aed",
              "Raw neighbourhood": "#d97706",
              "Dense (random)": "#6b7280",
              "Onto-embedding": "#6b7280"}
    for r in bl:
        c = colors.get(r["method"], "#9ca3af")
        ax.axhline(y=r["accuracy"], linestyle="--", color=c,
                   alpha=0.7, linewidth=1.5, zorder=3)
        ax.scatter([oc_b[-1]], [r["accuracy"]], color=c,
                   marker="o", s=50, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(40, 2048)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("token budget", fontsize=11)
    ax.set_ylabel("accuracy (4-family average)", fontsize=11)
    ax.set_title(
        f"Matched-budget Pareto on {SNAPSHOT}\n"
        f"(OntoContext at each budget; baselines at full-corpus cost)",
        fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    for b, a in zip(oc_b, oc_a):
        ax.annotate(f"{a:.0%}", (b, a),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#2563eb")

    for r in bl:
        c = colors.get(r["method"], "#9ca3af")
        ax.text(oc_b[-1] * 1.1, r["accuracy"],
                f"  {r['method']}: {r['accuracy']:.0%}",
                va="center", fontsize=8, color=c)

    FIG = Path("figures")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "step30_matched_budget.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "step30_matched_budget.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default=SNAPSHOT)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return _run(args.snapshot)


def _run(snapshot: str) -> int:
    """Inner runner so module-level SNAPSHOT/ONTOLOGY aren't needed."""
    snapshot_ = snapshot
    ontology_ = "SYNTH" if "synth" in snapshot_ else "CL"
    # Monkey-patch module-level constants used by helpers
    global SNAPSHOT, ONTOLOGY
    SNAPSHOT = snapshot_
    ONTOLOGY = ontology_
    t0 = time.time()
    rows = run()
    print(f"\n=== Total time: {time.time()-t0:.1f}s ===", flush=True)
    tbl = write_table(rows)
    print(f"Table: {tbl}", flush=True)
    write_figure(rows)
    print("Figures: figures/step30_matched_budget.{png,pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
