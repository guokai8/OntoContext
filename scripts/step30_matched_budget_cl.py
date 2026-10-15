"""
step30_matched_budget_cl.py — Single-family matched-budget on CL.

Runs ONE family at a time, so the foreground run stays under
the sandbox's 4-minute budget. We invoke this script per family
in 4 separate calls and combine the results downstream.

Usage:
    python3 scripts/step30_matched_budget_cl.py \
        --snapshot cl_2024-06-10 --family localization \
        --out tables/matched_cl_localization.csv
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
N_QUESTIONS = 10

_QUESTION_LOADERS = {
    "localization": _make_localization_questions,
    "hierarchy": _make_hierarchy_questions,
    "relation-completion": _make_relation_questions,
    "sibling-discrimination": _make_sibling_questions,
}


def _load_qs(family, snapshot, ontology, n):
    loader = _QUESTION_LOADERS[family]
    qs = loader(snapshot, ontology)
    if len(qs) > n:
        random.seed(17)
        qs = random.sample(qs, n)
    return qs


def _gold(q):
    return re.findall(r"\b([A-Z][A-Z0-9]+:[A-Z0-9_]+)\b", q)


def _score(retrieved, gold):
    return bool(set(gold) & set(retrieved))


def score_oc(q, snapshot, ontology, budget):
    rc = render(q, snapshot, ontology, budget)
    return _score(rc.get("selected_modules", []) + rc.get("selected_capsules", []),
                  _gold(q))


def score_bm25(q, snapshot, ontology, budget):
    k = max(1, budget // 64)
    return _score(_bm25_retrieve(q, snapshot, ontology, top_k=k), _gold(q))


def score_dense(q, snapshot, ontology, budget):
    k = max(1, budget // 64)
    return _score(_dense_retrieve(q, snapshot, ontology, top_k=k), _gold(q))


def score_graphrag(q, snapshot, ontology, budget):
    return _score(_graphrag_retrieve(q, snapshot), _gold(q))


def score_raw(q, snapshot, ontology, budget):
    return _score(_raw_retrieve(q, snapshot), _gold(q))


def score_onto(q, snapshot, ontology, budget):
    return _score(_onto_retrieve(q, snapshot), _gold(q))


SCORERS = {
    "OntoContext": score_oc,
    "BM25": score_bm25,
    "Dense (random)": score_dense,
    "GraphRAG": score_graphrag,
    "Raw neighbourhood": score_raw,
    # "Onto-embedding" is slow (spectral SVD on full ontology); skip for matched-budget
}


def run_one_family(family, snapshot, ontology, out_path=None):
    print(f"=== {family} | {snapshot} | N={N_QUESTIONS} ===", flush=True)
    qs = _load_qs(family, snapshot, ontology, N_QUESTIONS)
    rows = []
    t_start = time.time()
    for method, scorer in SCORERS.items():
        if method == "OntoContext":
            for b in BUDGETS:
                t0 = time.time()
                n_ok = sum(scorer(q["question"], snapshot, ontology, b) for q in qs)
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
                print(f"  OC@{b:4d}: {n_ok}/{len(qs)} = {acc:.0%} ({dt:.1f}s)",
                      flush=True)
                # Write incrementally to allow recovery
                if out_path:
                    write_table(rows, out_path)
        else:
            t0 = time.time()
            n_ok = sum(scorer(q["question"], snapshot, ontology, 256) for q in qs)
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
            print(f"  {method:18s}: {n_ok}/{len(qs)} = {acc:.0%} ({dt:.1f}s)",
                  flush=True)
            if out_path:
                write_table(rows, out_path)
    print(f"=== {family} done in {time.time()-t_start:.1f}s ===", flush=True)
    return rows



def write_table(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
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
    print(f"Wrote: {out_path}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument("--family", required=True,
                   choices=list(_QUESTION_LOADERS.keys()))
    p.add_argument("--out", required=True)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    ontology = "SYNTH" if "synth" in args.snapshot else "CL"
    rows = run_one_family(args.family, args.snapshot, ontology, args.out)
    write_table(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
