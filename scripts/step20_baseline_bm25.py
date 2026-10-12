"""
step20_baseline_bm25.py — BM25 baseline.

Standard BM25 retrieval over the V3 capsule texts (name +
definition + synonyms + features). For each question, retrieve
the top-k capsules and use the same scoring as the OntoContext
benchmarks (CURIE-mention for localization, etc.).

Output:
    evaluation/reports/<run_id>/baseline-bm25/accuracy.json
    tables/step20_baseline_bm25.csv/.tex
    figures/step20_baseline_bm25.png/.pdf

Usage:
    python3 scripts/step20_baseline_bm25.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import (
    _make_localization_questions, _score_answer, TBL_DIR, FIG_DIR, REPORT_DIR,
)
from step11_bench_hierarchy import _make_hierarchy_questions
from step13_bench_relation import _make_relation_questions

# Determinism per the plan.
np.random.seed(17)

K1 = 1.5
B = 0.75


def _tokenize(s: str) -> list[str]:
    """Lowercase alpha-numeric tokenize."""
    if not s:
        return []
    return re.findall(r"\b[a-z0-9_]+\b", s.lower())


def _bm25_index(docs: list[list[str]]):
    """Build BM25 inverted index from a list of tokenized documents.

    Returns: (idf, avgdl, doc_lens, postings)
        idf: dict[term, float]
        avgdl: float
        doc_lens: list[int]
        postings: dict[term, list[(doc_id, tf)]]
    """
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log((n - df_t + 0.5) / (df_t + 0.5) + 1)
            for t, df_t in df.items()}
    avgdl = sum(len(d) for d in docs) / max(1, n)
    doc_lens = [len(d) for d in docs]
    postings: dict[str, list[tuple[int, int]]] = {t: [] for t in idf}
    for i, d in enumerate(docs):
        for t, tf in Counter(d).items():
            postings[t].append((i, tf))
    return idf, avgdl, doc_lens, postings


def _bm25_score(
    query_tokens: list[str], idf: dict[str, float], avgdl: float,
    doc_lens: list[int], postings: dict[str, list[tuple[int, int]]],
) -> list[tuple[int, float]]:
    """Return [(doc_id, score), ...] sorted descending."""
    scores: dict[int, float] = {}
    for q in query_tokens:
        if q not in postings:
            continue
        for doc_id, tf in postings[q]:
            dl = doc_lens[doc_id]
            denom = tf + K1 * (1 - B + B * dl / max(1, avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf[q] * tf * (K1 + 1) / denom
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _build_corpus(snapshot_id: str, ontology: str) -> tuple[list[dict], list[list[str]]]:
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    docs: list[list[str]] = []
    for c in capsules:
        text = " ".join([
            c.get("name") or c["id"],
            c.get("identity", {}).get("definition") or "",
            " ".join(s.get("text", "") for s in c.get("identity", {}).get("synonyms", [])),
            " ".join(c.get("disambiguation", {}).get("distinguishing_features", [])),
        ])
        docs.append(_tokenize(text))
    return capsules, docs


def _bm25_retrieve(
    question: str, snapshot_id: str, ontology: str, top_k: int = 5,
) -> list[str]:
    """Return the top-k retrieved term IDs for a question."""
    capsules, docs = _build_corpus(snapshot_id, ontology)
    idf, avgdl, doc_lens, postings = _bm25_index(docs)
    q_tokens = _tokenize(question)
    ranked = _bm25_score(q_tokens, idf, avgdl, doc_lens, postings)
    return [capsules[d_id]["id"] for d_id, _ in ranked[:top_k]]


def run_benchmark(
    family: str, snapshot: str, ontology: str, top_k: int, run_id: str,
) -> dict:
    if family == "localization":
        questions = _make_localization_questions(snapshot, ontology)
    elif family == "hierarchy":
        questions = _make_hierarchy_questions(snapshot, ontology)
    elif family == "relation-completion":
        questions = _make_relation_questions(snapshot, ontology)
    else:
        raise RuntimeError(family)

    scores: list[dict] = []
    for q in questions:
        retrieved = _bm25_retrieve(q["question"], snapshot, ontology, top_k)
        answer = f"BM25 retrieved: {', '.join(retrieved)}"
        r = _score_answer(answer, "\n".join(retrieved), q.get("gold_term_ids", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = family
        scores.append(r)
    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    return {
        "run_id": run_id, "baseline": "bm25", "family": family,
        "top_k": top_k, "n_questions": n, "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n),
        "mean_score": sum(s["score"] for s in scores) / max(1, n),
        "per_question": scores,
    }


def _write_outputs(result: dict) -> None:
    out = REPORT_DIR / result["run_id"] / "baseline-bm25" / f"{result['family']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step20_baseline_bm25_{result['family']}_{result.get('snapshot', 'synth')}"
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["top_k", result["top_k"]])
        w.writerow(["n_questions", result["n_questions"]])
        w.writerow(["n_correct", result["n_correct"]])
        w.writerow(["overall_accuracy", f"{result['overall_accuracy']:.4f}"])


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--family", type=str, default="localization")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--run-id", type=str, default="baseline_bm25")
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
    result = run_benchmark(args.family, snapshot, ontology, args.top_k, args.run_id)
    result["snapshot"] = snapshot
    _write_outputs(result)
    print(
        f"✅ bm25/{args.family}/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"= {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
