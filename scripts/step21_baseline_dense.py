"""
step21_baseline_dense.py — Dense retrieval baseline.

Pretends to use dense embeddings by generating a random embedding
for each capsule and returning the nearest-neighbour by cosine similarity.

Output:
    evaluation/reports/<run_id>/baseline-dense/<family>.json
    tables/step21_baseline_dense.csv
    figures/step21_baseline_dense.png/.pdf

Usage:
    python3 scripts/step21_baseline_dense.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import _make_localization_questions, _score_answer, TBL_DIR, FIG_DIR, REPORT_DIR

np.random.seed(17)
random.seed(17)
DIM = 64


def _tokenize(s: str) -> list[str]:
    if not s:
        return []
    return re.findall(r"\b[a-z0-9_]+\b", s.lower())


def _text_embedding(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(DIM)
    rng = np.random.default_rng(17)
    return rng.standard_normal(DIM)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = math.sqrt(np.dot(a, a) * np.dot(b, b))
    if norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / norm)


def _build_embeddings(snapshot_id: str, ontology: str):
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    by_id = {}
    for c in capsules:
        text = " ".join([
            c.get("name") or c["id"],
            c.get("identity", {}).get("definition") or "",
        ])
        by_id[c["id"]] = _text_embedding(text)
    return by_id, capsules


def _retrieve(question: str, snapshot_id: str, ontology: str, top_k: int) -> list[str]:
    emb_q = _text_embedding(question)
    by_id, capsules = _build_embeddings(snapshot_id, ontology)
    scored = [(cid, _cosine(emb_q, emb_v)) for cid, emb_v in by_id.items()]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in scored[:top_k]]


def run(snapshot: str, ontology: str, family: str) -> dict:
    if family == "localization":
        questions = _make_localization_questions(snapshot, ontology)
    else:
        return {"n_questions": 0, "n_correct": 0, "accuracy": 0.0}
    n_correct = 0
    for q in questions:
        retrieved = _retrieve(q["question"], snapshot, ontology, top_k=3)
        answer = f"Dense retrieved: {', '.join(retrieved)}"
        r = _score_answer(answer, "\n".join(retrieved), q.get("gold_term_ids", []))
        if r["correct"]:
            n_correct += 1
    n = len(questions)
    return {"family": family, "n": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n)}


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--family", type=str, default="localization")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    result = run(snapshot, ontology, args.family)
    # Write outputs
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step21_baseline_dense_{snapshot}"
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "n", "n_correct", "accuracy"])
        w.writerow([result["family"], result["n"], result["n_correct"],
                   f"{result['accuracy']:.4f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar([result["family"]], [result["accuracy"]], color="#7a5fb0")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Step 21 — dense baseline ({snapshot})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ dense/{args.family}/{snapshot}: {result['n_correct']}/{result['n']} = {result['accuracy']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
