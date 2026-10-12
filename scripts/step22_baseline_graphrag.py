"""
step22_baseline_graphrag.py — GraphRAG-style neighbourhood baseline.

Retrieves all immediate neighbours (1-hop) of the queried term
from the V3 capsules. If the term is not in the ontology, returns
the global skeleton (same as OntoContext V3's fallback).

Output:
    tables/step22_baseline_graphrag.csv
    figures/step22_baseline_graphrag.png/.pdf

Usage:
    python3 scripts/step22_baseline_graphrag.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import _infer_ontology
from step10_bench_localization import _make_localization_questions, _score_answer, TBL_DIR, FIG_DIR

np.random.seed(17)


def _tokenize(s: str):
    if not s:
        return []
    return re.findall(r"\b[a-z0-9_]+\b", s.lower())


def _retrieve(question: str, snapshot_id: str) -> list[str]:
    """GraphRAG neighbourhood: return all 1-hop neighbours of the queried term."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, "SYNTH")
    q_tokens = set(_tokenize(question))
    # Find the most-relevant capsule by name-token overlap
    best = None
    best_score = -1
    for c in capsules:
        name_tokens = set(_tokenize(c.get("name") or c["id"]))
        score = len(q_tokens & name_tokens)
        if score > best_score:
            best_score = score
            best = c
    if not best:
        return [c["id"] for c in capsules]
    parents = best.get("coordinates", {}).get("parents", []) or []
    children = best.get("coordinates", {}).get("selected_children", []) or []
    return parents + children + [best["id"]]


def run(snapshot: str, ontology: str) -> dict:
    questions = _make_localization_questions(snapshot, ontology)
    n_correct = 0
    for q in questions:
        retrieved = _retrieve(q["question"], snapshot)
        answer = f"GraphRAG neighbourhood: {', '.join(retrieved)}"
        r = _score_answer(answer, "\n".join(retrieved), q.get("gold_term_ids", []))
        if r["correct"]:
            n_correct += 1
    n = len(questions)
    return {"n": n, "n_correct": n_correct, "accuracy": n_correct / max(1, n)}


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
    result = run(snapshot, ontology)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step22_baseline_graphrag_{snapshot}"
    with (TBL_DIR / f"{basename}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["family", "n", "n_correct", "accuracy"])
        csv.writer(fh).writerow(["localization", result["n"], result["n_correct"],
                                 f"{result['accuracy']:.4f}"])
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(["localization"], [result["accuracy"]], color="#5b8c5a")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Step 22 — GraphRAG neighbourhood ({snapshot})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ graphrag/localization/{snapshot}: {result['n_correct']}/{result['n']} = {result['accuracy']:.2%}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
