"""
step11_bench_hierarchy.py — Hierarchy benchmark.

PLAN.md step 11 contract:
    Hierarchy task: given a term, list its parents (or top-N ancestors).
    Accuracy = answer mentions ≥ 1 parent of the gold term.

For V3 (no external gold standard), the "gold" parent list is read
from the V3 capsule's `coordinates.parents` field. The agent
should mention at least one of those parents in its answer.

Outputs:
    evaluation/reports/<run_id>/hierarchy/accuracy.json
    tables/step11_bench_hierarchy.csv/.tex
    figures/step11_bench_hierarchy.png/.pdf

Plan verification:
    ≥ 1 benchmark run per task family; hierarchy question count
    must be ≥ 8 (per plan: "every V0/V1/V2/V3 capsule with a parent
    can yield a hierarchy question").

Usage:
    python3 scripts/step11_bench_hierarchy.py --synthetic synth_d02_b02_s017 --run-id bench_h
    python3 scripts/step11_bench_hierarchy.py --snapshot go_2024-06-10 --run-id bench_h_real
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import (
    _extract_curies, _score_answer, _load_gold,
    REPORT_DIR, FIG_DIR, TBL_DIR,
)

# Determinism per the plan.
np.random.seed(17)

GOLD_DIR = Path("evaluation/benchmarks/hierarchy")


def _make_hierarchy_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """Generate hierarchy questions for one snapshot.

    For each non-root capsule, generate:
      - "List the parents of {name} (term id {tid})."
      - "What is the direct parent of {name} (term id {tid})?"
    The gold term IDs are the capsule's parents. The term id is
    embedded in the question so the renderer's CURIE-priority path
    can lock onto the queried capsule on real ontologies.
    """
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        parents = c.get("coordinates", {}).get("parents", []) or []
        if not parents:
            continue
        for template in [
            f"List the parents of {name} (term id {tid}).",
            f"What is the direct parent of {name} (term id {tid})?",
        ]:
            q_counter += 1
            out.append({
                "qid": f"hierarchy_{q_counter:03d}",
                "question": template,
                "gold_term_ids": list(parents),
                "family": "hierarchy",
                "term_id": tid,
            })
    return out


def run_benchmark(
    questions: list[dict], snapshot: str, ontology: str, budget: int, run_id: str,
) -> dict:
    """Score every question, aggregate into a result dict."""
    scores: list[dict] = []
    for q in questions:
        rc = render(q["question"], snapshot, ontology, budget)
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}"
        r = _score_answer(answer, rc["text"], q.get("gold_term_ids", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = q.get("family", "hierarchy")
        scores.append(r)

    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    mean_score = sum(s["score"] for s in scores) / max(1, n)
    return {
        "run_id": run_id,
        "n_questions": n,
        "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n),
        "mean_score": mean_score,
        "per_question": scores,
    }


def _write_outputs(result: dict, snapshot: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / result["run_id"] / "hierarchy" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    TBL_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step11_bench_hierarchy_{snapshot}"
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
    correct = result["n_correct"]
    incorrect = n - correct
    ax.bar(["correct", "incorrect"], [correct, incorrect],
           color=["#3b6fb6", "#c0513b"])
    ax.set_ylabel("count")
    ax.set_title(f"Step 11 — hierarchy benchmark ({snapshot})\n"
                  f"accuracy={result['overall_accuracy']:.2%}")
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
    p.add_argument("--run-id", type=str, default="bench_h")
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

    gold = _load_gold("hierarchy")
    questions = gold if gold else _make_hierarchy_questions(snapshot, ontology)
    if args.n_questions and len(questions) > args.n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, args.n_questions)
        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')
    if not questions:
        print(f"ERROR: no hierarchy questions for {snapshot}", file=sys.stderr)
        return 2
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    acc = result["overall_accuracy"]
    print(
        f"✅ hierarchy/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"correct = {acc:.2%} accuracy; "
        f"figures/step11_bench_hierarchy_{snapshot}.{{png,pdf}} + "
        f"tables/step11_bench_hierarchy_{snapshot}.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
