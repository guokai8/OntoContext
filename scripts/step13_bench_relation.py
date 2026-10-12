"""
step13_bench_relation.py — Relation-completion benchmark.

Given a term, predict what relations it has (or might be missing).
The gold is the term's `logical_signature.defining_relations`.

Scoring: correct iff the answer mentions at least one of the gold
relations (as an IRI substring, e.g. "BFO_0000050" or "RO_0002131").

Outputs:
    evaluation/reports/<run_id>/relation-completion/accuracy.json
    tables/step13_bench_relation.csv/.tex
    figures/step13_bench_relation.png/.pdf

Usage:
    python3 scripts/step13_bench_relation.py --synthetic synth_d02_b02_s017
    python3 scripts/step13_bench_relation.py --snapshot go_2024-06-10
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
from step10_bench_localization import TBL_DIR, FIG_DIR, REPORT_DIR

# Determinism per the plan.
np.random.seed(17)


def _make_relation_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """For every term, generate 1 relation question. Gold = the
    term's defining_relations (any IRI)."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        rels = c.get("logical_signature", {}).get("defining_relations", []) or []
        if not rels:
            continue
        q_counter += 1
        out.append({
            "qid": f"relation-completion_{q_counter:03d}",
            "question": (
                f"What relations does {name} (term id {tid}) have? "
                f"List all OBO relations."
            ),
            "gold_relations": list(rels),
            "gold_term_ids": [tid],
            "family": "relation-completion",
            "term_id": tid,
        })
    return out


def _normalize_relation(rel: str) -> str:
    """Normalize a relation IRI / CURIE to its local, comparable form.

    Examples:
        "obo:BFO_0000050"   -> "bfo_0000050"
        "BFO_0000050"       -> "bfo_0000050"
        "http://purl.obolibrary.org/obo/BFO_0000050" -> "bfo_0000050"
        "RO_0002131"        -> "ro_0002131"
    """
    if not rel:
        return ""
    r = rel.strip()
    # Take last path segment after "/" or ":"
    for sep in ("/", ":"):
        if sep in r:
            r = r.rsplit(sep, 1)[-1]
    return r.lower()


def _score_relation(answer: str, gold_relations: list[str]) -> dict:
    """Score: answer mentions >= 1 of the gold relations (CURIE-aware).

    The matching normalizes both sides to a common local form
    (lowercased, prefix-stripped) and matches by substring of the
    local part. This catches:
        gold: 'obo:BFO_0000050'   answer: 'BFO_0000050'      ✓
        gold: 'BFO_0000050'       answer: 'part_of (BFO_0000050)' ✓
        gold: 'RO_0002131'        answer: 'ro_0002131'        ✓
        gold: 'http://.../BFO_50' answer: 'bfo_50'            ✓
    """
    answer_norm = (answer or "").lower()
    matched: list[str] = []
    matched_norm: list[str] = []
    for g in gold_relations:
        g_norm = _normalize_relation(g)
        if not g_norm:
            continue
        if g_norm in answer_norm:
            matched.append(g)
            matched_norm.append(g_norm)
    return {
        "correct": bool(matched),
        "matched": matched,
        "matched_normalized": matched_norm,
        "n_matched": len(matched),
        "n_gold": len(gold_relations),
        "score": len(matched) / max(1, len(gold_relations)),
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
        r = _score_relation(answer, q.get("gold_relations", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "relation-completion"
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
    out = REPORT_DIR / result["run_id"] / "relation-completion" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step13_bench_relation_{snapshot}"
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
    ax.set_title(f"Step 13 — relation-completion\n"
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
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--run-id", type=str, default="bench_rel")
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
    questions = _make_relation_questions(snapshot, ontology)
    if args.n_questions and len(questions) > args.n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, args.n_questions)
        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')
    if not questions:
        print(f"ERROR: no relation questions for {snapshot}", file=sys.stderr)
        return 2
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    print(
        f"✅ relation-completion/{snapshot}: "
        f"{result['n_correct']}/{result['n_questions']} = {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
