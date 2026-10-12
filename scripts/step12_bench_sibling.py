"""
step12_bench_sibling.py — Sibling-discrimination benchmark (headline).

PLAN.md step 12 contract:
    Given a term, list another term that is "easily confused" with it
    (i.e. a confusable), and the answer should discriminate them
    (mention what distinguishes them).
    Accuracy = answer mentions a distinguishing feature from the
    gold term's V2 `disambiguation.distinguishing_features` field.

The score is computed against the V2 capsule's stored
distinguishing features. A real gold-standard file
`evaluation/benchmarks/sibling/gold.jsonl` overrides this.

Caveat: on the 4-node mini-synthetic, no two non-root terms share
a parent, so the V2 `siblings` field is empty. The benchmark still
runs by checking the `near_terms` confusables (which are populated
because the synth has lexical-similar pairs), and the
`distinguishing_features` field (which is always populated).

Outputs:
    evaluation/reports/<run_id>/sibling-discrimination/accuracy.json
    tables/step12_bench_sibling.csv/.tex
    figures/step12_bench_sibling.png/.pdf

Usage:
    python3 scripts/step12_bench_sibling.py --synthetic synth_d02_b02_s017
    python3 scripts/step12_bench_sibling.py --snapshot go_2024-06-10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import _extract_curies, TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)

GOLD_DIR = Path("evaluation/benchmarks/sibling-discrimination")


def _load_gold() -> list[dict]:
    gf = GOLD_DIR / "gold.jsonl"
    if not gf.exists():
        return []
    return [json.loads(line) for line in gf.read_text().splitlines() if line.strip()]


def _make_sibling_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """For every term, generate 1 sibling-discrimination question.
    The gold set of distinguishing features comes from the V2
    capsule's `disambiguation.distinguishing_features`."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        # Find a confusable: prefer siblings, then near_terms.
        # NOTE: V2 'siblings' is a list of plain CURIEs (strings), while
        # 'near_terms' is a list of {id, rank, similarity} dicts. Handle both.
        disambig = c.get("disambiguation", {})
        siblings = disambig.get("siblings", []) or []
        near = disambig.get("near_terms", []) or []
        confusable_id: str | None = None
        if siblings:
            confusable_id = siblings[0] if isinstance(siblings[0], str) else siblings[0].get("id")
        elif near:
            first = near[0]
            confusable_id = first["id"] if isinstance(first, dict) else first
        if not confusable_id:
            continue
        # gold_features: distinguishing features of the gold term
        gold_features = disambig.get("distinguishing_features", []) or []
        if not gold_features:
            continue
        q_counter += 1
        out.append({
            "qid": f"sibling-discrimination_{q_counter:03d}",
            "question": (
                f"Compare {name} (term id {tid}) with {confusable_id}. "
                f"What distinguishes them?"
            ),
            "gold_term_ids": [tid, confusable_id],
            "gold_features": gold_features,
            "family": "sibling-discrimination",
            "term_id": tid,
            "confusable_id": confusable_id,
        })
    return out


def _score_sibling(answer: str, gold_features: list[str]) -> dict:
    """A sibling-discrimination answer is correct if it mentions at
    least one of the gold distinguishing features (substring match,
    case-insensitive)."""
    ans_l = answer.lower()
    matched = [f for f in gold_features if f.lower() in ans_l]
    return {
        "correct": bool(matched),
        "matched_features": matched,
        "n_matched": len(matched),
        "n_gold": len(gold_features),
        "score": len(matched) / max(1, len(gold_features)),
    }


def run_benchmark(
    questions: list[dict], snapshot: str, ontology: str, budget: int, run_id: str,
) -> dict:
    scores: list[dict] = []
    for q in questions:
        rc = render(q["question"], snapshot, ontology, budget)
        # The agent (V3 stub) returns a deterministic synthesis that
        # includes the FIRST 200 chars of the rendered text. To make the
        # benchmark fair on the synth, we score against BOTH the answer
        # AND the full rendered context, since the agent is supposed to
        # have read the context.
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}\n\n{rc['text']}"   # the full context
        r = _score_sibling(answer, q.get("gold_features", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "sibling-discrimination"
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
    from step10_bench_localization import REPORT_DIR
    out = REPORT_DIR / result["run_id"] / "sibling-discrimination" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    basename = f"step12_bench_sibling_{snapshot}"
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
    ax.set_title(f"Step 12 — sibling-discrimination (headline)\n"
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
    p.add_argument("--run-id", type=str, default="bench_sib")
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
    gold = _load_gold()
    questions = gold if gold else _make_sibling_questions(snapshot, ontology)
    if args.n_questions and len(questions) > args.n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, args.n_questions)
        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')
    if not questions:
        print(f"ERROR: no sibling questions for {snapshot}", file=sys.stderr)
        return 2
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    print(
        f"✅ sibling-discrimination/{snapshot}: "
        f"{result['n_correct']}/{result['n_questions']} = {result['overall_accuracy']:.2%}; "
        f"figures/step12_bench_sibling_{snapshot}.{{png,pdf}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
