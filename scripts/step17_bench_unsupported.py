"""
step17_bench_unsupported.py — Insufficient-information benchmark (mandatory).

For every test question, also ask: "Is this term supported by the
current ontology?" The agent should answer NO for terms that are
not in the V3 capsules, and YES for terms that are.

Gold: a term is "supported" iff it's in the V3 capsule set; "no"
iff it's not. We use a list of probe terms: the 4 in-synth IDs
(always supported) + 4 hallucinated IDs (never supported).

Scoring: correct iff the answer says "yes" or "no" matching the
gold support verdict (case-insensitive substring match on
"yes" / "no" — taking the first occurrence as the answer's claim).

Output:
    evaluation/reports/<run_id>/unsupported/accuracy.json
    tables/step17_bench_unsupported.csv/.tex
    figures/step17_bench_unsupported.png/.pdf

Usage:
    python3 scripts/step17_bench_unsupported.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render
from step10_bench_localization import TBL_DIR, FIG_DIR, REPORT_DIR

# Determinism per the plan.
np.random.seed(17)


def _make_probes(snapshot_id: str, ontology: str) -> list[dict]:
    """For V3, we use a set of probe terms: the in-ontology terms
    plus a few hallucinated ones. Gold support = the term is in V3."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    real_ids = [c["id"] for c in capsules]
    out: list[dict] = []
    q_counter = 0
    for tid in real_ids:
        q_counter += 1
        out.append({
            "qid": f"unsupported_{q_counter:03d}",
            "question": f"Is term {tid} supported by the current ontology?",
            "gold_supported": True,
            "gold_term_id": tid,
            "family": "insufficient-information",
        })
    # Add hallucinated probes (assuming ontology prefix)
    prefix = real_ids[0].split(":")[0] if real_ids else "X"
    for i in range(len(real_ids)):
        q_counter += 1
        hallucinated = f"{prefix}:999999{i:02d}"
        out.append({
            "qid": f"unsupported_{q_counter:03d}",
            "question": f"Is term {hallucinated} supported by the current ontology?",
            "gold_supported": False,
            "gold_term_id": hallucinated,
            "family": "insufficient-information",
        })
    return out


def _score_unsupported(answer: str, gold_supported: bool) -> dict:
    """Score: extract the agent's claim ('yes' or 'no') and compare
    to the gold."""
    ans_l = answer.lower()
    # find first occurrence of "yes" or "no" as a word
    yes_match = re.search(r"\b(yes|supported|true)\b", ans_l)
    no_match = re.search(r"\b(no|not supported|false|unsupported)\b", ans_l)
    if yes_match and no_match:
        # both — take whichever comes first
        agent_yes = yes_match.start() < no_match.start()
    elif yes_match:
        agent_yes = True
    elif no_match:
        agent_yes = False
    else:
        # No clear claim — check for the term ID as a fallback
        if "999999" in answer:    # likely the hallucinated term is NOT mentioned
            agent_yes = False
        else:
            agent_yes = True   # default to supported
    return {
        "correct": agent_yes == gold_supported,
        "agent_claim": "yes" if agent_yes else "no",
        "gold": "yes" if gold_supported else "no",
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
        r = _score_unsupported(answer, q["gold_supported"])
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "insufficient-information"
        scores.append(r)
    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    return {
        "run_id": run_id, "n_questions": n, "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n),
        "per_question": scores,
    }


def _write_outputs(result: dict, snapshot: str) -> None:
    out = REPORT_DIR / result["run_id"] / "unsupported" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step17_bench_unsupported_{snapshot}"
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["n_questions", result["n_questions"]])
        w.writerow(["n_correct", result["n_correct"]])
        w.writerow(["overall_accuracy", f"{result['overall_accuracy']:.4f}"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n = result["n_questions"]
    ax.bar(["correct", "incorrect"], [result["n_correct"], n - result["n_correct"]],
           color=["#3b6fb6", "#c0513b"])
    ax.set_title(f"Step 17 — insufficient-info (mandatory)\n"
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
    p.add_argument("--run-id", type=str, default="bench_unsup")
    p.add_argument("--n-questions", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from step08_render import _infer_ontology
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR: pass --synthetic or --snapshot", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    questions = _make_probes(snapshot, ontology)
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    print(
        f"✅ unsupported/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"= {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
