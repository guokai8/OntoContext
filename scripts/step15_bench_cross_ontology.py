"""
step15_bench_cross_ontology.py — Cross-ontology benchmark.

Given a term in ontology A, find its equivalent in ontology B.
The gold is the V3 capsule's `cross_ontology.relations` list.

For the 4-node synth, the cross_ontology_index has 0 rows, so the
benchmark will produce 0 questions (or 100% if we have any).
The script is still useful on real multi-ontology snapshots.

Output:
    evaluation/reports/<run_id>/cross-ontology/accuracy.json
    tables/step15_bench_cross_ontology.csv/.tex
    figures/step15_bench_cross_ontology.png/.pdf

Usage:
    python3 scripts/step15_bench_cross_ontology.py --synthetic synth_d02_b02_s017
    python3 scripts/step15_bench_cross_ontology.py --snapshot go_2024-06-10
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
from step08_render import render, _infer_ontology
from step10_bench_localization import TBL_DIR, FIG_DIR, REPORT_DIR

# Determinism per the plan.
np.random.seed(17)


def _make_xo_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """For every capsule that has cross-ontology relations, generate
    a question asking for the equivalent term in the other ontology."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        cross = c.get("cross_ontology", {}).get("relations", []) or []
        if not cross:
            continue
        # Use the first cross-relation
        rel = cross[0]
        q_counter += 1
        out.append({
            "qid": f"cross-ontology_{q_counter:03d}",
            "question": (
                f"Find the {rel.get('object_ontology', 'other ontology')} "
                f"equivalent of {c.get('name', c['id'])} (term id {c['id']})."
            ),
            "gold_term_ids": [rel.get("object_id") or rel.get("to")],
            "gold_relation": rel,
            "family": "cross-ontology",
            "term_id": c["id"],
        })
    return out


def _extract_curies(text: str) -> list[str]:
    """Extract OBO-style CURIEs from text (uppercase prefix + alnum)."""
    return list(dict.fromkeys(re.findall(r"\b([A-Z][A-Za-z0-9]*:[A-Z0-9_]{3,12})\b", text or "")))


def _normalize_iri(iri: str) -> str:
    """Normalize an IRI / CURIE to its comparable local form (lowercase)."""
    if not iri:
        return ""
    r = iri.strip()
    for sep in ("/", ":"):
        if sep in r:
            r = r.rsplit(sep, 1)[-1]
    return r.lower()


def _score_xo(answer: str, gold_term_ids: list[str]) -> dict:
    """Score by CURIE-aware matching: extract all CURIEs from the
    answer, normalize both gold and answer to lowercase local form,
    and check overlap. Catches:
        gold: 'FMA:79662'                answer: 'FMA:79662 (xref)'      ✓
        gold: 'http://.../FMA_79662'     answer: 'fma_79662'              ✓
        gold: 'UBERON:0004794'           answer: 'see UBERON_0004794 …'   ✓
    """
    answer_curies = _extract_curies(answer)
    answer_norm = {(c.split(":")[-1] if ":" in c else c).lower() for c in answer_curies}
    matched: list[str] = []
    for g in gold_term_ids:
        if not g:
            continue
        g_norm = _normalize_iri(g)
        if g_norm in answer_norm:
            matched.append(g)
    return {
        "correct": bool(matched),
        "matched": matched,
        "answer_curies": answer_curies,
        "n_matched": len(matched),
        "n_gold": len([g for g in gold_term_ids if g]),
        "score": len(matched) / max(1, len([g for g in gold_term_ids if g])),
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
        r = _score_xo(answer, q.get("gold_term_ids", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "cross-ontology"
        scores.append(r)
    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    return {
        "run_id": run_id, "n_questions": n, "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n),
        "mean_score": sum(s["score"] for s in scores) / max(1, n),
        "per_question": scores,
    }


def _write_outputs(result: dict, snapshot: str) -> None:
    out = REPORT_DIR / result["run_id"] / "cross-ontology" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step15_bench_cross_ontology_{snapshot}"
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
    ax.set_title(f"Step 15 — cross-ontology ({snapshot})")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--run-id", type=str, default="bench_xo")
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
    questions = _make_xo_questions(snapshot, ontology)
    if args.n_questions and len(questions) > args.n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, args.n_questions)
        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')
    if not questions:
        print(f"⚠️ {snapshot}: no cross-ontology questions (0 cross-relations); "
              f"writing empty accuracy file")
        result = {"run_id": args.run_id, "n_questions": 0, "n_correct": 0,
                  "overall_accuracy": 0.0, "mean_score": 0.0, "per_question": []}
        _write_outputs(result, snapshot)
        return 0
    result = run_benchmark(questions, snapshot, ontology, args.budget, args.run_id)
    _write_outputs(result, snapshot)
    print(
        f"✅ cross-ontology/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"= {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
