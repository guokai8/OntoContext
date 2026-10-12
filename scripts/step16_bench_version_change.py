"""
step16_bench_version_change.py — Version-change benchmark.

Given two snapshots (old, new), and a term, predict what changed
in the term between old and new. Gold = the diff of the term's
identity fields.

For V3 (no real version diffs available), we use a synthetic
diff: a small "v0" and "v1" snapshot of the same ontology.

Output:
    evaluation/reports/<run_id>/version-change/accuracy.json
    tables/step16_bench_version_change.csv/.tex
    figures/step16_bench_version_change.png/.pdf

Usage:
    python3 scripts/step16_bench_version_change.py --old synth_d02_b02_s017 --new synth_d02_b02_s017_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render
from step10_bench_localization import TBL_DIR, FIG_DIR, REPORT_DIR

# Determinism per the plan.
np.random.seed(17)


def _load_caps(snapshot_id: str, ontology: str) -> dict[str, dict]:
    from step07_build_world_model import _load_capsules
    return {c["id"]: c for c in _load_capsules(snapshot_id, ontology)}


def _compute_diff(old: dict, new: dict) -> list[str]:
    """Compute a list of human-readable changes between two capsules."""
    changes: list[str] = []
    for k in ("name", "obsolete"):
        if old.get(k) != new.get(k):
            changes.append(f"{k}: {old.get(k)!r} → {new.get(k)!r}")
    for k in ("parents", "selected_children"):
        old_v = set(old.get("coordinates", {}).get(k, []) or [])
        new_v = set(new.get("coordinates", {}).get(k, []) or [])
        added = new_v - old_v
        removed = old_v - new_v
        if added:
            changes.append(f"added-{k}: {sorted(added)}")
        if removed:
            changes.append(f"removed-{k}: {sorted(removed)}")
    return changes


def _make_vc_questions(old: str, new: str, ontology: str) -> list[dict]:
    old_caps = _load_caps(old, ontology)
    new_caps = _load_caps(new, ontology)
    all_ids = sorted(set(old_caps) | set(new_caps))
    out: list[dict] = []
    q_counter = 0
    for tid in all_ids:
        o = old_caps.get(tid)
        n = new_caps.get(tid)
        if o is None and n is not None:
            changes = [f"added: {tid} (new in {new})"]
        elif o is not None and n is None:
            changes = [f"removed: {tid} (in {old}, gone in {new})"]
        elif o and n:
            changes = _compute_diff(o, n)
        else:
            continue
        if not changes:
            continue
        q_counter += 1
        out.append({
            "qid": f"version-change_{q_counter:03d}",
            "question": (
                f"What changed for term {tid} between snapshot {old} "
                f"and snapshot {new}?"
            ),
            "gold_changes": changes,
            "gold_term_ids": [tid],
            "family": "version-change",
            "term_id": tid,
        })
    return out


def _score_vc(answer: str, gold_changes: list[str]) -> dict:
    """Score: answer mentions a recognizable substring of ≥ 1 gold change.
    Heuristic: split each gold change on `:` and check if the prefix
    (e.g. 'name', 'parents', 'added') appears in the answer. As a
    secondary signal, check that the answer mentions the term ID
    mentioned in the gold (e.g. 'SYNTH:N000100000')."""
    ans_l = answer.lower()
    matched: list[str] = []
    for ch in gold_changes:
        prefix = ch.split(":", 1)[0].strip().lower()
        if prefix and prefix in ans_l:
            matched.append(ch)
            continue
        # Secondary: does the answer mention any CURIEs from the change?
        import re
        curies_in_change = re.findall(r"\b([A-Z][A-Za-z]+:[A-Za-z0-9_]+)\b", ch)
        for cur in curies_in_change:
            if cur in answer:
                matched.append(ch)
                break
    return {
        "correct": bool(matched),
        "matched": matched,
        "n_matched": len(matched),
        "n_gold": len(gold_changes),
        "score": len(matched) / max(1, len(gold_changes)),
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
        r = _score_vc(answer, q.get("gold_changes", []))
        r["question"] = q["question"]
        r["qid"] = q.get("qid", "")
        r["family"] = "version-change"
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
    out = REPORT_DIR / result["run_id"] / "version-change" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    basename = f"step16_bench_version_change_{snapshot}"
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
    ax.set_title(f"Step 16 — version-change ({snapshot})")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--run-id", type=str, default="bench_vc")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from step08_render import _infer_ontology
    ontology = _infer_ontology(args.new)
    questions = _make_vc_questions(args.old, args.new, ontology)
    if not questions:
        print(f"⚠️ no version-change questions between {args.old} and {args.new}")
        return 0
    result = run_benchmark(questions, args.new, ontology, args.budget, args.run_id)
    _write_outputs(result, args.new)
    print(
        f"✅ version-change/{args.old}→{args.new}: "
        f"{result['n_correct']}/{result['n_questions']} = {result['overall_accuracy']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
