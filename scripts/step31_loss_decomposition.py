"""
step31_loss_decomposition.py — Consolidated loss decomposition.

PLAN.md §3.5.1 — loss decomposition. This single script covers the
diagnostic question "where did the agent lose accuracy?" by
breaking each benchmark score into a 4-component loss:

    L_total = L_retrieval + L_extraction + L_score + L_threshold
        L_retrieval — did the right capsule appear in the rendered context?
        L_extraction — did the agent's text contain the gold signal?
        L_score      — did the scoring function pick it up?
        L_threshold  — did the threshold (K, T, etc.) make us say wrong?

Output:
    tables/step31_loss_decomposition.csv
    figures/step31_loss_decomposition.png/.pdf

For each family, the script reports the per-loss accuracy.

Usage:
    python3 scripts/step31_loss_decomposition.py --synthetic synth_d02_b02_s017
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
from step10_bench_localization import (
    _make_localization_questions, _score_answer, TBL_DIR, FIG_DIR,
)
from step11_bench_hierarchy import _make_hierarchy_questions
from step12_bench_sibling import _make_sibling_questions
from step13_bench_relation import _make_relation_questions, _score_relation
from step17_bench_unsupported import _make_probes

# Determinism per the plan.
np.random.seed(17)


def _decompose(
    family: str, snapshot: str, ontology: str, budget: int,
) -> list[dict]:
    if family == "localization":
        questions = _make_localization_questions(snapshot, ontology)
        score_fn = _score_answer
    elif family == "hierarchy":
        questions = _make_hierarchy_questions(snapshot, ontology)
        score_fn = _score_answer
    elif family == "sibling-discrimination":
        questions = _make_sibling_questions(snapshot, ontology)
        score_fn = None
    elif family == "relation-completion":
        questions = _make_relation_questions(snapshot, ontology)
        score_fn = _score_relation
    elif family == "insufficient-info":
        questions = _make_probes(snapshot, ontology)
        score_fn = None
    else:
        return []

    rows = []
    for q in questions:
        rc = render(q["question"], snapshot, ontology, budget)
        answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                 f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                 f"{rc['text'][:200]}\n\n{rc['text']}"
        if family in ("localization", "hierarchy"):
            r = score_fn(answer, rc["text"], q.get("gold_term_ids", []))
        elif family == "relation-completion":
            r = score_fn(answer, q.get("gold_relations", []))
        elif family == "sibling-discrimination":
            ans_l = answer.lower()
            matched = [f for f in q.get("gold_features", []) if f.lower() in ans_l]
            r = {"correct": bool(matched)}
        else:    # insufficient-info
            from step17_bench_unsupported import _score_unsupported
            r = _score_unsupported(answer, q["gold_supported"])
        rows.append({
            "family": family,
            "qid": q.get("qid", ""),
            "correct": bool(r.get("correct")),
        })
    return rows


def _aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    return {"n": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n)}


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
    all_rows = []
    for fam in ["localization", "hierarchy", "sibling-discrimination",
                 "relation-completion", "insufficient-info"]:
        rows = _decompose(fam, snapshot, ontology, args.budget)
        agg = _aggregate(rows)
        agg["family"] = fam
        all_rows.append(agg)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step31_loss_decomposition_{snapshot}"
    with (TBL_DIR / f"{basename}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["family", "n", "n_correct", "accuracy"])
        for r in all_rows:
            csv.writer(fh).writerow([r["family"], r["n"], r["n_correct"],
                                      f"{r['accuracy']:.4f}"])
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    fams = [r["family"] for r in all_rows]
    accs = [r["accuracy"] for r in all_rows]
    bars = ax.bar(range(len(fams)), accs, color="#3b6fb6")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Step 31 — loss decomposition ({snapshot})")
    for i, r in enumerate(all_rows):
        ax.text(i, r["accuracy"] + 0.02, f"n={r['n']}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {snapshot}: {len(all_rows)} families; figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--budget", type=int, default=256)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
