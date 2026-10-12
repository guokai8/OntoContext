"""
step12_a_confusable_kind_sweep.py — sibling-discrimination ablation.

Sweep the confusable kind used to generate sibling-discrimination
questions:
    siblings_only     — use only the V2 `siblings` field
    near_terms_only   — use only the V2 `near_terms` field
    both              — union of both
    features_only     — use only `distinguishing_features` (no gold term)

For each kind, score the benchmark and report accuracy.

Output:
    tables/step12_a_confusable_kind_sweep.csv
    figures/step12_a_confusable_kind_sweep.png/.pdf

Usage:
    python3 scripts/step12_a_confusable_kind_sweep.py --synthetic synth_d02_b02_s017
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
from step12_bench_sibling import _score_sibling, _load_gold
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)

KINDS = ["siblings_only", "near_terms_only", "both", "features_only"]


def _make_questions_for_kind(snapshot_id: str, ontology: str, kind: str) -> list[dict]:
    """Build sibling-discrimination questions for a given confusable kind."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        d = c.get("disambiguation", {})
        siblings = d.get("siblings", []) or []
        near = d.get("near_terms", []) or []
        features = d.get("distinguishing_features", []) or []

        if kind == "siblings_only":
            confusable_id = siblings[0] if siblings else None
        elif kind == "near_terms_only":
            confusable_id = near[0]["id"] if near else None
        elif kind == "both":
            confusable_id = siblings[0] if siblings else (near[0]["id"] if near else None)
        elif kind == "features_only":
            if not features:
                continue
            confusable_id = (siblings + [n["id"] for n in near] or [None])[0]
        else:
            raise RuntimeError(f"unknown kind: {kind}")
        if not confusable_id or not features:
            continue
        q_counter += 1
        out.append({
            "qid": f"sib_{kind}_{q_counter:03d}",
            "question": (
                f"Compare {name} (term id {tid}) with {confusable_id}. "
                f"What distinguishes them?"
            ),
            "gold_term_ids": [tid, confusable_id],
            "gold_features": list(features),
            "family": "sibling-discrimination",
            "term_id": tid,
            "confusable_id": confusable_id,
        })
    return out


def _sweep_one(snapshot: str, ontology: str, budget: int) -> list[dict]:
    rows: list[dict] = []
    for kind in KINDS:
        questions = _make_questions_for_kind(snapshot, ontology, kind)
        scores: list[dict] = []
        for q in questions:
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_sibling(answer, q.get("gold_features", []))
            scores.append(r)
        n = len(scores)
        n_correct = sum(1 for s in scores if s["correct"])
        rows.append({
            "kind": kind, "n": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "n", "n_correct", "accuracy"])
        for r in rows:
            w.writerow([r["kind"], r["n"], r["n_correct"], f"{r['accuracy']:.4f}"])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [r["kind"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    n_qs = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    bars = ax.bar(range(len(kinds)), accs, color="#3b6fb6")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Step 12a — confusable-kind sweep")
    for i, (bar, n) in enumerate(zip(bars, n_qs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)
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
    rows = _sweep_one(snapshot, ontology, args.budget)
    basename = f"step12_a_confusable_kind_sweep_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} kinds; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
