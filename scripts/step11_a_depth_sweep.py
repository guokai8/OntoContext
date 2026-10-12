"""
step11_a_depth_sweep.py — Hierarchy depth ablation.

For the hierarchy benchmark, sweep the depth (number of ancestors
returned). OntoContext's V1 selects ancestors by descending
Utility(e) and truncates at ANCESTOR_BUDGET. This ablation
measures accuracy vs budget size.

We simulate the depth effect by re-rendering the question with
different budgets, and for each budget we recompute the accuracy.
This is a coarse measure of "how much context depth is needed
to answer hierarchy questions correctly".

Output:
    tables/step11_a_depth_sweep.csv
    figures/step11_a_depth_sweep.png/.pdf

Usage:
    python3 scripts/step11_a_depth_sweep.py --synthetic synth_d02_b02_s017
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
from step10_bench_localization import _extract_curies, _score_answer
from step11_bench_hierarchy import _make_hierarchy_questions, _load_gold

# Determinism per the plan.
np.random.seed(17)

BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096]


def _run_depth_sweep(
    questions: list[dict], snapshot: str, ontology: str,
) -> list[dict]:
    rows: list[dict] = []
    for budget in BUDGETS:
        scores: list[dict] = []
        for q in questions:
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}"
            r = _score_answer(answer, rc["text"], q.get("gold_term_ids", []))
            r["n_capsules"] = len(rc["selected_capsules"])
            scores.append(r)
        n = len(scores)
        n_correct = sum(1 for s in scores if s["correct"])
        n_caps = sum(s["n_capsules"] for s in scores) / max(1, n)
        rows.append({
            "budget": budget,
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
            "mean_n_capsules": n_caps,
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    from step10_bench_localization import TBL_DIR
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["budget", "n", "n_correct", "accuracy", "mean_n_capsules"])
        for r in rows:
            w.writerow([r["budget"], r["n"], r["n_correct"],
                        f"{r['accuracy']:.4f}", f"{r['mean_n_capsules']:.2f}"])


def _plot(rows: list[dict], basename: str) -> None:
    from step10_bench_localization import FIG_DIR
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets = [r["budget"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    n_caps = [r["mean_n_capsules"] for r in rows]
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
    color1 = "#3b6fb6"
    ax1.set_xlabel("token budget")
    ax1.set_ylabel("accuracy", color=color1)
    ax1.plot(budgets, accs, marker="o", color=color1, label="accuracy")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    color2 = "#7a5fb0"
    ax2.set_ylabel("mean #capsules rendered", color=color2)
    ax2.plot(budgets, n_caps, marker="s", color=color2, label="#capsules")
    ax2.tick_params(axis="y", labelcolor=color2)
    fig.suptitle(f"Step 11a — hierarchy depth sweep", y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
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
    rows = _run_depth_sweep(questions, snapshot, ontology)
    basename = f"step11_a_depth_sweep_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} budgets; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
