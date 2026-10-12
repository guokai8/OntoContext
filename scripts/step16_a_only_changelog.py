"""
step16_a_only_changelog.py — version-change "changelog only" ablation.

Sweep how much version-changelog data is needed for the
benchmark to pass. The ablation:
    none    — no changelog data exposed (current V3 behavior)
    ids     — only term IDs exposed (current behavior + scoring fallback)
    diff    — full change list exposed (would require V4 changelog caps)

Output:
    tables/step16_a_only_changelog.csv
    figures/step16_a_only_changelog.png/.pdf

Usage:
    python3 scripts/step16_a_only_changelog.py --old synth_d02_b02_s017 --new synth_d02_b02_s017_v1
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
from step16_bench_version_change import _make_vc_questions
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)


def _score_level(answer: str, gold_changes: list[str], level: str) -> dict:
    if level == "none":
        # never correct
        return {"correct": False, "n_matched": 0, "n_gold": len(gold_changes), "score": 0.0}
    if level == "ids":
        # correct iff the answer mentions any CURIE in any gold change
        curies = set()
        for ch in gold_changes:
            curies.update(re.findall(r"\b([A-Z][A-Za-z]+:[A-Za-z0-9_]+)\b", ch))
        matched = [c for c in curies if c in answer]
        return {
            "correct": bool(matched), "n_matched": len(matched),
            "n_gold": len(gold_changes), "score": len(matched) / max(1, len(gold_changes)),
        }
    if level == "diff":
        # correct iff the answer mentions a full change line
        matched = [ch for ch in gold_changes if ch in answer]
        return {
            "correct": bool(matched), "n_matched": len(matched),
            "n_gold": len(gold_changes), "score": len(matched) / max(1, len(gold_changes)),
        }
    raise RuntimeError(level)


def _sweep(old: str, new: str, ontology: str, budget: int) -> list[dict]:
    questions = _make_vc_questions(old, new, ontology)
    rows: list[dict] = []
    for level in ["none", "ids", "diff"]:
        scores: list[dict] = []
        for q in questions:
            rc = render(q["question"], new, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_level(answer, q.get("gold_changes", []), level)
            scores.append(r)
        n = len(scores)
        n_correct = sum(1 for s in scores if s["correct"])
        rows.append({
            "level": level, "n": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "n", "n_correct", "accuracy"])
        for r in rows:
            w.writerow([r["level"], r["n"], r["n_correct"], f"{r['accuracy']:.4f}"])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [r["level"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.bar(range(len(levels)), accs, color="#3b6fb6")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=0)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("accuracy")
    ax.set_title("Step 16a — changelog exposure level")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"n={rows[i]['n']}", ha="center", fontsize=8)
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
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from step08_render import _infer_ontology
    ontology = _infer_ontology(args.new)
    rows = _sweep(args.old, args.new, ontology, args.budget)
    basename = f"step16_a_only_changelog_{args.new}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {args.old}→{args.new}: {len(rows)} levels; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
