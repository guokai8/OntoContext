"""
step17_a_threshold_calibration.py — insufficient-info threshold calibration.

Sweep the yes/no threshold for the insufficient-info benchmark.
The default scoring is "first word wins"; this ablation measures
how sensitive the score is to that heuristic.

Levels:
    first_word    — first 'yes' or 'no' in the answer (default)
    word_count    — majority of 'yes'/'no' occurrences
    explicit_only — only exact "(yes)" / "(no)" markers
    id_match      — score by whether the term ID is in the context

Output:
    tables/step17_a_threshold_calibration.csv
    figures/step17_a_threshold_calibration.png/.pdf

Usage:
    python3 scripts/step17_a_threshold_calibration.py --synthetic synth_d02_b02_s017
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
from step17_bench_unsupported import _make_probes
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)


def _score_level(answer: str, gold_supported: bool, term_id: str, level: str) -> dict:
    ans_l = answer.lower()
    yes_words = re.findall(r"\b(yes|supported|true|known|present)\b", ans_l)
    no_words = re.findall(r"\b(no|not supported|false|unsupported|absent)\b", ans_l)
    if level == "first_word":
        yes_m = re.search(r"\b(yes|supported|true)\b", ans_l)
        no_m = re.search(r"\b(no|not supported|false|unsupported)\b", ans_l)
        if yes_m and no_m:
            agent_yes = yes_m.start() < no_m.start()
        elif yes_m:
            agent_yes = True
        elif no_m:
            agent_yes = False
        else:
            agent_yes = "999999" not in answer
    elif level == "word_count":
        if len(yes_words) > len(no_words):
            agent_yes = True
        elif len(no_words) > len(yes_words):
            agent_yes = False
        else:
            agent_yes = "999999" not in answer
    elif level == "explicit_only":
        if "(yes)" in ans_l:
            agent_yes = True
        elif "(no)" in ans_l:
            agent_yes = False
        else:
            agent_yes = "999999" not in answer
    elif level == "id_match":
        # supported iff the term ID is in the answer's context
        agent_yes = term_id in answer and "999999" not in term_id
    else:
        raise RuntimeError(level)
    return {
        "correct": agent_yes == gold_supported,
        "agent_yes": agent_yes,
    }


def _sweep(snapshot: str, ontology: str, budget: int) -> list[dict]:
    questions = _make_probes(snapshot, ontology)
    rows: list[dict] = []
    for level in ["first_word", "word_count", "explicit_only", "id_match"]:
        n_correct = 0
        n = 0
        for q in questions:
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_level(answer, q["gold_supported"], q["gold_term_id"], level)
            n += 1
            if r["correct"]:
                n_correct += 1
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
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(range(len(levels)), accs, color="#3b6fb6")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("accuracy")
    ax.set_title("Step 17a — insufficient-info threshold calibration")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
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
    rows = _sweep(snapshot, ontology, args.budget)
    basename = f"step17_a_threshold_calibration_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} levels; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
