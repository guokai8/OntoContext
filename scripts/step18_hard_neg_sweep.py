"""
step18_hard_neg_sweep.py — consolidated hard-negative sweep.

Sweeps hard-negative thresholds across the localization, hierarchy,
sibling-discrimination, and relation-completion benchmarks. Produces
a single per-benchmark × per-K table for the report.

Output:
    tables/step18_hard_neg_sweep.csv
    figures/step18_hard_neg_sweep.png/.pdf

Usage:
    python3 scripts/step18_hard_neg_sweep.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology
from step10_bench_localization import _extract_curies, _score_answer
from step11_bench_hierarchy import _make_hierarchy_questions
from step12_bench_sibling import _make_sibling_questions
from step13_bench_relation import _make_relation_questions, _score_relation
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 16]


def _answer_for(q, snapshot, ontology, budget):
    rc = render(q["question"], snapshot, ontology, budget)
    return f"Modules: {', '.join(rc['selected_modules'])}\n" \
           f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
           f"{rc['text'][:200]}\n\n{rc['text']}", rc["text"]


def _sweep_one_family(family: str, snapshot: str, ontology: str, budget: int) -> list[dict]:
    if family == "localization":
        questions = _make_hierarchy_questions(snapshot, ontology)   # no, use the right one
        from step10_bench_localization import _make_localization_questions
        questions = _make_localization_questions(snapshot, ontology)
    elif family == "hierarchy":
        questions = _make_hierarchy_questions(snapshot, ontology)
    elif family == "sibling-discrimination":
        questions = _make_sibling_questions(snapshot, ontology)
    elif family == "relation-completion":
        questions = _make_relation_questions(snapshot, ontology)
    else:
        raise RuntimeError(f"unknown family: {family}")

    rows: list[dict] = []
    for k in K_VALUES:
        n_correct = 0
        n_with_hn = 0
        n = 0
        for q in questions:
            answer, context = _answer_for(q, snapshot, ontology, budget)
            n += 1
            mentioned = set(_extract_curies(answer))
            context_ids = set(_extract_curies(context))
            gold = set(q.get("gold_term_ids", []))
            has_gold = bool(gold & mentioned)
            # hard-neg = any CURIE in the answer that is in the context
            # but NOT in the gold set (capped at K — but for K≤16 in a 4-node
            # synth, K=∞ so we just count every context non-gold CURIE)
            non_gold_in_context = context_ids - gold
            capped_hn = set(list(non_gold_in_context)[:k])
            has_hn = bool(mentioned & capped_hn)
            if has_gold and not has_hn:
                n_correct += 1
            if has_hn:
                n_with_hn += 1
        rows.append({
            "family": family, "k": k, "n": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
            "n_with_hn": n_with_hn,
            "hn_rate": n_with_hn / max(1, n),
        })
    return rows


def _sweep(snapshot: str, ontology: str, budget: int) -> list[dict]:
    out: list[dict] = []
    for fam in ["localization", "hierarchy", "sibling-discrimination",
                 "relation-completion"]:
        try:
            out.extend(_sweep_one_family(fam, snapshot, ontology, budget))
        except Exception as e:
            print(f"  ! {fam}: {e}", file=sys.stderr)
    return out


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "k", "n", "n_correct", "accuracy",
                    "n_with_hn", "hn_rate"])
        for r in rows:
            w.writerow([r["family"], r["k"], r["n"], r["n_correct"],
                        f"{r['accuracy']:.4f}", r["n_with_hn"],
                        f"{r['hn_rate']:.4f}"])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fam, rlist in by_fam.items():
        ks = [r["k"] for r in rlist]
        accs = [r["accuracy"] for r in rlist]
        rates = [r["hn_rate"] for r in rlist]
        axes[0].plot(ks, accs, marker="o", label=fam)
        axes[1].plot(ks, rates, marker="s", label=fam)
    axes[0].set_xlabel("K (max confusable rank)")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Accuracy vs K (consolidated)")
    axes[0].legend()
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("hard-neg rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Hard-neg rate vs K (consolidated)")
    axes[1].legend()
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
    basename = f"step18_hard_neg_sweep_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} cells; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
