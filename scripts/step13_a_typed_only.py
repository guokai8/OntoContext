"""
step13_a_typed_only.py — typed-only relation ablation.

Sweep the relation-type filter used in the relation-completion
benchmark:
    all          — all relations in defining_relations
    is_a_only   — only BFO_0000050 (is_a)
    non_is_a     — only non-is_a relations
    typed        — only the *typed* relations (RO, BFO non-is_a)

Ablation point: the renderer and benchmark may behave differently
for typed vs untyped relations, especially for the cross-ontology
sibling-discrimination task.

Output:
    tables/step13_a_typed_only.csv
    figures/step13_a_typed_only.png/.pdf

Usage:
    python3 scripts/step13_a_typed_only.py --synthetic synth_d02_b02_s017
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
from step13_bench_relation import _make_relation_questions, _score_relation
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)


def _filter(questions: list[dict], kind: str) -> list[dict]:
    """Filter each question's gold_relations to a subset."""
    out: list[dict] = []
    for q in questions:
        rels = q.get("gold_relations", [])
        if kind == "all":
            keep = rels
        elif kind == "is_a_only":
            keep = [r for r in rels if "BFO_0000050" in r]
        elif kind == "non_is_a":
            keep = [r for r in rels if "BFO_0000050" not in r]
        elif kind == "typed":
            # typed = RO_xxxx or BFO non-is_a
            keep = [r for r in rels if "RO_" in r or
                    ("BFO_" in r and "BFO_0000050" not in r)]
        else:
            raise RuntimeError(f"unknown kind: {kind}")
        if not keep:
            continue
        new_q = dict(q)
        new_q["gold_relations"] = keep
        out.append(new_q)
    return out


def _sweep(snapshot: str, ontology: str, budget: int) -> list[dict]:
    rows: list[dict] = []
    questions = _make_relation_questions(snapshot, ontology)
    for kind in ["all", "is_a_only", "non_is_a", "typed"]:
        fqs = _filter(questions, kind)
        scores: list[dict] = []
        for q in fqs:
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_relation(answer, q.get("gold_relations", []))
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
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(range(len(kinds)), accs, color="#7a5fb0")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Step 13a — typed-only relation sweep")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f"n={rows[i]['n']}", ha="center", fontsize=8)
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
    basename = f"step13_a_typed_only_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} kinds; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
