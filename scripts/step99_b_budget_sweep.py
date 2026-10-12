"""
step99_b_budget_sweep.py — Accuracy vs token budget on real CL.

Picks the 4 most-commonly-reported benchmarks (localization, hierarchy,
sibling-discrimination, relation-completion) and re-runs each at a
range of small-to-large token budgets:

    8, 16, 32, 64, 128, 256, 512, 1024, 2048

The result is a CSV + plot showing how each benchmark's accuracy
varies with budget. On real CL the larger benchmarks (localization,
hierarchy) are at ceiling once budget >= 64 because the renderer's
CURIE-priority lane always expands the queried term in the first 60%
of the budget. Smaller budgets (8, 16, 32) are where the priority
lane itself gets truncated, exposing the rising curve.

Usage:
    python3 scripts/step99_b_budget_sweep.py --snapshot cl_2024-06-10 \\
        --n-questions 50 --budgets 8,16,32,64,128,256,512,1024,2048 \\
        --out-prefix cl_sweep_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TBL_DIR = Path("tables")
FIG_DIR = Path("figures")


def _synth_questions(snap: str, ont: str, n: int) -> dict:
    """Return {family: [questions]} for the 4 families we sweep."""
    from step10_bench_localization import _make_localization_questions
    from step11_bench_hierarchy import _make_hierarchy_questions
    from step12_bench_sibling import _make_sibling_questions
    from step13_bench_relation import _make_relation_questions
    families = {
        "localization": _make_localization_questions(snap, ont),
        "hierarchy": _make_hierarchy_questions(snap, ont),
        "sibling-discrimination": _make_sibling_questions(snap, ont),
        "relation-completion": _make_relation_questions(snap, ont),
    }
    # Subsample each to n (deterministic seed)
    import random
    random.seed(17)
    for k, v in families.items():
        if len(v) > n:
            families[k] = random.sample(v, n)
    return families


def _score(family: str, q: dict, rc: dict) -> bool:
    """Re-use the per-family scorer from the benchmark script."""
    answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
             f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
             f"{rc['text']}"
    if family == "localization":
        from step10_bench_localization import _score_answer
        return _score_answer(answer, rc["text"], q.get("gold_term_ids")).get("correct", False)
    if family == "hierarchy":
        # Hierarchy uses _score_answer (CURIE-mention overlap with gold parents).
        from step10_bench_localization import _score_answer
        return _score_answer(answer, rc["text"], q.get("gold_term_ids", [])).get("correct", False)
    if family == "sibling-discrimination":
        from step12_bench_sibling import _score_sibling
        # Sibling scorer takes gold_features (distinguishing features of
        # the gold term, not the confusable).
        return _score_sibling(answer, q.get("gold_features", [])).get("correct", False)
    if family == "relation-completion":
        from step13_bench_relation import _score_relation
        return _score_relation(answer, q.get("gold_relations", [])).get("correct", False)
    raise ValueError(family)


def run_sweep(snap: str, ont: str, budgets: list[int], n_per_family: int,
              *, no_v3_fix: bool = False) -> dict:
    from step08_render import render as _render
    if no_v3_fix:
        # Flip the module-level flag the renderer reads at call time.
        import step08_render as _r
        _r.ONTO_NO_V3_FIX = True
    questions = _synth_questions(snap, ont, n_per_family)
    matrix: dict[str, dict[int, dict]] = {}
    t0 = time.time()
    for fam, qs in questions.items():
        matrix[fam] = {}
        print(f"  {fam}: {len(qs)} questions × {len(budgets)} budgets", flush=True)
        for b in budgets:
            n_correct = 0
            n_total = 0
            for q in qs:
                try:
                    rc = _render(q["question"], snap, ont, b)
                except Exception as e:
                    print(f"    ! render failed: {e}", flush=True)
                    continue
                n_total += 1
                if _score(fam, q, rc):
                    n_correct += 1
            matrix[fam][b] = {"n": n_total, "correct": n_correct,
                                "accuracy": n_correct / max(1, n_total)}
            print(f"    budget={b:5d}  {n_correct}/{n_total} = {matrix[fam][b]['accuracy']:.2%}",
                  flush=True)
    if no_v3_fix:
        import step08_render as _r
        _r.ONTO_NO_V3_FIX = False
    print(f"  total sweep time: {time.time() - t0:.1f}s", flush=True)
    return {"snapshot": snap, "ontology": ont,
            "budgets": budgets, "n_per_family": n_per_family, "matrix": matrix,
            "v3_fix_enabled": not no_v3_fix}


def _write_csv(sweep: dict, out_prefix: str) -> Path:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    out = TBL_DIR / f"{out_prefix}_budget_sweep.csv"
    families = sorted(sweep["matrix"].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["budget"] + families)
        for b in sweep["budgets"]:
            row = [b]
            for f in families:
                d = sweep["matrix"][f].get(b, {})
                row.append(f"{d.get('accuracy', 0.0):.4f}" if d else "")
            w.writerow(row)
    # Also write the per-budget n/correct detail
    out2 = TBL_DIR / f"{out_prefix}_budget_sweep_detail.csv"
    with out2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "budget", "n", "correct", "accuracy"])
        for f in families:
            for b in sweep["budgets"]:
                d = sweep["matrix"][f].get(b, {})
                if d:
                    w.writerow([f, b, d["n"], d["correct"], f"{d['accuracy']:.4f}"])
    return out


def _plot(sweep: dict, out_prefix: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = sorted(sweep["matrix"].keys())
    budgets = sweep["budgets"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    markers = {"localization": "o", "hierarchy": "s", "sibling-discrimination": "^",
                "relation-completion": "D"}
    colors = {"localization": "#3b6fb6", "hierarchy": "#7a5fb0",
              "sibling-discrimination": "#3b8f4f", "relation-completion": "#c0513b"}
    for f in families:
        accs = [sweep["matrix"][f].get(b, {}).get("accuracy", 0.0) for b in budgets]
        ax.plot(budgets, accs, marker=markers.get(f, "o"),
                color=colors.get(f, None), label=f, linewidth=1.6, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("token budget")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy vs token budget — {sweep['snapshot']} "
                  f"(n={sweep['n_per_family']}/family, "
                  f"{len(families)} families, "
                  f"v3_fix={'on' if sweep.get('v3_fix_enabled', True) else 'OFF'})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{out_prefix}_budget_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default="cl_2024-06-10")
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--budgets", type=str, default="8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--out-prefix", type=str, default="budget_sweep")
    p.add_argument("--no-v3-fix", action="store_true",
                   help="ablation: disable the V3 query-priority expansion block")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snap, ont = args.synthetic, "SYNTH"
    else:
        snap, ont = args.snapshot, args.snapshot.split("_")[0].upper()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    print(f"=== Budget sweep on {snap} ({ont}) ===", flush=True)
    print(f"  budgets: {budgets}", flush=True)
    print(f"  n_per_family: {args.n_questions}", flush=True)
    print(f"  v3_fix_enabled: {not args.no_v3_fix}", flush=True)
    sweep = run_sweep(snap, ont, budgets, args.n_questions,
                       no_v3_fix=args.no_v3_fix)
    csv_path = _write_csv(sweep, args.out_prefix)
    fig_path = _plot(sweep, args.out_prefix)
    # Also save the full JSON
    json_path = Path("tables") / f"{args.out_prefix}_budget_sweep.json"
    json_path.write_text(json.dumps(sweep, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ Wrote {csv_path}, {json_path}, {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
