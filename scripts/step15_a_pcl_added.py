"""
step15_a_pcl_added.py — cross-ontology "pcl added" ablation.

PLAN.md step 15a: re-run the cross-ontology benchmark after
*adding* a synthetic placeholder cross-ontology relation. The
point is to measure how the benchmark responds to growing the
cross-ontology index.

For the 4-node synth (no real xrefs), this script adds a
synthetic `pcl` relation to one capsule and re-runs the
benchmark. On real data, this is a no-op (the index is real).

Output:
    tables/step15_a_pcl_added.csv
    figures/step15_a_pcl_added.png/.pdf

Usage:
    python3 scripts/step15_a_pcl_added.py --synthetic synth_d02_b02_s017
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
from step15_bench_cross_ontology import _make_xo_questions, _score_xo
from step10_bench_localization import TBL_DIR, FIG_DIR

# Determinism per the plan.
np.random.seed(17)

ENTITIES = ["SYNTH:R000000000", "SYNTH:R000000001", "SYNTH:N000100000", "SYNTH:N000100001"]


def _inject_pcl(snapshot: str, ontology: str, n_rels: int) -> list[dict]:
    """Inject `n_rels` synthetic pcl cross-relations into the V3
    capsules, in-memory (no disk write). Returns the modified
    question list."""
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot, ontology)
    by_id = {c["id"]: c for c in capsules}
    for i in range(n_rels):
        src = ENTITIES[i % len(ENTITIES)]
        dst = ENTITIES[(i + 1) % len(ENTITIES)]
        if src == dst:
            continue
        c = by_id[src]
        c.setdefault("cross_ontology", {})
        c["cross_ontology"].setdefault("relations", []).append({
            "object_ontology": "PCL",
            "object_id": f"PCL:{i:07d}",
            "to": f"PCL:{i:07d}",
            "relation": "obo:RO_0002131",
        })
    # Build questions from the modified capsules
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        cross = c.get("cross_ontology", {}).get("relations", []) or []
        for rel in cross:
            q_counter += 1
            out.append({
                "qid": f"xo_pcl_{q_counter:03d}",
                "question": (
                    f"Find the PCL equivalent of {c.get('name', c['id'])} "
                    f"(term id {c['id']})."
                ),
                "gold_term_ids": [rel.get("object_id") or rel.get("to")],
                "gold_relation": rel,
                "family": "cross-ontology",
                "term_id": c["id"],
            })
    return out


def _sweep(snapshot: str, ontology: str, budget: int) -> list[dict]:
    rows: list[dict] = []
    for n_rels in [0, 1, 2, 4, 8]:
        questions = _inject_pcl(snapshot, ontology, n_rels)
        scores: list[dict] = []
        for q in questions:
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_xo(answer, q.get("gold_term_ids", []))
            scores.append(r)
        n = len(scores)
        n_correct = sum(1 for s in scores if s["correct"])
        rows.append({
            "n_rels": n_rels, "n_q": n, "n_correct": n_correct,
            "accuracy": n_correct / max(1, n),
        })
    return rows


def _write_csv(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n_rels", "n_q", "n_correct", "accuracy"])
        for r in rows:
            w.writerow([r["n_rels"], r["n_q"], r["n_correct"], f"{r['accuracy']:.4f}"])


def _plot(rows: list[dict], basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rels = [r["n_rels"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    n_qs = [r["n_q"] for r in rows]
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))
    color1 = "#3b6fb6"
    ax1.set_xlabel("# pcl cross-relations injected")
    ax1.set_ylabel("accuracy", color=color1)
    ax1.plot(n_rels, accs, marker="o", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.1)
    ax2 = ax1.twinx()
    color2 = "#7a5fb0"
    ax2.set_ylabel("# questions generated", color=color2)
    ax2.plot(n_rels, n_qs, marker="s", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    fig.suptitle("Step 15a — pcl-added cross-ontology sweep", y=1.02)
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
    basename = f"step15_a_pcl_added_{snapshot}"
    _write_csv(rows, basename)
    _plot(rows, basename)
    print(f"✅ {snapshot}: {len(rows)} pcl levels; "
          f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
