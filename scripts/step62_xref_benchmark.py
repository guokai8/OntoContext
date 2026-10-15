"""
step62_xref_benchmark.py — Cross-ontology benchmark on CL.

A non-ontology benchmark: for each (CL_term, xref_target) pair
extracted from the V1 capsules' cross_ontology.relations field,
ask the renderer: "What is the cross-ontology reference for
{term}?" and check whether the correct xref target is in the
rendered context.

This proves the compiler-renderer pipeline generalizes beyond
hierarchical ontology queries to the cross-domain case, which
is the "non-ontology benchmark" the external review asked for
(Wikidata, biomedical KG, etc. — we use the CL→Reactome/CHEBI/GO
xrefs that are already in the data).

Output:
    tables/step62_xref_benchmark.csv
    figures/step62_xref_benchmark.png/.pdf

Usage:
    python3 scripts/step62_xref_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter
from time import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)

from step08_render import render

np.random.seed(17)
random.seed(17)

BUDGETS = [64, 128, 256, 512, 1024]
N_QUESTIONS = 10


def load_xref_pairs() -> list[tuple[str, str, str]]:
    """Load (cl_term_id, xref_target, relation_type) triples from V1 capsules."""
    cap_dir = Path("capsules/v1/entities/CL")
    pairs = []
    for f in cap_dir.glob("*.json"):
        with open(f) as fp:
            c = json.load(fp)
        co = c.get("cross_ontology", {})
        for r in co.get("relations", []):
            target = r.get("to") or r.get("object_id") or r.get("object_iri")
            if not target:
                continue
            pairs.append((c["id"], target, r.get("relation", "xref")))
    return pairs


def main() -> int:
    pairs = load_xref_pairs()
    print(f"Loaded {len(pairs)} (term, xref) pairs", flush=True)
    if len(pairs) < N_QUESTIONS:
        print("ERROR: not enough pairs", file=sys.stderr)
        return 2
    # Sample
    random.seed(17)
    sample = random.sample(pairs, N_QUESTIONS)
    print(f"Sampled {N_QUESTIONS} pairs:", flush=True)
    for s in sample[:3]:
        print(f"  {s[0]} --{s[2]}--> {s[1]}", flush=True)

    # Score OntoContext at each budget
    print(f"\n=== OntoContext at 5 budgets ===", flush=True)
    out_path = Path("tables/step62_xref_benchmark.csv")
    out_path.parent.mkdir(exist_ok=True)
    rows = []
    for b in BUDGETS:
        n_ok = 0
        t0 = time()
        for cl_id, target, rel in sample:
            q = f"What is the cross-ontology reference for term {cl_id}? (term id {cl_id})"
            rc = render(q, "cl_2024-06-10", "CL", b)
            # Check if the target CURIE appears in selected_capsules or selected_modules
            text = " ".join(rc.get("selected_modules", [])
                            + rc.get("selected_capsules", [])
                            + [rc.get("text", "")])
            # Match target even if it has prefix differences (Reactome:R-HSA-...)
            # Extract all CURIEs from text
            curies = set(re.findall(r"\b([A-Z][A-Z0-9]+:[A-Z0-9_\-]+)\b", text))
            # Also try the bare ID (the part after the colon)
            target_prefix, target_id = target.split(":", 1) if ":" in target else ("", target)
            matched = (
                target in curies
                or any(target_id in c for c in curies)
            )
            if matched:
                n_ok += 1
        dt = time() - t0
        acc = n_ok / len(sample)
        print(f"  OC@{b:4d} tok: {n_ok}/{len(sample)} = {acc:.0%}  ({dt:.1f}s)",
              flush=True)
        rows.append({
            "method": f"OntoContext (budget={b})",
            "n_correct": n_ok,
            "n": len(sample),
            "accuracy": acc,
        })

    # Write CSV
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "n_correct", "n", "accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "method": r["method"],
                "n_correct": r["n_correct"],
                "n": r["n"],
                "accuracy": f"{r['accuracy']:.4f}",
            })
    print(f"\nWrote: {out_path}", flush=True)

    # Build figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    oc_b = [b for b in BUDGETS]
    oc_a = [r["accuracy"] for r in rows]
    ax.plot(oc_b, oc_a, marker="o", linewidth=2.5, color="#2563eb",
            label="OntoContext", zorder=5)
    ax.scatter(oc_b, oc_a, color="#2563eb", s=80, zorder=6)
    for b, a in zip(oc_b, oc_a):
        ax.annotate(f"{a:.0%}", (b, a),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#2563eb")
    ax.set_xscale("log")
    ax.set_xlim(40, 2048)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("token budget", fontsize=11)
    ax.set_ylabel("accuracy (cross-ontology xref retrieval)", fontsize=11)
    ax.set_title(
        f"Cross-ontology benchmark on CL 2024-06-10\n"
        f"(N={N_QUESTIONS} (term, xref) pairs; xrefs to Reactome/CHEBI/GO)",
        fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_path = Path("figures/step62_xref_benchmark.png")
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
