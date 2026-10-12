"""
quick_pareto_cl.py — fast Pareto plot on real CL, OC only (no slow baselines).

Same output schema as step30_pareto_*.csv, but only OntoContext
at 6 budgets × N=50 questions. Baselines are skipped because
BM25 on 18k docs takes 5+ minutes per retriever.

Output:
    tables/quick_pareto_cl_2024-06-10.csv
    figures/quick_pareto_cl_2024-06-10.{png,pdf}
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str("scripts"))
from step08_render import render, _infer_ontology  # noqa: E402
from step10_bench_localization import _make_localization_questions, _score_answer  # noqa: E402

np.random.seed(17)
random.seed(17)

BUDGETS = [64, 128, 256, 512, 1024, 2048]
N_QUESTIONS = 50


def _oc_curve(snapshot: str, ontology: str) -> list[dict]:
    questions = _make_localization_questions(snapshot, ontology)
    if len(questions) > N_QUESTIONS:
        questions = random.sample(questions, N_QUESTIONS)
    rows: list[dict] = []
    for b in BUDGETS:
        t0 = time.time()
        n_correct = 0
        for q in questions:
            rc = render(q["question"], snapshot, ontology, b)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}\n\n{rc['text']}"
            r = _score_answer(answer, rc["text"], q.get("gold_term_ids", []))
            if r["correct"]:
                n_correct += 1
        rows.append({
            "method": f"OntoContext (budget={b})",
            "n_correct": n_correct,
            "n": len(questions),
            "accuracy": n_correct / max(1, len(questions)),
            "wall_seconds": time.time() - t0,
        })
        print(f"  budget={b:5d}: {n_correct}/{len(questions)} = "
              f"{rows[-1]['accuracy']:.2%} in {rows[-1]['wall_seconds']:.1f}s",
              flush=True)
    return rows


def _main():
    snapshot = "cl_2024-06-10"
    ontology = _infer_ontology(snapshot)
    rows = _oc_curve(snapshot, ontology)

    Path("tables").mkdir(parents=True, exist_ok=True)
    out_csv = Path("tables") / f"quick_pareto_{snapshot}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_correct", "n", "accuracy", "wall_seconds"])
        for r in rows:
            w.writerow([r["method"], r["n_correct"], r["n"],
                        f"{r['accuracy']:.4f}", f"{r['wall_seconds']:.2f}"])
    print(f"  → {out_csv}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    xs = [int(r["method"].split("=")[1].rstrip(")")) for r in rows]
    ys = [r["accuracy"] for r in rows]
    ax.plot(xs, ys, marker="o", linewidth=2, color="#3b6fb6", label="OntoContext")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("token budget")
    ax.set_ylabel("localization accuracy (n=50)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"OntoContext Pareto curve on real CL\n"
                 f"cl_2024-06-10, 18,964 terms, 2,349 modules, 51,001 edges")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png = Path("figures") / f"quick_pareto_{snapshot}.png"
    out_pdf = Path("figures") / f"quick_pareto_{snapshot}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_png}, {out_pdf}")


if __name__ == "__main__":
    _main()
