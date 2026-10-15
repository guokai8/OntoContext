"""step63_aggregate_models.py — Aggregate LLM-agent eval across all 4 models.

Reads tables/step63_runs/<model>/run1_summary.csv for each model and
emits:
  - tables/step63_model_comparison.csv (one row per model × condition × mode)
  - figures/step63_llm_agent_eval_model_comparison.png/.pdf

The figure is the camera-ready Story 5 cross-model figure.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Volumes/Data/ontocontext")
TBL = ROOT / "tables"
FIG = ROOT / "figures"
TBL.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def load_run1(model_dir: Path) -> list[dict]:
    """Load the most recent run summary CSV for one model.

    Picks the file with the latest mtime under `<model_dir>/*_summary.csv`
    so re-runs with a new `--run-tag` (e.g. run_v3_patched) automatically
    supersede older ones. Returns rows.
    """
    candidates = sorted(
        model_dir.glob("*_summary.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        print(f"[load_run1] missing {model_dir}/*_summary.csv", file=sys.stderr)
        return []
    p = candidates[-1]
    return list(csv.DictReader(p.open()))


def aggregate():
    """Build a single combined CSV across all models."""
    out = TBL / "step63_model_comparison.csv"
    rows = []
    models_seen = []
    for d in sorted(TBL.glob("step63_runs/gpt_*")):
        model_name = d.name.replace("gpt_", "gpt-")  # gpt_4o_mini -> gpt-4o-mini
        if "gpt_4o_mini" in d.name:
            model_name = "gpt-4o-mini"
        elif "gpt_5_mini" in d.name:
            model_name = "gpt-5-mini"
        elif "gpt_5_pro" in d.name:
            model_name = "gpt-5-pro"
        elif "gpt_6_astra" in d.name:
            model_name = "gpt-6-astra"
        else:
            continue
        for r in load_run1(d):
            # F2: carry n_failed through so the cross-model table
            # surfaces cells with billing / network issues.
            n_failed = int(r.get("n_failed", 0) or 0)
            n_total = int(r.get("n", 0) or 0) + n_failed
            rows.append({
                "model": model_name,
                "condition": r["condition"],
                "mode": r["mode"],
                "n": r["n"],
                "n_failed": n_failed,
                "n_total_attempted": n_total,
                "accuracy": r["accuracy"],
                "hallucination_rate": r["hallucination_rate"],
                "tool_call_rate": r["tool_call_rate"],
                "mean_tokens": r["mean_tokens"],
                "mean_latency": r["mean_latency"],
            })
        models_seen.append(model_name)

    if not rows:
        print("[aggregate] no data found", file=sys.stderr)
        return

    with out.open("w", newline="", encoding="utf-8") as fh:
        # F2: data-status banner so reviewers can spot cells with
        # billing-failure noise. 5% threshold matches the rule in
        # step63_aggregate_runs.py.
        n_dropped = sum(int(r.get("n_failed", 0) or 0) for r in rows)
        n_total = sum(int(r.get("n_total_attempted", 0) or 0) for r in rows)
        pct = (n_dropped / max(1, n_total)) * 100
        if n_dropped > 0 and pct >= 5.0:
            print(
                f"DATA-STATUS: {n_dropped} of {n_total} LLM calls "
                f"({pct:.1f}%) were dropped because of billing / network "
                f"failures across the {len({r['model'] for r in rows})} models. "
                f"Excluded from the rate denominators.",
                file=sys.stderr,
            )
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote: {out}")
    print(f"Models included: {sorted(models_seen)}")
    return rows


def make_figure(rows: list[dict]):
    """Build a 2-panel figure (default + hard) with 4 models per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter out models with ALL conditions showing rate-limit / no real output.
    # If a model has at least 1 non-zero row, keep all its rows.
    nonzero_models = set()
    for r in rows:
        if float(r["accuracy"]) > 0 or float(r["hallucination_rate"]) > 0 or float(r["tool_call_rate"]) > 0:
            nonzero_models.add(r["model"])
    valid = [r for r in rows if r["model"] in nonzero_models]
    print(f"Models with non-zero data: {sorted(nonzero_models)}")

    # Model order
    model_order = ["gpt-4o-mini", "gpt-5-mini", "gpt-5-pro", "gpt-6-astra"]
    models = sorted({r["model"] for r in valid}, key=lambda m: (
        model_order.index(m) if m in model_order else 99, m,
    ))
    conds = ["OntoContext @ 256 tok", "Full corpus (top-5)",
             "BM25 @ 256 tok", "GraphRAG @ 256 tok"]
    modes = ["default", "hard"]
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mode in zip(axes, modes):
        n_models = len(models)
        n_conds = len(conds)
        bar_w = 0.8 / n_models
        group_w = n_models * bar_w + 0.05

        # We'll plot acc cluster then hall cluster, side by side per condition
        for ci, cond in enumerate(conds):
            for mi, model in enumerate(models):
                cell = next((r for r in valid if r["model"] == model
                             and r["condition"] == cond and r["mode"] == mode), None)
                if not cell:
                    continue
                acc = float(cell["accuracy"])
                hall = float(cell["hallucination_rate"])
                tool = float(cell["tool_call_rate"])
                # Accuracy cluster
                x_acc = ci + (mi - n_models / 2 + 0.5) * bar_w
                # Hallucination cluster
                x_hall = ci + n_conds * 0.3 + (mi - n_models / 2 + 0.5) * bar_w
                c = colors[mi % len(colors)]
                ax.bar(x_acc, acc, bar_w * 0.9, color=c, alpha=0.85, label=model if ci == 0 else "")
                ax.bar(x_hall, hall, bar_w * 0.9, color=c, alpha=0.45)

        # X axis: two clusters of conditions
        x_centers = list(range(n_conds)) + [n_conds * 1.3 + i for i in range(n_conds)]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            conds + conds, rotation=18, ha="right", fontsize=8.5,
        )
        ax.set_xlim(-0.7, n_conds * 1.3 + n_conds - 0.3)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(
            f"{mode.replace('default','Default (term id in prompt)').replace('hard','Hard (term id removed)')}",
            fontsize=11)
        ax.set_ylabel("Rate")
        # Legend
        ax.legend(fontsize=8, loc="upper left", title="Model")

        # Cluster labels above
        for cluster_i, label in enumerate(["Accuracy", "Hallucination rate"]):
            center = (cluster_i * n_conds + (n_conds - 1) / 2)
            ax.text(center, 1.08, label, ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "LLM-agent evaluation on CL — cross-model comparison\n"
        "Each cell: 30 questions, default vs hard mode; colors = model",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "step63_llm_agent_eval_model_comparison.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "step63_llm_agent_eval_model_comparison.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {FIG / 'step63_llm_agent_eval_model_comparison.png'}")
    print(f"Wrote: {FIG / 'step63_llm_agent_eval_model_comparison.pdf'}")


def main():
    rows = aggregate()
    if rows:
        make_figure(rows)


if __name__ == "__main__":
    main()
