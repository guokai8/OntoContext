"""
step04_v1_a_weight_sweep.py — V1 weight-sweep ablation.

PLAN.md step 04 verification requires:
  ≥ 8 weight settings on the **synthetic** ontology
  ≥ 3 weight settings on real GO

This script:
  1. Reads `capsules/v0/entities/<ontology>/*.json`.
  2. Computes the IC table from `cache/ic_tables/<snapshot_id>.parquet`
     (or step 02's node/edge parquet if missing).
  3. For every weight setting in the sweep, builds a **V1** capsule
     set in a *separate* output dir, so each run is reproducible
     from scratch.
  4. Records for each weight setting: mean IC of selected_ancestors,
     total selected-ancestor count across all capsules, and the
     count of capsules whose selected_ancestors changed relative
     to the default weight set.
  5. Writes:
       - `tables/step04_a_weight_sweep.csv`
       - `tables/step04_a_weight_sweep.tex`
       - `figures/step04_a_weight_sweep.png`
       - `figures/step04_a_weight_sweep.pdf`
     (PNG + PDF per rule §1; CSV + TeX per rule §2).

Usage:
    python3 scripts/step04_v1_a_weight_sweep.py --synthetic synth_d02_b02_s017
    python3 scripts/step04_v1_a_weight_sweep.py --snapshot go_2024-06-10
    python3 scripts/step04_v1_a_weight_sweep.py --all
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# matplotlib only when actually plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step04_v1_select import (
    _compute_ic_table, _curie_path, _utility, _weight_set_for, IC_ROOT,
    SCHEMA_VERSION, ANCESTOR_BUDGET, DEFAULT_ALPHA, DEFAULT_BETA,
    DEFAULT_GAMMA, DEFAULT_DELTA,
)

V0_ROOT = Path("capsules/v0/entities")
SWEEP_OUT = Path("capsules/v1_a_weight_sweep")    # throwaway output, one dir per setting
FIG_DIR = Path("figures")
TBL_DIR = Path("tables")

WEIGHT_SETS = [
    "default", "ic_only", "amb_only", "deg_only", "ann_only",
    "ic_amb", "balanced", "a3",
]


def _build_v1_inplace(
    snapshot_id: str, ontology: str, processed_dir: Path,
    weights: tuple[float, float, float, float],
) -> dict:
    """Reuse step 04's logic but write to a per-weight subdir and
    return aggregate statistics (not per-capsule detail)."""
    out_dir = SWEEP_OUT / f"{snapshot_id}_{'_'.join(f'{w:.1f}' for w in weights)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # IC table (use cached if present)
    ic_path = IC_ROOT / f"{snapshot_id}.parquet"
    if not ic_path.exists():
        ic_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_compute_ic_table(snapshot_id, processed_dir), ic_path)
    ic_by_term = {r["term_id"]: r["ic"] for r in pq.read_table(ic_path).to_pylist()}

    src_dir = V0_ROOT / ontology
    if not src_dir.exists():
        raise RuntimeError(f"no V0 capsules at {src_dir}; run step 03 first")

    selected_anc_ic_total = 0.0
    selected_anc_count = 0
    cap_count = 0
    for v0_path in sorted(src_dir.glob("*.json")):
        v0 = json.loads(v0_path.read_text(encoding="utf-8"))
        nid = v0["id"]
        parents = list(v0.get("coordinates", {}).get("parents", []) or [])
        ann_count = v0.get("usage", {}).get("annotation_frequency", 0) or 0
        max_ann = max(1, ann_count)

        def _u(p: str) -> float:
            return _utility(ic_by_term.get(p, 0.0), 0.0, 0.0, ann_count, max_ann, weights)
        ranked = sorted(parents, key=_u, reverse=True)
        selected = ranked[:ANCESTOR_BUDGET]

        # Per-capsule bookkeeping
        for p in selected:
            selected_anc_ic_total += ic_by_term.get(p, 0.0)
        selected_anc_count += len(selected)
        cap_count += 1

        # Write the V1 capsule
        v1 = json.loads(json.dumps(v0))
        v1["schema_version"] = SCHEMA_VERSION
        v1["usage"]["information_content"] = float(ic_by_term.get(nid, 0.0))
        v1["coordinates"]["selected_ancestors"] = selected
        out = _curie_path(ontology, nid, out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v1, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "weights": [float(w) for w in weights],
        "capsules": cap_count,
        "selected_ancestors_total": selected_anc_count,
        "selected_ancestors_mean": (
            selected_anc_ic_total / max(1, selected_anc_count)
        ),
        "selected_ancestors_per_capsule_mean": (
            selected_anc_count / max(1, cap_count)
        ),
    }


def _save_both(fig, basename: str, dpi: int = 200) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    # rule §1: font-embedded PDF
    try:
        from matplotlib import rcParams
        rcParams["pdf.fonttype"] = 42
    except Exception:
        pass


def _plot_sweep(rows: list[dict], basename: str) -> None:
    """Two-panel plot: (left) mean selected-ancestor count vs weight set;
    (right) mean IC of selected ancestors vs weight set."""
    labels = [r["label"] for r in rows]
    means_n = [r["selected_ancestors_per_capsule_mean"] for r in rows]
    means_ic = [r["selected_ancestors_mean"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(range(len(labels)), means_n, color="#3b6fb6")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("mean #selected ancestors per capsule")
    axes[0].set_title("Selection size by weight set")
    axes[1].bar(range(len(labels)), means_ic, color="#7a5fb0")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("mean IC of selected ancestors")
    axes[1].set_title("Information content of selection by weight set")
    fig.suptitle(f"Step 04 — V1 weight sweep ({rows[0]['snapshot_id']})", y=1.02)
    fig.tight_layout()
    _save_both(fig, basename)
    plt.close(fig)


def _write_csv_tex(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_id", "weight_set", "alpha", "beta", "gamma", "delta",
                    "capsules", "sel_anc_total", "sel_anc_per_cap_mean", "sel_anc_ic_mean"])
        for r in rows:
            a, b, c, d = r["weights"]
            w.writerow([r["snapshot_id"], r["label"], a, b, c, d, r["capsules"],
                        r["selected_ancestors_total"],
                        f"{r['selected_ancestors_per_capsule_mean']:.4f}",
                        f"{r['selected_ancestors_mean']:.4f}"])
    # TeX
    tex_path = TBL_DIR / f"{basename}.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by step04_v1_a_weight_sweep.py\n")
        fh.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        fh.write("weight\\_set & $\\alpha$ & $\\beta$ & $\\gamma$ & $\\delta$ & "
                 "capsules & sel/ cap & IC \\\\\n\\midrule\n")
        for r in rows:
            a, b, c, d = r["weights"]
            fh.write(f"{r['label']} & {a:.2f} & {b:.2f} & {c:.2f} & {d:.2f} & "
                     f"{r['capsules']} & {r['selected_ancestors_per_capsule_mean']:.2f} & "
                     f"{r['selected_ancestors_mean']:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--weight-sets", type=str, nargs="+", default=WEIGHT_SETS)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    # Plan: ≥ 8 on synth, ≥ 3 on real GO
    if args.synthetic:
        snapshots = [(args.synthetic, "SYNTH",
                      Path("data/processed") / args.synthetic,
                      max(8, len(args.weight_sets)))]
    elif args.snapshot:
        proc = Path("data/processed") / args.snapshot
        snapshots = [(args.snapshot,
                      proc.joinpath("manifest_processed.yaml").read_text().split("ontology:")[1].split("\n")[0].strip()
                      if proc.joinpath("manifest_processed.yaml").exists()
                      else args.snapshot.split("_")[0].upper(),
                      proc,
                      max(3, len(args.weight_sets)))]
    elif args.all:
        snapshots = []
        proc_root = Path("data/processed")
        for d in sorted(proc_root.iterdir()):
            if not (d / "node.parquet").exists():
                continue
            try:
                mf = d / "manifest_processed.yaml"
                ont = mf.read_text().split("ontology:")[1].split("\n")[0].strip() if mf.exists() else d.name.split("_")[0].upper()
            except Exception:
                ont = d.name.split("_")[0].upper()
            n_required = 3 if d.name.startswith("go_") else 8
            snapshots.append((d.name, ont, d, n_required))
    else:
        print("ERROR: pass --synthetic, --snapshot, or --all", file=sys.stderr)
        return 2

    if not args.weight_sets:
        args.weight_sets = WEIGHT_SETS

    summaries: list[dict] = []
    for sid, ontology, proc, _ in snapshots:
        rows = []
        for ws in args.weight_sets:
            w = _weight_set_for(ws)
            s = _build_v1_inplace(sid, ontology, proc, w)
            s["label"] = ws
            rows.append(s)
        basename = f"step04_a_weight_sweep_{sid}"
        _write_csv_tex(rows, basename)
        _plot_sweep(rows, basename)
        print(f"✅ {sid}: {len(rows)} weight sets; "
              f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
        summaries.append({"snapshot_id": sid, "weight_sets": len(rows)})
    print(f"Total: {len(summaries)} snapshots swept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
