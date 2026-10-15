#!/usr/bin/env python3
"""
step63_aggregate_runs.py — Cross-run statistics for the LLM-agent
evaluation. Reads every `tables/step63_runs/run*_summary.csv` and
emits a single aggregated table with mean ± SD per (condition, mode).

Output:
  tables/step63_llm_agent_eval_cross_run.csv
  tables/step63_llm_agent_eval_cross_run_summary.txt
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path("/Volumes/Data/ontocontext")
RUNS = ROOT / "tables" / "step63_runs"
OUT_CSV = ROOT / "tables" / "step63_llm_agent_eval_cross_run.csv"
OUT_TXT = ROOT / "tables" / "step63_llm_agent_eval_cross_run_summary.txt"


def _load_runs() -> list[dict]:
    """Read every run*_summary.csv and return one row per (condition, mode, run).

    Two layouts are supported:
      A) Flat: `tables/step63_runs/run*_summary.csv` (legacy single-model
         runs from before the multi-model refactor). These are *pre-fix*
         OntoContext runs and are skipped here; they live on in the
         archives for reproducibility.
      B) Per-model: `tables/step63_runs/<model>/*_summary.csv`. We pick
         the most recent run-tag per model by file mtime so a re-run
         with a new `--run-tag` (e.g. run_v3_patched) automatically
         supersedes older ones.
    """
    runs: list[dict] = []

    # Layout B: per-model subfolders. Pick the latest run-tag per model.
    for model_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        summaries = sorted(
            model_dir.glob("*_summary.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if not summaries:
            continue
        latest = summaries[-1]
        run_id = f"{model_dir.name}/{latest.stem.replace('_summary', '')}"
        with latest.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                runs.append({
                    "run": run_id,
                    "condition": r["condition"],
                    "mode": r["mode"],
                    "n": int(r["n"]),
                    "n_failed": int(r.get("n_failed", 0) or 0),
                    "accuracy": float(r["accuracy"]),
                    "hallucination_rate": float(r["hallucination_rate"]),
                    "tool_call_rate": float(r.get("tool_call_rate", 0) or 0),
                })
    return runs


def _aggregate(runs: list[dict]) -> list[dict]:
    """Group by (condition, mode) and compute mean ± SD across runs."""
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        by_cell.setdefault((r["condition"], r["mode"]), []).append(r)

    agg = []
    for (cond, mode), recs in sorted(by_cell.items()):
        accs = [r["accuracy"] for r in recs]
        halls = [r["hallucination_rate"] for r in recs]
        tools = [r["tool_call_rate"] for r in recs]
        # F2: `n` in the per-run summary is now the count of *valid*
        # (non-failed) LLM calls. `n_failed` is the count of calls
        # skipped because of a billing / network failure. `n_per_run`
        # here is the valid count, and `n_total` is the sum of valid
        # counts across runs. The text output adds a `data-status`
        # column so reviewers can see how many cells had to be dropped.
        n_per = recs[0]["n"]
        n_failed_total = sum(int(r.get("n_failed", 0) or 0) for r in recs)
        n_runs = len(recs)
        agg.append({
            "condition": cond,
            "mode": mode,
            "n_runs": n_runs,
            "n_per_run": n_per,
            "n_failed_total": n_failed_total,
            "n_total": n_runs * n_per,
            "acc_mean": statistics.fmean(accs),
            "acc_sd": statistics.stdev(accs) if n_runs > 1 else 0.0,
            "acc_min": min(accs),
            "acc_max": max(accs),
            "hall_mean": statistics.fmean(halls),
            "hall_sd": statistics.stdev(halls) if n_runs > 1 else 0.0,
            "hall_min": min(halls),
            "hall_max": max(halls),
            "tool_mean": statistics.fmean(tools),
            "tool_sd": statistics.stdev(tools) if n_runs > 1 else 0.0,
        })
    return agg


def _write_csv(agg: list[dict]) -> Path:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        for r in agg:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    return OUT_CSV


def _write_text(agg: list[dict], runs: list[dict]) -> Path:
    n_runs = len({r["run"] for r in runs})
    lines = [
        f"Cross-run statistics for step63 LLM-agent evaluation",
        f"Runs aggregated: {n_runs}  ({', '.join(sorted({r['run'] for r in runs}))})",
        "",
        f"{'condition':<24} {'mode':<8} {'n':>5} {'failed':>7}  "
        f"{'acc mean':>9} {'acc sd':>7}  "
        f"{'hall mean':>10} {'hall sd':>8}  "
        f"{'tool mean':>10} {'tool sd':>8}",
        "-" * 110,
    ]
    for r in agg:
        lines.append(
            f"{r['condition']:<24} {r['mode']:<8} {r['n_total']:>5} "
            f"{r.get('n_failed_total', 0):>7}  "
            f"{r['acc_mean']*100:>8.2f}% {r['acc_sd']*100:>6.2f}  "
            f"{r['hall_mean']*100:>9.2f}% {r['hall_sd']*100:>7.2f}  "
            f"{r['tool_mean']*100:>9.2f}% {r['tool_sd']*100:>7.2f}"
        )
    # F2: data-status banner so reviewers can spot cells where >5% of
    # the LLM calls were dropped because of billing / network failures.
    n_dropped_total = sum(int(r.get("n_failed_total", 0) or 0) for r in agg)
    if n_dropped_total > 0:
        n_total_all = sum(int(r.get("n_total", 0) or 0) for r in agg)
        pct = (n_dropped_total / max(1, n_total_all + n_dropped_total)) * 100
        lines.append("")
        lines.append(
            f"DATA-STATUS: {n_dropped_total} LLM calls ({pct:.1f}%) were dropped "
            f"because of billing / network failures (OpenAI 429 etc.). "
            f"Cells with n_failed > 0 are excluded from the rate denominators."
        )
    text = "\n".join(lines) + "\n"
    OUT_TXT.write_text(text)
    return OUT_TXT


def main() -> int:
    if not RUNS.is_dir():
        print(f"ERROR: missing {RUNS}", file=sys.stderr)
        return 1
    runs = _load_runs()
    if not runs:
        print(f"ERROR: no run*_summary.csv files in {RUNS}", file=sys.stderr)
        return 1
    agg = _aggregate(runs)
    csv_p = _write_csv(agg)
    txt_p = _write_text(agg, runs)
    print(f"Wrote: {csv_p}")
    print(f"Wrote: {txt_p}")
    print()
    print(txt_p.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
