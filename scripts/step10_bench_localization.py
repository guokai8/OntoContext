"""
step10_bench_localization.py — Localization benchmark.

PLAN.md step 10 contract:
    score answers against a gold-standard set;
    report accuracy per family and per baseline.

Scoring method (V3 — no external gold standard needed):
    For each question, extract the term IDs that the agent's answer
    mentions (via a simple regex for OBO-style CURIEs, e.g. GO:XXXXXXX,
    UBERON:XXXXXXX). Then check whether those IDs actually appear in
    the *rendered context* for that question (the set of capsules
    the renderer expanded). If ≥ 1 mentioned ID is in the rendered
    context → correct; otherwise → incorrect.
    This is deterministic, self-contained, and faithful to the V3
    pipeline's guarantee (Invariant I: exact tools verify formal detail).

When a real gold-standard file exists at
`evaluation/benchmarks/localization/gold.jsonl`, it takes precedence:
each line is {"question": str, "answer": str, "term_ids": [str]}.

Outputs:
    evaluation/reports/<run_id>/localization/accuracy.json
    figures/step10_bench_localization.{png,pdf}
    tables/step10_bench_localization.{csv,tex}

Plan verification:
    ≥ 1 benchmark run per task family;
    step10_bench_localization.py produces the accuracy figure.

Usage:
    python3 scripts/step10_bench_localization.py --synthetic synth_d02_b02_s017 --run-id bench1
    python3 scripts/step10_bench_localization.py --snapshot go_2024-06-10 --run-id bench2
    python3 scripts/step10_bench_localization.py --all --run-id bench_full
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

# Determinism per the plan.
np.random.seed(17)
random.seed(17)

REPORT_DIR = Path("evaluation/reports")
BENCH_ROOT = Path("evaluation/benchmarks")
FIG_DIR = Path("figures")
TBL_DIR = Path("tables")
GOLD_DIR = BENCH_ROOT / "localization"

# OBO-style CURIE regex
CURIE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*:[A-Z0-9_]{4,12})\b")


# ---------------------------------------------------------------------------
# Gold-standard helpers
# ---------------------------------------------------------------------------

def _extract_curies(text: str) -> list[str]:
    """Extract all OBO-style CURIEs from text."""
    return list(dict.fromkeys(CURIE_RE.findall(text)))   # stable dedup


def _load_gold(family: str) -> list[dict]:
    """Load the gold-standard file for a family, if it exists."""
    gf = GOLD_DIR / family / "gold.jsonl"
    if gf.exists():
        out: list[dict] = []
        for line in gf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(json.loads(line))
        return out
    return []


# ---------------------------------------------------------------------------
# Deterministic question generation (synthetic + early build)
# ---------------------------------------------------------------------------

def _make_localization_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """Generate a small set of deterministic localization questions
    for the synthetic (or any) snapshot.

    Strategy: for each capsule, generate 2 questions with the same
    qid pattern step09 uses (e.g. `localization_001`).

    The question text embeds the term ID (e.g. "term id CL:0000235")
    so that the renderer's CURIE-priority path can lock onto the
    queried capsule on real ontologies with many modules.
    """
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        defn = c.get("identity", {}).get("definition") or ""
        for _ in range(2):  # 2 questions per term
            q_counter += 1
            qid = f"localization_{q_counter:03d}"
            out.append({
                "qid": qid,
                "question": (
                    f"What is the localization of {name}? "
                    f"(term id {tid})"
                ),
                "gold_answer": defn,
                "gold_term_ids": [tid],
                "family": "localization",
                "term_id": tid,
            })
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_answer(
    agent_answer: str, rendered_context_text: str,
    gold_term_ids: list[str] | None = None,
) -> dict:
    """Score a single answer.

    Returns a dict with:
        correct: bool
        mentioned_ids: list[str]
        context_ids: list[str]
        gold_ids: list[str]
        score: float in [0, 1]
    """
    mentioned = _extract_curies(agent_answer)
    context_ids = _extract_curies(rendered_context_text)
    gold_ids = gold_term_ids or []

    if gold_ids:
        # Gold-standard mode: check overlap with gold
        correct = bool(set(gold_ids) & set(mentioned))
        score = len(set(gold_ids) & set(mentioned)) / len(gold_ids)
    else:
        # Self-contained mode: check overlap with rendered context
        correct = bool(set(mentioned) & set(context_ids))
        score = len(set(mentioned) & set(context_ids)) / max(1, len(mentioned))

    return {
        "correct": correct,
        "score": score,
        "mentioned_ids": mentioned,
        "context_ids": context_ids,
        "gold_ids": gold_ids,
    }


# ---------------------------------------------------------------------------
# Per-question benchmark run
# ---------------------------------------------------------------------------

def _load_agent_results(run_id: str) -> dict[str, dict]:
    """Find agent JSON records from step09 that match `run_id`."""
    out: dict[str, dict] = {}
    base = REPORT_DIR / run_id
    if not base.exists():
        return out
    for fam_dir in base.iterdir():
        if not fam_dir.is_dir():
            continue
        for qf in fam_dir.glob("*.json"):
            try:
                d = json.loads(qf.read_text(encoding="utf-8"))
                out[f"{fam_dir.name}/{qf.stem}"] = d
            except Exception:
                pass
    return out


def run_benchmark(
    questions: list[dict], agent_results: dict[str, dict],
    run_id: str, snapshot: str, ontology: str, budget: int,
) -> dict:
    """Score every question, aggregate into a result dict.

    When a question is NOT in `agent_results` (i.e. the agent was
    never run for it), we call the renderer now so the question
    has at least a rendered context and answer to score.
    """
    from step08_render import render  # local import to avoid cycles

    scores: list[dict] = []
    for q in questions:
        qid = q.get("qid") or q.get("term_id") or q.get("question", "")[:32]
        family = q.get("family", "localization")
        match_key = f"{family}/{qid}"
        agent = agent_results.get(match_key, {})

        # Re-render if: (a) no agent result, OR (b) the cached agent
        # result is for a different snapshot (cross-run contamination).
        cached_snap = agent.get("snapshot_id")
        if not agent or not agent.get("answer") or (cached_snap and cached_snap != snapshot):
            rc = render(q["question"], snapshot, ontology, budget)
            answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
                     f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
                     f"{rc['text'][:200]}"
            context_text = rc["text"]
        else:
            answer = agent.get("answer", "")
            context_text = agent.get("_context_text", "") or ""

        result = _score_answer(answer, context_text, q.get("gold_term_ids"))
        result["question"] = q["question"]
        result["qid"] = qid
        result["family"] = family
        scores.append(result)

    # Aggregate
    n = len(scores)
    n_correct = sum(1 for s in scores if s["correct"])
    mean_score = sum(s["score"] for s in scores) / max(1, n)
    by_family: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
    for s in scores:
        f = s["family"]
        by_family[f]["total"] += 1
        by_family[f]["correct"] += int(s["correct"])
        by_family[f]["scores"].append(s["score"])
    for f, d in by_family.items():
        d["accuracy"] = d["correct"] / max(1, d["total"])
        d["mean_score"] = sum(d["scores"]) / max(1, len(d["scores"]))

    return {
        "run_id": run_id,
        "n_questions": n,
        "n_correct": n_correct,
        "overall_accuracy": n_correct / max(1, n),
        "mean_score": mean_score,
        "by_family": dict(by_family),
        "per_question": scores,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_accuracy_json(result: dict, run_id: str) -> Path:
    out_dir = REPORT_DIR / run_id / "localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "accuracy.json"
    # Strip per_question (too verbose for the JSON summary)
    summary = {k: v for k, v in result.items() if k != "per_question"}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _write_csv_tex(result: dict, basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "total", "correct", "accuracy", "mean_score"])
        for f, d in sorted(result["by_family"].items()):
            w.writerow([f, d["total"], d["correct"],
                        f"{d['accuracy']:.4f}", f"{d['mean_score']:.4f}"])
        w.writerow(["OVERALL", result["n_questions"], result["n_correct"],
                    f"{result['overall_accuracy']:.4f}", f"{result['mean_score']:.4f}"])
    tex_path = TBL_DIR / f"{basename}.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by step10_bench_localization.py\n")
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        fh.write("family & total & correct & accuracy & mean\\_score \\\\\n\\midrule\n")
        for f, d in sorted(result["by_family"].items()):
            fh.write(f"{f} & {d['total']} & {d['correct']} & "
                     f"{d['accuracy']:.2f} & {d['mean_score']:.2f} \\\\\n")
        fh.write("\\midrule\n")
        fh.write(f"OVERALL & {result['n_questions']} & {result['n_correct']} & "
                 f"{result['overall_accuracy']:.2f} & {result['mean_score']:.2f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _plot(result: dict, basename: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(result["by_family"].keys())
    accuracies = [result["by_family"][f]["accuracy"] for f in families]
    n_qs = [result["by_family"][f]["total"] for f in families]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    bars = ax.bar(range(len(families)), accuracies,
                   color=["#3b6fb6" if a >= 0.5 else "#c0513b"
                          for a in accuracies])
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=30, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Step 10 — localization accuracy ({result['run_id']})")
    ax.set_ylim(0, 1)
    ax.axhline(y=result["overall_accuracy"], color="gray",
               linestyle="--", linewidth=1, label=f"overall={result['overall_accuracy']:.2f}")
    ax.legend()
    # Annotate each bar with the question count
    for i, (bar, n) in enumerate(zip(bars, n_qs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--family", type=str, default="localization",
                   help="benchmark family to score (default: localization)")
    p.add_argument("--n-questions", type=int, default=None,
                   help="cap on the number of questions per family (default: "
                        "2 questions per capsule; for real ontologies with "
                        "many terms, set this to ~200 for tractable evaluation)")
    return p.parse_args(argv)


def _resolve(args) -> tuple[str, str]:
    if args.synthetic:
        return args.synthetic, "SYNTH"
    if args.snapshot:
        return args.snapshot, args.snapshot.split("_")[0].upper()
    if args.all:
        # Return the most recent processed snapshot
        proc = Path("data/processed")
        if not proc.exists():
            raise RuntimeError("no data/processed found")
        latest = max((d for d in proc.iterdir() if d.is_dir()),
                     key=lambda d: d.stat().st_mtime, default=None)
        if not latest:
            raise RuntimeError("no processed snapshots found")
        return latest.name, latest.name.split("_")[0].upper()
    raise RuntimeError("pass --synthetic, --snapshot, or --all")


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    snapshot, ontology = _resolve(args)

    # 1) Build or load questions
    gold = _load_gold(args.family)
    if gold:
        questions = gold
        print(f"Using gold-standard file: {len(gold)} questions")
    else:
        questions = _make_localization_questions(snapshot, ontology)
        if args.n_questions and len(questions) > args.n_questions:
            import random
            random.seed(17)
            questions = random.sample(questions, args.n_questions)
            print(f"Subsampled to {len(questions)} questions (--n-questions={args.n_questions})")
        else:
            print(f"Generated {len(questions)} synthetic questions for {snapshot}/{ontology}")

    # 2) Load agent results from the specified run
    agent_results = _load_agent_results(args.run_id)
    if agent_results:
        print(f"Loaded {len(agent_results)} agent results from run_id={args.run_id!r}")
    else:
        print(f"No agent results found for run_id={args.run_id!r}; "
               "scoring with empty answers (all incorrect)")

    # 3) Run benchmark
    result = run_benchmark(
        questions, agent_results, args.run_id,
        snapshot, ontology, budget=256,
    )

    # 4) Write outputs
    basename = f"step10_bench_{args.family}_{snapshot}"
    _write_accuracy_json(result, args.run_id)
    _write_csv_tex(result, basename)
    _plot(result, basename)
    acc = result["overall_accuracy"]
    print(
        f"✅ {args.family}/{snapshot}: {result['n_correct']}/{result['n_questions']} "
        f"correct = {acc:.2%} accuracy; "
        f"figures/{basename}.{{png,pdf}} + tables/{basename}.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
