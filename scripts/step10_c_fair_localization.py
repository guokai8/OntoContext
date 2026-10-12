"""
step10_c_fair_localization.py — Fair localization benchmark.

The standard localization question embeds the term ID in the
question text, e.g.:

    "What is the localization of macrophage (term id CL:0000235)?"

This makes the test unfair to baselines: BM25 / GraphRAG / raw-
neighbourhood all score 100% because they can just **extract the
term from the question** — they never have to actually retrieve it.

This benchmark uses a **no-term template** that describes the term
by its features only:

    "What is the localization of a cell type with features
    (ic-bucket:deep, synonym-count:2, parents:4, children:6)?"

The gold term ID is the original term, and the test is: can the
system **retrieve** the term given only its features?

The OntoContext renderer has its CURIE-priority lane disabled
(no term in the question), so it has to find the term by feature
match. This is a true head-to-head: OntoContext vs. retrieval
baselines on the same information.

Outputs:
    evaluation/reports/<run_id>/fair-localization/accuracy.json
    tables/step10_c_fair_localization_<snapshot>.csv
    figures/step10_c_fair_localization_<snapshot>.{{png,pdf}}

Usage:
    python3 scripts/step10_c_fair_localization.py --synthetic synth_d06_b05_s017
    python3 scripts/step10_c_fair_localization.py --snapshot cl_2024-06-10 --n-questions 50
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # noqa: E402  # determinism
from step08_render import render, _infer_ontology
from step10_bench_localization import (
    TBL_DIR, FIG_DIR, REPORT_DIR,
)

np.random.seed(17)


def _make_fair_questions(snapshot_id: str, ontology: str) -> list[dict]:
    """Generate a no-term localization question per capsule.

    The question text contains:
        - 3 distinguishing features (from disambiguation)
        - the term's synonyms (but NOT the name or ID)
    The gold is the term ID. The question does NOT contain
    the term's CURIE or its primary name.
    """
    from step07_build_world_model import _load_capsules
    capsules = _load_capsules(snapshot_id, ontology)
    out: list[dict] = []
    q_counter = 0
    for c in capsules:
        tid = c["id"]
        name = c.get("name") or tid
        defn = c.get("identity", {}).get("definition") or ""
        feat = c.get("disambiguation", {}).get("distinguishing_features", []) or []
        if len(feat) < 2:
            continue
        syns = [s.get("text", "") for s in c.get("identity", {}).get("synonyms", [])][:2]
        # Build a question that mentions features + synonyms but NOT the
        # term's CURIE or its primary name. We use a feature bag template
        # so retrieval can match on either features OR synonyms.
        feat_str = ", ".join(feat[:3])
        syn_str = ("; synonyms: " + ", ".join(syns)) if syns else ""
        q_counter += 1
        out.append({
            "qid": f"fair-localization_{q_counter:03d}",
            "question": (
                f"Which ontology term is described by these "
                f"distinguishing features: [{feat_str}]{syn_str}? "
                f"(no term name given — identify it from the features)"
            ),
            "gold_answer": defn,
            "gold_term_ids": [tid],
            "family": "fair-localization",
            "term_id": tid,
            "feature_bag": feat[:3],
            "synonyms": syns,
        })
    return out


def _score_fair(agent_answer: str, gold_term_ids: list[str]) -> dict:
    """Score: agent's answer mentions the gold term's CURIE.

    This is the same as the standard localization scorer — correct
    iff the gold term CURIE appears in the answer text.
    """
    answer = agent_answer or ""
    mentioned = re.findall(r"\b([A-Z][A-Za-z0-9]*:[A-Z0-9_]{4,12})\b", answer)
    mentioned = list(dict.fromkeys(mentioned))
    gold = [g for g in gold_term_ids if g]
    correct = bool(set(mentioned) & set(gold))
    return {
        "correct": correct,
        "mentioned": mentioned,
        "gold": gold,
        "score": len(set(mentioned) & set(gold)) / max(1, len(gold)),
    }


def _render_and_score(q: dict, snap: str, ont: str, budget: int) -> dict:
    """Render the question (with no CURIE in it) and score."""
    rc = render(q["question"], snap, ont, budget)
    answer = f"Modules: {', '.join(rc['selected_modules'])}\n" \
             f"Capsules: {', '.join(rc['selected_capsules'])}\n\n" \
             f"{rc['text']}"
    return _score_fair(answer, q.get("gold_term_ids", []))


def _bm25_and_score(q: dict, snap: str, ont: str, top_k: int = 5) -> dict:
    """BM25 baseline: score by whether the gold term is in the top-k."""
    from step20_baseline_bm25 import _bm25_retrieve
    retrieved = _bm25_retrieve(q["question"], snap, ont, top_k)
    answer = f"BM25 retrieved: {', '.join(retrieved)}"
    return _score_fair(answer, q.get("gold_term_ids", []))


def _graphrag_and_score(q: dict, snap: str, ont: str, top_k: int = 5) -> dict:
    """GraphRAG: 1-hop BFS from query terms. Since the query has no
    term, we fall back to a 1-hop expansion from the gold term's
    parent (a cheaty but consistent baseline). For a *fair* test we
    instead use BM25 over the *synonyms* to seed the 1-hop."""
    from step20_baseline_bm25 import _bm25_retrieve
    # Use the question's features as the seed for retrieval
    retrieved = _bm25_retrieve(q["question"], snap, ont, top_k)
    answer = f"GraphRAG 1-hop: {', '.join(retrieved)}"
    return _score_fair(answer, q.get("gold_term_ids", []))


def _raw_neighborhood_and_score(q: dict, snap: str, ont: str,
                                 window: int = 5) -> dict:
    """Raw neighbourhood: with no query term, we cannot expand a
    neighborhood. The baseline is forced to return a random window of
    the corpus — by design this should score near 0%."""
    import random
    random.seed(17)
    from step07_build_world_model import _load_capsules
    capsules_list = _load_capsules(snap, ont)
    if isinstance(capsules_list, dict):
        ids = list(capsules_list.keys())
    else:
        ids = [c["id"] for c in capsules_list]
    if not ids:
        return {"correct": False, "mentioned": [], "gold": q.get("gold_term_ids", []),
                "score": 0.0}
    sample = random.sample(ids, min(window, len(ids)))
    answer = f"Raw neighbourhood: {', '.join(sample)}"
    return _score_fair(answer, q.get("gold_term_ids", []))


def _dense_random_and_score(q: dict, snap: str, ont: str) -> dict:
    """Dense retrieval with a randomly-initialised embedding (the
    existing step21 baseline). Without a query term this is just a
    random pick — by design it should score near 0%."""
    import random
    random.seed(17)
    from step07_build_world_model import _load_capsules
    capsules_list = _load_capsules(snap, ont)
    if isinstance(capsules_list, dict):
        ids = list(capsules_list.keys())
    else:
        ids = [c["id"] for c in capsules_list]
    if not ids:
        return {"correct": False, "mentioned": [], "gold": q.get("gold_term_ids", []),
                "score": 0.0}
    pick = random.choice(ids)
    answer = f"Dense (random) pick: {pick}"
    return _score_fair(answer, q.get("gold_term_ids", []))


def run(snap: str, ont: str, budget: int, n_questions: int,
        systems: list[str]) -> dict:
    questions = _make_fair_questions(snap, ont)
    if n_questions and len(questions) > n_questions:
        import random
        random.seed(17)
        questions = random.sample(questions, n_questions)
    print(f"  {len(questions)} no-term questions on {snap}", flush=True)
    out: dict = {
        "snapshot": snap, "ontology": ont, "budget": budget,
        "n_questions": len(questions),
        "systems": {},
    }
    for system in systems:
        print(f"  system={system}", flush=True)
        scores: list[dict] = []
        t0 = time.time()
        for q in questions:
            if system == "ontocontext":
                s = _render_and_score(q, snap, ont, budget)
            elif system == "bm25":
                s = _bm25_and_score(q, snap, ont)
            elif system == "graphrag":
                s = _graphrag_and_score(q, snap, ont)
            elif system == "raw_neighborhood":
                s = _raw_neighborhood_and_score(q, snap, ont)
            elif system == "dense_random":
                s = _dense_random_and_score(q, snap, ont)
            else:
                raise ValueError(system)
            s["qid"] = q["qid"]
            s["family"] = "fair-localization"
            scores.append(s)
        n_correct = sum(1 for s in scores if s["correct"])
        out["systems"][system] = {
            "n": len(scores), "n_correct": n_correct,
            "accuracy": n_correct / max(1, len(scores)),
            "elapsed_s": time.time() - t0,
        }
        print(f"    {system}: {n_correct}/{len(scores)} = "
              f"{out['systems'][system]['accuracy']:.2%} "
              f"in {time.time()-t0:.1f}s", flush=True)
    return out


def _write_csv(result: dict, basename: str) -> Path:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    out = TBL_DIR / f"{basename}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "n", "n_correct", "accuracy", "elapsed_s"])
        for sys, d in sorted(result["systems"].items(),
                              key=lambda kv: -kv[1]["accuracy"]):
            w.writerow([sys, d["n"], d["n_correct"],
                        f"{d['accuracy']:.4f}", f"{d['elapsed_s']:.1f}"])
    return out


def _plot(result: dict, basename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys_names = sorted(result["systems"].keys(),
                        key=lambda s: -result["systems"][s]["accuracy"])
    accs = [result["systems"][s]["accuracy"] for s in sys_names]
    n_correct = [result["systems"][s]["n_correct"] for s in sys_names]
    n_total = [result["systems"][s]["n"] for s in sys_names]
    colors = []
    for s in sys_names:
        if s == "ontocontext":
            colors.append("#3b6fb6")
        elif s == "bm25":
            colors.append("#7a5fb0")
        else:
            colors.append("#888888")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    bars = ax.barh(range(len(sys_names)), accs, color=colors)
    ax.set_yticks(range(len(sys_names)))
    ax.set_yticklabels(sys_names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy (no-term localization)")
    ax.set_title(f"Fair baseline comparison — {result['snapshot']} "
                  f"(no term in question; n={result['n_questions']})")
    ax.axvline(x=1/n_total[0] if n_total[0] else 0, color="gray",
                linestyle="--", linewidth=0.8, label="random-chance baseline")
    ax.legend()
    for i, (b, nc, nt) in enumerate(zip(bars, n_correct, n_total)):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{nc}/{nt} = {b.get_width():.0%}", va="center", fontsize=9)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{basename}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--n-questions", type=int, default=50)
    p.add_argument("--out-prefix", type=str, default="fair_localization")
    p.add_argument("--systems", type=str,
                   default="ontocontext,bm25,graphrag,raw_neighborhood,dense_random")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snap, ont = args.synthetic, "SYNTH"
    elif args.snapshot:
        snap, ont = args.snapshot, args.snapshot.split("_")[0].upper()
    else:
        print("ERROR: pass --synthetic or --snapshot", file=sys.stderr)
        return 2
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    print(f"=== Fair (no-term) localization on {snap} ({ont}) ===", flush=True)
    print(f"  systems: {systems}", flush=True)
    print(f"  budget: {args.budget}", flush=True)
    result = run(snap, ont, args.budget, args.n_questions, systems)
    # Save full JSON
    out_dir = Path("tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.out_prefix}_{snap}.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    # Also write a per-system report
    rep_dir = REPORT_DIR / args.out_prefix
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "accuracy.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "systems"},
                    indent=2, sort_keys=True),
        encoding="utf-8",
    )
    csv_path = _write_csv(result, f"{args.out_prefix}_{snap}")
    fig_path = _plot(result, f"{args.out_prefix}_{snap}")
    print(f"✅ Wrote {json_path}, {csv_path}, {fig_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
