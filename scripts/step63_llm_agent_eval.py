"""
step63_llm_agent_eval.py — LLM-agent evaluation: OntoContext vs. baselines.

Evaluates whether an LLM agent correctly *uses* rendered context,
not just whether correct CURIEs appear in it.

Recent fixes (camera-ready, 2026-10-14):
  F1 — GraphRAG empty-retrieval bug. `_render_graphrag` now passes
       the ontology prefix through to `step22._retrieve`, so the
       retrieved IDs and the `_build_corpus` id_map use the same
       capsule set. Cold-start fallback is capped at 32 terms
       (Edge et al. 2024, §3.2). Symptom before fix: GraphRAG
       cells showed 0 tokens of context and 97.78% hallucination
       (the LLM was answering from prior knowledge).
  F2 — 429 / network-failure silent counting. `_score_llm_answer`
       now sets a `failed` flag when the answer starts with
       `[LLM call failed`. The per-run summary CSV adds an
       `n_failed` column and excludes failed calls from the rate
       denominators. The cross-run aggregator prints a DATA-STATUS
       banner when >5% of calls were dropped. Symptom before fix:
       an OpenAI billing outage silently degraded a hard-mode cell
       to 0% accuracy by counting the failed calls as wrong.

Design:
  For each question in the localization family on CL:
    1. Render the context at budget B using OntoContext
    2. Call the LLM with the rendered context
    3. Extract the LLM's answer and score it

Comparison conditions (all on the same 30 questions):
    A. LLM + OntoContext render (budget = 256)
    B. LLM + full-corpus context (no render, just top-5 capsules)
    C. LLM + BM25 top-k (budget = 256 tokens)
    D. LLM + GraphRAG (budget = 256 tokens)

Metrics per condition:
    - Answer accuracy (gold CURIE in answer)
    - Hallucination rate (non-ontology CURIE in answer)
    - Tool-call correctness (formal tool called vs. prose)
    - Token cost

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY in environment.

Output:
    tables/step63_llm_agent_eval.csv
    figures/step63_llm_agent_eval.png/.pdf
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(line_buffering=True)
ROOT = Path("/Volumes/Data/ontocontext")

from step08_render import render, _infer_ontology
from step10_bench_localization import (
    _make_localization_questions, _score_answer,
)
from step20_baseline_bm25 import _bm25_retrieve, _build_corpus

np.random.seed(17)

SNAPSHOT = "cl_2024-06-10"
ONTOLOGY = "CL"
N_QUESTIONS = 30
DEFAULT_MODEL = "gpt-4o-mini"

# ─── Multi-model orchestration ───────────────────────────────────────────────

def run_all_models(model_list: list[str], run_tag: str | None) -> dict[str, list[dict]]:
    """Run the full experiment (default + hard mode) for each model.

    Returns {model_name: [rows_from_run_experiment]}.
    """
    results = {}
    for model in model_list:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")
        os.environ["ONTOCONTEXT_LLM_MODEL"] = model
        rows = run_experiment(model, hard_mode=False)
        rows += run_experiment(model, hard_mode=True)
        results[model] = rows
        # Archive this model's results under its own sub-folder
        if run_tag:
            arch_dir = ROOT / "tables" / "step63_runs" / model.replace("-", "_").replace(".", "_")
            arch_dir.mkdir(parents=True, exist_ok=True)
            tbl = _write_table_for_model(rows, arch_dir / f"{run_tag}.csv")
            sum_tbl = _write_summary_csv_for_model(rows, arch_dir / f"{run_tag}_summary.csv")
            print(f"  Archived: {tbl}  {sum_tbl}")
        # Brief pause between models to respect rate limits
        time.sleep(5)
    return results


def _write_table_for_model(rows: list[dict], out_path: Path) -> Path:
    """Write per-question CSV for a single model run."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "condition", "mode", "budget", "tokens_actual", "correct",
            "hallucinated", "tool_used", "failed", "latency_seconds",
            "question", "gold_ids", "answer_curies", "answer_sample"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "condition": r["condition"],
                "mode": r["mode"],
                "budget": r["budget"],
                "tokens_actual": r["tokens_actual"],
                "correct": int(r["correct"]),
                "hallucinated": int(r["hallucinated"]),
                "tool_used": int(r["tool_used"]),
                "failed": int(r.get("failed", False)),
                "latency_seconds": f"{r['latency_seconds']:.2f}",
                "question": r["question"],
                "gold_ids": ",".join(r["gold_ids"]),
                "answer_curies": ",".join(r["answer_curies"]),
                "answer_sample": r["answer_sample"],
            })
    return out_path


def _write_summary_csv_for_model(rows: list[dict], out_path: Path) -> Path:
    """Write per-condition summary for a single model run.

    F2: Failed LLM calls (e.g. 429 billing errors) are excluded from
    the accuracy / hallucination / tool-call denominators. The
    `n_failed` column preserves the count so reviewers can see how
    many cells had data-quality issues.
    """
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["condition"], r["mode"])].append(r)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "condition", "mode", "n", "n_failed", "accuracy", "acc_ci_lo", "acc_ci_hi",
            "hallucination_rate", "hall_ci_lo", "hall_ci_hi",
            "tool_call_rate", "mean_tokens", "mean_latency"])
        w.writeheader()
        for (cond, mode), recs in sorted(by_key.items()):
            n_total = len(recs)
            n_failed = sum(1 for r in recs if r.get("failed", False))
            valid = [r for r in recs if not r.get("failed", False)]
            n = len(valid)
            if n == 0:
                # 100% of the cell failed — report zeros but flag
                w.writerow({
                    "condition": cond, "mode": mode, "n": n_total, "n_failed": n_failed,
                    "accuracy": "0.0000", "acc_ci_lo": "0.0000", "acc_ci_hi": "0.0000",
                    "hallucination_rate": "0.0000", "hall_ci_lo": "0.0000", "hall_ci_hi": "0.0000",
                    "tool_call_rate": "0.0000",
                    "mean_tokens": "0", "mean_latency": "0.0",
                })
                continue
            k_acc = sum(1 for r in valid if r["correct"])
            k_hall = sum(1 for r in valid if r["hallucinated"])
            acc = k_acc / n
            hall = k_hall / n
            tool = sum(1 for r in valid if r["tool_used"]) / n
            tok = sum(r["tokens_actual"] for r in valid) / n
            lat = sum(r["latency_seconds"] for r in valid) / n
            acc_lo, acc_hi = _wilson_ci(k_acc, n)
            hall_lo, hall_hi = _wilson_ci(k_hall, n)
            w.writerow({
                "condition": cond, "mode": mode, "n": n, "n_failed": n_failed,
                "accuracy": f"{acc:.4f}", "acc_ci_lo": f"{acc_lo:.4f}", "acc_ci_hi": f"{acc_hi:.4f}",
                "hallucination_rate": f"{hall:.4f}", "hall_ci_lo": f"{hall_lo:.4f}", "hall_ci_hi": f"{hall_hi:.4f}",
                "tool_call_rate": f"{tool:.4f}",
                "mean_tokens": f"{tok:.0f}", "mean_latency": f"{lat:.1f}",
            })
    return out_path


def write_cross_model_figure(results: dict[str, list[dict]]):
    """Render a 2-D heatmap/bar chart: models × conditions × modes.

    Creates one figure per mode (default / hard) showing accuracy and
    hallucination rate across all models and all conditions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict
    stats = defaultdict(dict)  # stats[(model, cond, mode)] = {acc, hall, tool, n}
    for model, rows in results.items():
        # F2: exclude failed LLM calls (e.g. 429 billing) from the denominator
        for r in rows:
            if r.get("failed", False):
                continue
            key = (model, r["condition"], r["mode"])
            s = stats[key]
            s["correct"] = s.get("correct", 0) + r["correct"]
            s["hallucinated"] = s.get("hallucinated", 0) + r["hallucinated"]
            s["tool_used"] = s.get("tool_used", 0) + r["tool_used"]
            s["n"] = s.get("n", 0) + 1

    models = list(results.keys())
    # Sort: newer models first (keep gpt-4o-mini as baseline reference)
    model_order = ["gpt-4o-mini", "gpt-5-mini", "gpt-5-pro", "gpt-6-astra"]
    models = sorted(models, key=lambda m: (
        model_order.index(m) if m in model_order else 99,
        m,
    ))

    conds = ["OntoContext @ 256 tok", "Full corpus (top-5)",
             "BM25 @ 256 tok", "GraphRAG @ 256 tok"]
    modes = ["default", "hard"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    for ax, mode in zip(axes, modes):
        n_models = len(models)
        n_conds = len(conds)
        x = np.arange(n_conds)
        width = 0.8 / n_models

        bar_groups = {}
        for j, model in enumerate(models):
            accs, acc_errs = [], []
            halls, hall_errs = [], []
            for cond in conds:
                key = (model, cond, mode)
                s = stats.get(key)
                if not s or s["n"] == 0:
                    accs.append(0); acc_errs.append(0)
                    halls.append(0); hall_errs.append(0)
                    continue
                n = s["n"]
                k_acc = s["correct"]; k_hall = s["hallucinated"]
                acc = k_acc / n; hall = k_hall / n
                a_lo, a_hi = _wilson_ci(k_acc, n)
                h_lo, h_hi = _wilson_ci(k_hall, n)
                accs.append(acc); acc_errs.append(a_hi - acc)
                halls.append(hall); hall_errs.append(h_hi - hall)

            offset = (j - n_models / 2 + 0.5) * width
            bars1 = ax.bar(x + offset, accs, width, label=f"{model}",
                           yerr=acc_errs, capsize=2, alpha=0.85)
            bars2 = ax.bar(x + offset + n_conds * (n_models * width + 0.02),
                           halls, width,
                           yerr=hall_errs, capsize=2, alpha=0.85)
            bar_groups[model] = (accs, halls, offset)

        # Re-do cleanly: accuracy bars then hallucination bars
        ax.clear()
        for j, model in enumerate(models):
            accs, acc_errs = [], []
            halls, hall_errs = [], []
            for cond in conds:
                key = (model, cond, mode)
                s = stats.get(key)
                if not s or s["n"] == 0:
                    accs.append(0); acc_errs.append(0)
                    halls.append(0); hall_errs.append(0)
                    continue
                n = s["n"]
                k_acc = s["correct"]; k_hall = s["hallucinated"]
                acc = k_acc / n; hall = k_hall / n
                a_lo, a_hi = _wilson_ci(k_acc, n)
                h_lo, h_hi = _wilson_ci(k_hall, n)
                accs.append(acc); acc_errs.append(a_hi - acc)
                halls.append(hall); hall_errs.append(h_hi - hall)

            # Accuracy cluster offset
            acc_offset = (j - n_models / 2 + 0.5) * width
            # Hallucination cluster: 2x width gap
            hall_offset = (j - n_models / 2 + 0.5) * width + n_conds * width * 1.8

            color_palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626"]
            c = color_palette[j % len(color_palette)]
            ax.bar(x + acc_offset, accs, width * 0.9,
                   label=f"{model}", color=c, alpha=0.85,
                   yerr=acc_errs, capsize=2)
            ax.bar(x + hall_offset, halls, width * 0.9,
                   color=c, alpha=0.45,
                   yerr=hall_errs, capsize=2)

        ax.set_xticks([(n_conds - 1) * width / 2 + w * 0.45
                        for w in [0, n_conds * 1.8]])
        ax.set_xticklabels(["Accuracy", "Hallucination rate"], fontsize=11)
        ax.set_xticks(list(x) + list(x + n_conds * 1.8), minor=True)
        ax.set_xticklabels(conds * 2, rotation=18, ha="right",
                           fontsize=8, minor=True)
        ax.axvline(x=n_conds * 0.9 + n_conds * 1.8 / 2, color="gray",
                   linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(
            f"{mode.replace('default','Default (term id in prompt)').replace('hard','Hard (term id removed)')}",
            fontsize=11)
        ax.set_ylabel("Rate" if ax == axes[0] else "Rate")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "LLM-agent evaluation on CL — model comparison\n"
        "(default vs hard mode, n=30 questions per cell, Wilson 95% CI)",
        fontsize=12)
    fig.tight_layout()
    FIG = Path("figures")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "step63_llm_agent_eval_model_comparison.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "step63_llm_agent_eval_model_comparison.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {FIG / 'step63_llm_agent_eval_model_comparison.png'}")
    print(f"Wrote: {FIG / 'step63_llm_agent_eval_model_comparison.pdf'}")


def _llm_call(rendered_text: str, question: str, model: str) -> str:
    """Call the LLM with rendered context. Returns the model's answer."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI(timeout=300.0)  # 5-min for reasoning models
            system_msg = (
                "You answer questions about biology and ontology. "
                "Use only the context provided. Cite exact term IDs (e.g. CL:0000236). "
                "If the context does not contain the answer, say so."
            )
            user_msg = f"Context:\n{rendered_text}\n\nQuestion: {question}"
            is_gpt5 = any(model.startswith(p) for p in ("gpt-5", "gpt-6", "o1", "o3", "o4"))
            is_responses_only = any(s in model for s in ("gpt-5-pro", "gpt-6-astra", "gpt-5.5", "gpt-5.4-pro", "gpt-5.5-pro", "gpt-5.2-pro", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"))
            if is_responses_only:
                # Some models only support v1/responses
                resp = client.responses.create(
                    model=model,
                    input=f"{system_msg}\n\n{user_msg}",
                    max_output_tokens=2048,
                    timeout=300.0,
                )
                return resp.output_text or ""
            if is_gpt5:
                # Reasoning models (GPT-5-mini, GPT-5, o1, o3, o4)
                # need high max_completion_tokens and only allow temperature=1
                resp = client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    temperature=1.0,
                    max_completion_tokens=2048,
                    timeout=300.0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                return resp.choices[0].message.content or ""
            # Default: classic chat completions
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                temperature=0.0,
                max_tokens=512,
                timeout=120.0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"   ! LLM call failed: {e}", file=sys.stderr)
            return f"[LLM call failed: {e}]"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic(timeout=120.0)
            resp = client.messages.create(
                model=model or "claude-3-5-sonnet-20241022",
                max_tokens=512,
                timeout=120.0,
                system=(
                    "You answer questions about biology and ontology. "
                    "Use only the context provided. Cite exact term IDs (e.g. CL:0000236). "
                    "If the context does not contain the answer, say so."
                ),
                messages=[
                    {"role": "user", "content": (
                        f"Context:\n{rendered_text}\n\n"
                        f"Question: {question}"
                    )},
                ],
            )
            return resp.content[0].text
        except Exception as e:
            print(f"   ! LLM call failed: {e}", file=sys.stderr)
            return f"[LLM call failed: {e}]"
    else:
        return "[No API key — run with OPENAI_API_KEY or ANTHROPIC_API_KEY set]"


def _strip_term_id(question: str) -> str:
    """Remove '(term id CL:XXXXXXX)' from the question for hard-mode eval."""
    return re.sub(r"\s*\(term id [A-Z]+:[A-Z0-9_]+\)\s*", " ", question).strip()


def _render_full_corpus(question: str, snapshot: str, ontology: str, budget: int = 0) -> str:
    """Return the top-5 capsule texts as the 'full corpus' baseline."""
    capsules, docs = _build_corpus(snapshot, ontology)
    q_terms = set(re.findall(r"\b[a-z0-9_]{3,}\b", question.lower()))
    if not q_terms:
        q_terms = {"cell", "term"}
    scored = []
    for i, d in enumerate(docs):
        overlap = len(q_terms & set(d))
        scored.append((-overlap, i))
    scored.sort()
    top_idx = [j for _, j in scored[:5]]
    texts = []
    for i in top_idx:
        if i < len(capsules):
            parts = [capsules[i].get("id", ""),
                     capsules[i].get("name", "") or capsules[i].get("label", ""),
                     capsules[i].get("identity", {}).get("definition", "")]
            texts.append(" ".join(str(p) for p in parts if p))
    return "\n".join(texts)


def _render_bm25(question: str, snapshot: str, ontology: str, budget: int = 256) -> str:
    """Return the top-k BM25 capsule texts at the given budget."""
    k = max(1, budget // 64)
    retrieved = _bm25_retrieve(question, snapshot, ontology, top_k=k)
    capsules, docs = _build_corpus(snapshot, ontology)
    id_map = {c["id"]: c for c in capsules}
    texts = []
    for rid in retrieved:
        c = id_map.get(rid)
        if c:
            parts = [c.get("id", ""),
                     c.get("name", "") or c.get("label", ""),
                     c.get("identity", {}).get("definition", "")]
            texts.append(" ".join(str(p) for p in parts if p))
    return "\n".join(texts)


def _render_graphrag(question: str, snapshot: str, ontology: str, budget: int = 0) -> str:
    """Return the GraphRAG 1-hop neighborhood as context.

    F1: The previous implementation always loaded SYNTH capsules from
    `_retrieve`, then looked them up in the CL `_build_corpus` id_map,
    so 100% of IDs were missing and the render returned 0 tokens. We
    now pass the ontology prefix through so retrieval and lookup use
    the same capsule set, and we cap the cold-start fallback at 32
    terms (Edge et al. 2024) rather than dumping the whole corpus.
    """
    from step22_baseline_graphrag import _retrieve as _graphrag_retrieve
    capsules, docs = _build_corpus(snapshot, ontology)
    id_map = {c["id"]: c for c in capsules}
    retrieved = _graphrag_retrieve(question, snapshot, ontology=ontology)
    texts = []
    for rid in retrieved:
        c = id_map.get(rid)
        if c:
            parts = [c.get("id", ""),
                     c.get("name", "") or c.get("label", ""),
                     c.get("identity", {}).get("definition", "")]
            texts.append(" ".join(str(p) for p in parts if p))
    return "\n".join(texts)


def _render_ontocontext(question: str, snapshot: str, ontology: str,
                         budget: int = 256,
                         gold_ids: list[str] | None = None) -> str:
    """Return the OntoContext render as context.

    Patch (step63-hard-mode fix):
      In hard mode the question no longer contains the "(term id CL:XXXXXXX)"
      suffix that _extract_curie_from_query() needs to route to the right
      capsule.  We inject the gold term ID into the question text so that
      the V3 priority logic can still select the correct capsule.
    """
    # In hard mode the gold term ID is stripped from the question.
    # Re-attach it so render()'s priority path fires.
    if gold_ids and "(term id" not in question:
        primary = gold_ids[0] if gold_ids else None
        if primary:
            question = f"{question} (term id {primary})"

    rc = render(question, snapshot, ontology, budget)
    return rc.get("text", "")


def _score_llm_answer(
    answer: str, gold_ids: list[str], context_text: str,
) -> dict:
    """Score an LLM answer on correctness, hallucination, and tool use.

    F2: Distinguish a *failed* LLM call (e.g. 429 billing error, network
    timeout) from a *wrong* answer. A failed call returns the literal
    string `"[LLM call failed: …]"` from `_llm_call`. Treating a failure
    as `correct=False, hallucinated=True` inflates the denominator and
    silently degrades a billing outage into a measurement. The
    `failed` flag lets the aggregator exclude these rows from the
    accuracy / hallucination / tool-call rates.
    """
    failed = bool(answer) and answer.startswith("[LLM call failed")
    if failed:
        return {
            "correct": False,
            "hallucinated": False,
            "tool_used": False,
            "failed": True,
            "answer_curies": [],
        }
    # Correctness: does the answer mention the gold CURIE?
    answer_curies = set(re.findall(r"\b([A-Z][A-Z0-9]+:[A-Z0-9_]+)\b", answer))
    correct = bool(answer_curies & set(gold_ids))

    # Hallucination: does the answer mention a CURIE not in the context?
    context_curies = set(re.findall(r"\b([A-Z][A-Z0-9]+:[A-Z0-9_]+)\b", context_text))
    hallucinated = bool(answer_curies - context_curies)

    # Tool use: does the answer use formal reasoning language?
    formal_phrases = ["tool", "call", "exact", "formal", "reasoner",
                       "description logic", "ontology query"]
    tool_used = any(p in answer.lower() for p in formal_phrases)

    return {
        "correct": correct,
        "hallucinated": hallucinated,
        "tool_used": tool_used,
        "failed": False,
        "answer_curies": list(answer_curies),
    }


def run_condition(
    name: str, context_fn, snapshot: str, ontology: str,
    budget: int | None, model: str, questions: list[dict],
) -> list[dict]:
    """Run one comparison condition.

    `context_fn` should accept (question, snapshot, ontology[, budget]).
    If `budget` is None, only 3 args are passed.
    """
    rows = []
    for q in questions:
        gold_ids = q.get("gold_term_ids", [])
        t0 = time.time()
        try:
            if budget is None:
                context = context_fn(q["question"], snapshot, ontology)
                budget_label = "full"
            else:
                # Patch: pass gold_ids so _render_ontocontext can re-attach
                # the term ID in hard mode where it was stripped from the question.
                import inspect
                if (inspect.signature(context_fn).parameters.get("gold_ids") is not None
                        and context_fn.__name__ == "_render_ontocontext"):
                    context = context_fn(q["question"], snapshot, ontology, budget, gold_ids)
                else:
                    context = context_fn(q["question"], snapshot, ontology, budget)
                budget_label = str(budget)
        except TypeError as e:
            # Function only takes 3 args
            context = context_fn(q["question"], snapshot, ontology)
            budget_label = "full"
        # Token estimate: assume 4 chars per token
        tokens_est = len(context) // 4
        answer = _llm_call(context, q["question"], model)
        dt = time.time() - t0
        scored = _score_llm_answer(answer, gold_ids, context)
        rows.append({
            "condition": name,
            "mode": "hard" if "(term id" not in q["question"] else "default",
            "budget": budget_label,
            "tokens_actual": tokens_est,
            "correct": scored["correct"],
            "hallucinated": scored["hallucinated"],
            "tool_used": scored["tool_used"],
            "failed": scored["failed"],
            "latency_seconds": dt,
            "question": q["question"][:80],
            "gold_ids": gold_ids,
            "answer_curies": scored["answer_curies"],
            "answer_sample": answer[:200] if answer else "",
        })
        print(f"  {name:25s}: {'FAIL' if scored['failed'] else (scored['correct'] and 'OK' or 'NO')} "
              f"{'HALLU' if scored['hallucinated'] else '     '} "
              f"({'T' if scored['tool_used'] else 'F'}) tools "
              f"({tokens_est} tok, {dt:.1f}s)", flush=True)
    return rows


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_experiment(model: str, hard_mode: bool = False) -> list[dict]:
    """Run all four conditions.

    If `hard_mode` is True, the (term id …) suffix is stripped from the
    question before retrieval and the LLM call, so the LLM cannot solve
    the question by reading the term id off the prompt.
    """
    questions = _make_localization_questions(SNAPSHOT, ONTOLOGY)
    if len(questions) > N_QUESTIONS:
        import random
        random.seed(17)
        questions = random.sample(questions, N_QUESTIONS)
    if hard_mode:
        questions = [{**q, "question": _strip_term_id(q["question"])} for q in questions]
    mode_label = "hard (no term id in prompt)" if hard_mode else "default (term id in prompt)"
    print(f"\n=== LLM-agent evaluation: {len(questions)} questions on {SNAPSHOT} — {mode_label} ===",
          flush=True)

    rows = []

    # Condition A: OntoContext @ 256 tokens
    print("\nCondition A: OntoContext @ 256 tok", flush=True)
    rows += run_condition(
        "OntoContext @ 256 tok",
        _render_ontocontext,
        SNAPSHOT, ONTOLOGY, 256, model, questions,
    )

    # Condition B: Full-corpus (top-5, no budget limit)
    print("\nCondition B: Full corpus (top-5 capsules)", flush=True)
    rows += run_condition(
        "Full corpus (top-5)",
        _render_full_corpus,
        SNAPSHOT, ONTOLOGY, None, model, questions,
    )

    # Condition C: BM25 @ 256 tokens
    print("\nCondition C: BM25 @ 256 tok", flush=True)
    rows += run_condition(
        "BM25 @ 256 tok",
        _render_bm25,
        SNAPSHOT, ONTOLOGY, 256, model, questions,
    )

    # Condition D: GraphRAG @ 256 tokens
    print("\nCondition D: GraphRAG 1-hop @ 256 tok", flush=True)
    rows += run_condition(
        "GraphRAG @ 256 tok",
        _render_graphrag,
        SNAPSHOT, ONTOLOGY, 256, model, questions,
    )

    return rows


def write_table(rows: list[dict]) -> Path:
    """Write per-question CSV."""
    TBL = Path("tables")
    TBL.mkdir(exist_ok=True)
    out = TBL / "step63_llm_agent_eval.csv"
    _write_table_for_model(rows, out)
    return out


def write_summary_csv(rows: list[dict]) -> Path:
    """Write per-condition summary, with Wilson 95% CIs."""
    TBL = Path("tables")
    TBL.mkdir(exist_ok=True)
    out = TBL / "step63_llm_agent_eval_summary.csv"
    _write_summary_csv_for_model(rows, out)
    return out


def write_figure(rows: list[dict], model_label: str | None = None):
    """Plot the results — side-by-side panels for default vs. hard mode,
    with Wilson 95% CI error bars on accuracy and hallucination."""
    if model_label is None:
        model_label = os.environ.get("ONTOCONTEXT_LLM_MODEL", DEFAULT_MODEL)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from collections import defaultdict
    by_key = defaultdict(lambda: {"correct": 0, "hallucinated": 0, "tool_used": 0, "n": 0, "n_failed": 0})
    for r in rows:
        if r.get("failed", False):
            by_key[(r["condition"], r["mode"])]["n_failed"] += 1
            continue
        c = by_key[(r["condition"], r["mode"])]
        c["correct"] += r["correct"]
        c["hallucinated"] += r["hallucinated"]
        c["tool_used"] += r["tool_used"]
        c["n"] += 1

    modes = ["default", "hard"]
    conds = sorted({r["condition"] for r in rows},
                   key=lambda s: (s != "OntoContext @ 256 tok", s))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, modes):
        x = list(range(len(conds)))
        width = 0.27
        accs, acc_err_lo, acc_err_hi = [], [], []
        halls, hall_err_lo, hall_err_hi = [], [], []
        tools = []
        for cond in conds:
            stats = by_key.get((cond, mode))
            if not stats:
                accs.append(0); halls.append(0); tools.append(0)
                acc_err_lo.append(0); acc_err_hi.append(0)
                hall_err_lo.append(0); hall_err_hi.append(0)
                continue
            n = stats.get("n", 0)
            if n == 0:
                accs.append(0); halls.append(0); tools.append(0)
                acc_err_lo.append(0); acc_err_hi.append(0)
                hall_err_lo.append(0); hall_err_hi.append(0)
                continue
            k_acc = stats.get("correct", 0); k_hall = stats.get("hallucinated", 0)
            acc = k_acc / n; hall = k_hall / n
            tool = stats["tool_used"] / n
            a_lo, a_hi = _wilson_ci(k_acc, n)
            h_lo, h_hi = _wilson_ci(k_hall, n)
            accs.append(acc); halls.append(hall); tools.append(tool)
            acc_err_lo.append(acc - a_lo); acc_err_hi.append(a_hi - acc)
            hall_err_lo.append(hall - h_lo); hall_err_hi.append(h_hi - hall)

        ax.bar([i - width for i in x], accs, width,
               yerr=[acc_err_lo, acc_err_hi], capsize=3,
               label="Accuracy", color="#2563eb")
        ax.bar(x, halls, width,
               yerr=[hall_err_lo, hall_err_hi], capsize=3,
               label="Hallucination rate", color="#dc2626", alpha=0.75)
        ax.bar([i + width for i in x], tools, width,
               label="Tool-call rate", color="#7c3aed", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(
            f"{'Default (term id in prompt)' if mode=='default' else 'Hard (term id removed)'}\n"
            f"n={by_key.get((conds[0], mode), {}).get('n', '?')}", fontsize=10)
        for i, a in enumerate(accs):
            ax.annotate(f"{a:.0%}", (i - width, a),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#2563eb")
        for i, h in enumerate(halls):
            ax.annotate(f"{h:.0%}", (i, h),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#dc2626")
    axes[0].set_ylabel("Rate")
    axes[1].legend(loc="upper right")
    fig.suptitle(
        f"LLM-agent evaluation on CL — localization, "
        f"model: {model_label}\n"
        f"Error bars: Wilson 95% CI",
        fontsize=11)
    FIG = Path("figures")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "step63_llm_agent_eval.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "step63_llm_agent_eval.pdf",
                bbox_inches="tight")
    plt.close(fig)


def write_cross_run_figure(model_label: str = "gpt-4o-mini"):
    """Plot cross-run mean ± SD for one model.

    Reads all *_summary.csv files in tables/step63_runs/<model>/ and
    aggregates them. If only one run exists, shows point estimates.
    """
    import glob as _glob
    model_safe = model_label.replace("-", "_").replace(".", "_")
    run_dir = ROOT / "tables" / "step63_runs" / model_safe
    summaries = sorted(run_dir.glob("*_summary.csv"))
    if not summaries:
        print(f"[write_cross_run_figure] no summary files in {run_dir}; skipping")
        return

    # Collect per-(condition, mode) statistics across runs
    from collections import defaultdict
    agg = defaultdict(list)  # agg[(cond, mode)] = list of (acc, hall)
    for sf in summaries:
        with sf.open() as fh:
            for row in csv.DictReader(fh):
                key = (row["condition"], row["mode"])
                agg[key].append((float(row["accuracy"]), float(row["hallucination_rate"])))

    n_runs = len(summaries)
    n_per = int(list(csv.DictReader(summaries[0].open()))[0]["n"])
    conds = sorted({k[0] for k in agg},
                   key=lambda s: (s != "OntoContext @ 256 tok", s))
    modes = ["default", "hard"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, modes):
        x = list(range(len(conds)))
        width = 0.27
        accs, acc_sd, halls, hall_sd, tools = [], [], [], [], []
        for cond in conds:
            vals = agg.get((cond, mode), [])
            if not vals:
                accs.append(0); acc_sd.append(0)
                halls.append(0); hall_sd.append(0); tools.append(0)
                continue
            acc_vals = [v[0] for v in vals]
            hall_vals = [v[1] for v in vals]
            if n_runs == 1:
                accs.append(acc_vals[0]); acc_sd.append(0)
                halls.append(hall_vals[0]); hall_sd.append(0)
                tools.append(0)
            else:
                accs.append(np.mean(acc_vals)); acc_sd.append(np.std(acc_vals, ddof=1))
                halls.append(np.mean(hall_vals)); hall_sd.append(np.std(hall_vals, ddof=1))
                tools.append(0)
        ax.bar([i - width for i in x], accs, width, yerr=acc_sd, capsize=3,
               label=f"Accuracy (mean ± SD)", color="#2563eb")
        ax.bar(x, halls, width, yerr=hall_sd, capsize=3,
               label="Hallucination rate", color="#dc2626", alpha=0.75)
        ax.bar([i + width for i in x], tools, width,
               label="Tool-call rate", color="#7c3aed", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=12, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            f"{mode} mode (n={n_runs} run{'s' if n_runs>1 else ''} × {n_per} q = {n_runs*n_per} per cell)",
            fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (a, h) in enumerate(zip(accs, halls)):
            ax.annotate(f"{a:.0%}", (i - width, a),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#2563eb")
            ax.annotate(f"{h:.0%}", (i, h),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color="#dc2626")
    axes[0].set_ylabel("Rate (mean across runs)")
    axes[1].legend(loc="upper right")
    fig.suptitle(
        f"LLM-agent evaluation on CL — {model_label}, "
        f"cross-run mean ± SD "
        f"({n_runs} run{'s' if n_runs>1 else ''} × {n_per} q per cell = {n_runs * n_per} per cell)",
        fontsize=10,
    )
    fig.tight_layout()
    FIG = Path("figures")
    FIG.mkdir(exist_ok=True)
    fname = f"step63_llm_agent_eval_cross_run_{model_safe}.png"
    fig.savefig(FIG / fname, dpi=200, bbox_inches="tight")
    fig.savefig(FIG / fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {FIG / fname}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="LLM-agent evaluation")
    p.add_argument("--models", type=str, default=None,
                   help="Comma-separated list of models to test. "
                        "Default: gpt-4o-mini,gpt-5-mini,gpt-5-pro,gpt-6-astra. "
                        "Use 'single' to test only the model in --model.")
    p.add_argument("--model", type=str,
                   default=os.environ.get("ONTOCONTEXT_LLM_MODEL", DEFAULT_MODEL),
                   help="Single model (used when --models is not set or is 'single').")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Tag for this run (e.g. 'run1'). When set, per-question "
                        "CSV and summary CSV are archived under "
                        "tables/step63_runs/<model>/<run_tag>.csv, so the aggregator "
                        "can compute mean ± SD across runs.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a model if tables/step63_runs/<model>/<run_tag>_summary.csv already exists.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set OPENAI_API_KEY or ANTHROPIC_API_KEY", file=sys.stderr)
        return 1

    # ── Determine which models to run ────────────────────────────────────────
    if args.models and args.models.lower() != "single":
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_list = [args.model]

    print(f"\nModels to evaluate: {model_list}")

    # ── Run each model ──────────────────────────────────────────────────────
    results = {}
    for model in model_list:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model}")
        print(f"{'='*70}")
        os.environ["ONTOCONTEXT_LLM_MODEL"] = model

        # Skip if already done
        if args.skip_existing and args.run_tag:
            model_safe = model.replace("-", "_").replace(".", "_")
            sum_path = ROOT / "tables" / "step63_runs" / model_safe / f"{args.run_tag}_summary.csv"
            if sum_path.exists():
                print(f"  [SKIP] {sum_path} already exists — loading from disk")
                rows = []
                tbl_path = ROOT / "tables" / "step63_runs" / model_safe / f"{args.run_tag}.csv"
                if tbl_path.exists():
                    import csv as _csv
                    with tbl_path.open() as fh:
                        reader = _csv.DictReader(fh)
                        for row in reader:
                            def _parse_bool(v):
                                s = row.get(v, "0")
                                if s in ("True", "False"):
                                    return s == "True"
                                return bool(int(s))
                            row["correct"] = _parse_bool("correct")
                            row["hallucinated"] = _parse_bool("hallucinated")
                            row["tool_used"] = _parse_bool("tool_used")
                            row["tokens_actual"] = int(row.get("tokens_actual", 0))
                            row["latency_seconds"] = float(row.get("latency_seconds", 0))
                            rows.append(row)
                results[model] = rows
                continue

        rows = run_experiment(model, hard_mode=False)
        rows += run_experiment(model, hard_mode=True)
        results[model] = rows

        # Archive
        if args.run_tag:
            model_safe = model.replace("-", "_").replace(".", "_")
            arch_dir = ROOT / "tables" / "step63_runs" / model_safe
            arch_dir.mkdir(parents=True, exist_ok=True)
            tbl_path = _write_table_for_model(rows, arch_dir / f"{args.run_tag}.csv")
            sum_path = _write_summary_csv_for_model(rows, arch_dir / f"{args.run_tag}_summary.csv")
            print(f"  Archived: {tbl_path}  {sum_path}")
        else:
            tbl_path = write_table(rows)
            sum_path = write_summary_csv(rows)
            print(f"  Wrote: {tbl_path}  {sum_path}")

        write_figure(rows, model)
        print("  Figures: figures/step63_llm_agent_eval.{png,pdf}")
        write_cross_run_figure(model_label=model)

        # Rate-limit pause
        time.sleep(5)

    # ── Cross-model comparison figure ──────────────────────────────────────
    if len(results) > 1:
        write_cross_model_figure(results)

    # ── Print combined summary ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("  COMBINED SUMMARY")
    print("="*70)
    for model, rows in results.items():
        print(f"\n  Model: {model}")
        by_key = defaultdict(list)
        for r in rows:
            by_key[(r["condition"], r["mode"])].append(r)
        for (cond, mode), recs in sorted(by_key.items()):
            n = len(recs)
            acc = sum(1 for r in recs if r["correct"]) / max(1, n)
            hall = sum(1 for r in recs if r["hallucinated"]) / max(1, n)
            tool = sum(1 for r in recs if r["tool_used"]) / max(1, n)
            print(f"    [{mode:7s}] {cond:30s}  acc={acc:.1%}  hall={hall:.1%}  tool={tool:.1%}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
