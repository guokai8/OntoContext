"""
step09_agent.py — benchmark agent.

Reads a question (from `evaluation/benchmarks/<family>/questions.jsonl`)
or accepts one on the CLI, calls the renderer (step 08) to get a
`RenderedContext`, and emits a JSON record per question to
`evaluation/reports/<run_id>/<qid>.json` with the §9 schema:

    { question, answer, tool_calls_used, tokens_used, latency_seconds }

In this V3 implementation we don't have a real LLM call: the
"answer" is a deterministic, dependency-free synthesis that
just composes the rendered context's first line into a reply.
The real-LLM hook is plumbed in `_call_llm` — when
`OPENAI_API_KEY` (or another provider) is present in the
environment, the script calls the model with the rendered
context as the user message; otherwise it falls back to the
deterministic synthesis.

Plan verification (PLAN.md step 09):
  agent runs end-to-end on ≥ 8 questions across 2 families
  without crashing; tools-used list is non-empty on ≥ 1
  question per family that implies formal operations.

Usage:
    python3 scripts/step09_agent.py --synthetic synth_d02_b02_s017 \\
        --family hierarchy --n 4 --budget 512
    python3 scripts/step09_agent.py --snapshot go_2024-06-10 \\
        --family localization --n 8 --budget 1024
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import render, _infer_ontology

# Determinism per the plan.
np.random.seed(17)

REPORT_DIR = Path("evaluation/reports")
BENCH_ROOT = Path("evaluation/benchmarks")

# A small set of demo questions per family, used when no
# questions.jsonl exists yet (early in the build).
DEMO_QUESTIONS: dict[str, list[str]] = {
    "localization": [
        "What is the anatomical localization of term GO:0008150?",
        "Where does biological_process occur in the cell?",
    ],
    "hierarchy": [
        "List the parents of GO:0008150.",
        "What is the top-level ancestor of cell differentiation?",
    ],
    "sibling-discrimination": [
        "Compare GO:0008150 with GO:0008151 — what distinguishes them?",
    ],
    "relation-completion": [
        "Which relations are missing for GO:0008150?",
    ],
    "multi-hop": [
        "Find a 2-hop path from GO:0008150 to GO:0000000.",
    ],
    "cross-ontology": [
        "Map CL:0000236 (astrocyte) to its UBERON equivalent.",
    ],
    "version-change": [
        "What changed in GO:0008150 between version 2024-06-10 and 2024-09-16?",
    ],
    "insufficient-information": [
        "Is GO:9999999 supported by the current ontology?",
    ],
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _deterministic_synth(rc: dict) -> str:
    """A dependency-free answer synthesis: take the rendered
    skeleton + first expanded capsule, and emit a short reply."""
    parts: list[str] = []
    parts.append(f"Query: {rc['query']}")
    parts.append(f"Rendered at {rc['rendered_at']}, "
                  f"budget={rc['budget_tokens']} tokens, "
                  f"used={rc['tokens_used']}.")
    parts.append("Modules considered: " + ", ".join(rc["selected_modules"]))
    parts.append("Capsules expanded: " + ", ".join(rc["selected_capsules"]))
    if rc["exact_tool_calls"]:
        parts.append("Tool calls emitted: " + ", ".join(rc["exact_tool_calls"]))
    parts.append("--- first 200 chars of context ---")
    parts.append(rc["text"][:200])
    return "\n".join(parts)


def _call_llm(rc: dict) -> str | None:
    """If a real LLM is configured, call it. Otherwise return None
    so the caller falls back to the deterministic synthesis."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        import openai  # type: ignore
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("ONTOCONTEXT_LLM_MODEL", "gpt-4o"),
            temperature=0.0,
            max_tokens=512,
            messages=[
                {"role": "system", "content": (
                    "You answer questions about an ontology. Use only the "
                    "context provided. Cite exact term IDs."
                )},
                {"role": "user", "content": rc["text"] + "\n\nQuestion: " + rc["query"]},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"   ! LLM call failed: {e}", file=sys.stderr)
        return None


def _load_questions(family: str) -> list[str]:
    """Read a benchmark folder's questions.jsonl; fall back to demos."""
    qf = BENCH_ROOT / family / "questions.jsonl"
    if qf.exists():
        out: list[str] = []
        for line in qf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(d.get("question") or d.get("q") or "")
        return [q for q in out if q]
    return DEMO_QUESTIONS.get(family, [])


def run(
    family: str, n: int, snapshot: str, ontology: str, budget: int, run_id: str,
) -> list[dict]:
    """Run the agent on `n` questions from `family`. Returns a list of
    per-question result dicts."""
    out_dir = REPORT_DIR / run_id / family
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = _load_questions(family)[:n]
    results: list[dict] = []
    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        rc = render(q, snapshot, ontology, budget)
        ans = _call_llm(rc) or _deterministic_synth(rc)
        latency = time.time() - t0
        rec = {
            "qid": f"{family}_{i:03d}",
            "family": family,
            "snapshot_id": snapshot,
            "ontology": ontology,
            "question": q,
            "answer": ans,
            "tool_calls_used": list(rc["exact_tool_calls"]),
            "tokens_used": rc["tokens_used"],
            "selected_capsules": list(rc["selected_capsules"]),
            "selected_modules": list(rc["selected_modules"]),
            "synopsis_used": list(rc["synopsis_used"]),
            "latency_seconds": latency,
            "schema_version": rc["schema_version"],
            "rendered_at": rc["rendered_at"],
        }
        out_path = out_dir / f"{rec['qid']}.json"
        out_path.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--run-id", type=str, default=None)
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
    run_id = args.run_id or f"agent_{_now_iso().replace(':', '').replace('-', '')}"
    results = run(args.family, args.n, snapshot, ontology, args.budget, run_id)
    print(f"✅ {args.family}: {len(results)} questions "
          f"(tokens used: mean={sum(r['tokens_used'] for r in results)/max(1,len(results)):.1f}, "
          f"tools used on ≥ 1: {any(r['tool_calls_used'] for r in results)}) "
          f"→ {REPORT_DIR / run_id / args.family}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
