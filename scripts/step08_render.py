"""
step08_render.py — adaptive budgeted renderer + exact-tool routing.

PLAN.md step 08 contract:
    renderer.render(query, world_model, budget_tokens) -> RenderedContext

`RenderedContext` (rule §1):
    text: str                     # exactly what goes to the LLM
    tokens_used: int              # counted by tiktoken (cl100k_base)
    selected_capsules: list[str]
    selected_modules: list[str]
    exact_tool_calls: list[str]
    synopsis_used: list[str]      # always empty in V3 (rule §5)

The renderer always starts from the **global skeleton** (rule §2):
    1. Read the top-level modules of the world model.
    2. For each module, render its `centroid_term` name + a 1-line
       "what this module covers" hint (derived from the centroid
       capsule's definition).
    3. Find the modules whose term list intersects the query's
       `matched_terms` (an input field; if empty, the renderer
       assumes the query targets the global skeleton only).
    4. For each matched module, render up to `B(v)` tokens worth
       of capsule summaries (per the Utility formula in
       ontocontext-rendering-protocol.mdc §3).
    5. For any "formal" question, emit an `exact_tool_calls` pointer
       (per Invariant I).
    6. The total `tokens_used` is measured with tiktoken's
       cl100k_base; the renderer never exceeds `budget_tokens`.

In this V3 implementation we use a **dependency-free** token
estimator: bytes / 4 (a coarse but stable approximation). The
real tokenizer is plumbed in `step08_render_a_diagnostic.py`
where tiktoken is optional.

Usage:
    python3 scripts/step08_render.py --snapshot synth_d02_b02_s017 \\
        --query "What is the root term?" --budget 256
    python3 scripts/step08_render.py --snapshot go_2024-06-10 \\
        --query "Which GO terms relate to cell death?" --budget 1024
    # diagnostic mode: also write a render-trace figure
    python3 scripts/step08_render.py --snapshot synth_d02_b02_s017 \\
        --query "..." --budget 256 --diagnose
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

# Determinism per the plan.
np.random.seed(17)

DEFAULT_BUDGET = 1024
SCHEMA_VERSION = "0.1.0"
REPORT_DIR = Path("evaluation/reports")

# Ablation toggle: when True, the V3 query-priority expansion block
# (the 60% budget reserve for the queried CURIE + its module-mates,
# and the always-full relations/cross emission in the priority lane)
# is bypassed. This isolates the contribution of that V3 fix to the
# relation-completion / sibling-discrimination benchmarks.
#
# The flag is exposed as both an env-var (ONTO_NO_V3_FIX=1) and a
# module attribute (step08_render.ONTO_NO_V3_FIX = True) so callers
# (sweeps, ablation scripts) can flip it without forking the code.
ONTO_NO_V3_FIX = bool(int(os.environ.get("ONTO_NO_V3_FIX", "0")))

# Token-budget constants from the rendering protocol §3.
B_0 = 64
ALPHA = 256
BETA = 192
GAMMA = 48
DELTA = 64
ETA = 64   # reserved for change-risk; unused in V3

# Format-string approximation: 1 token ≈ 4 chars of plain English.
CHARS_PER_TOKEN = 4.0


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Coarse token estimator: max(1, round(len(text) / 4))."""
    if not text:
        return 0
    return max(1, int(round(len(text) / CHARS_PER_TOKEN)))


# ---------------------------------------------------------------------------
# World model loaders
# ---------------------------------------------------------------------------

def _load_world_model(snapshot_id: str, ontology: str) -> tuple[list[dict], dict[str, dict]]:
    """Return (modules, entities_by_id) for a snapshot/ontology.

    Both lists come from the per-snapshot world_model.jsonl; we filter
    to the requested ontology (a single world model can hold multiple).
    """
    wm_path = Path("capsules/v3/world_model.jsonl")
    if not wm_path.exists():
        raise RuntimeError(
            f"no world model at {wm_path}; run step 07 first"
        )
    modules: list[dict] = []
    entities: dict[str, dict] = {}
    for line in wm_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("snapshot_id") != snapshot_id or d.get("ontology") != ontology:
            continue
        if d["kind"] == "module":
            modules.append(d)
        else:
            entities[d["term_id"]] = d
    return modules, entities


def _load_capsules_for(snapshot_id: str, ontology: str) -> dict[str, dict]:
    """Load V3 capsules (full JSON) for the renderer's capsule text."""
    src = Path("capsules/v3/entities") / ontology
    if not src.exists():
        return {}
    out: dict[str, dict] = {}
    for p in src.glob("*.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        if c.get("snapshot_id") == snapshot_id:
            out[c["id"]] = c
    return out


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _capsule_summary(c: dict, char_budget: int) -> str:
    """Render a capsule as a fixed-template summary.

    The char_budget is a **soft** cap on the *name + def + features*
    block (per-capsule header). The *relations* and *cross* lines
    are NEVER truncated mid-CURIE — they are emitted in full so the
    benchmarks (relation-completion, cross-ontology) can match by
    CURIE-substring.
    """
    header_parts: list[str] = []
    header_parts.append(f"[{c['id']}] {c.get('name') or c['id']}")
    defn = c.get("identity", {}).get("definition") or ""
    if defn:
        header_parts.append(f"  - def: {defn}")
    syns = c.get("identity", {}).get("synonyms", [])
    if syns:
        header_parts.append("  - syn: " + "; ".join(
            s.get("text", "") for s in syns[:3]
        ))
    feat = c.get("disambiguation", {}).get("distinguishing_features", [])
    if feat:
        header_parts.append("  - features: " + ", ".join(feat[:4]))
    header_text = "\n".join(header_parts)[:char_budget]

    tail_parts: list[str] = []
    relations = c.get("logical_signature", {}).get("defining_relations", []) or []
    if relations:
        # Strip IRI prefixes (obo:RO_..., http://.../BFO_...) so the
        # surface form is the bare CURIE-local, which is both
        # human-readable AND what the relation-completion benchmark
        # matches against.
        rels_local: list[str] = []
        for r in relations:
            r2 = r
            for sep in ("/", ":"):
                if sep in r2:
                    r2 = r2.rsplit(sep, 1)[-1]
            rels_local.append(r2)
        # Emit up to 8 relations (was 4) so the benchmark has more signal.
        tail_parts.append("  - relations: " + ", ".join(rels_local[:8]))
    cross = c.get("cross_ontology", {}).get("relations", [])
    if cross:
        # Always emit up to 6 cross-relations in full (no truncation
        # mid-CURIE) so the cross-ontology benchmark can match.
        cross_strs: list[str] = []
        for r in cross[:6]:
            to = r.get("to") or r.get("object_id") or ""
            rel = r.get("relation") or "xref"
            cross_strs.append(f"{to} ({rel})")
        tail_parts.append("  - cross: " + ", ".join(cross_strs))
    tail_text = "\n".join(tail_parts)
    if not tail_text:
        return header_text
    return header_text + "\n" + tail_text


def _module_summary(m: dict, entities: dict[str, dict], char_budget: int) -> str:
    centroid = m.get("centroid_term")
    n = m.get("size", 0)
    txt = f"module {m['module_id']} ({n} terms) — centroid: {centroid}"
    if centroid and centroid in entities:
        ent = entities[centroid]
        ic = ent.get("information_content", 0.0)
        txt += f", IC={ic:.2f}"
    return txt[:char_budget]


def _extract_curie_from_query(query: str) -> str | None:
    """If the query mentions an OBO-style CURIE (e.g. CL:0000235,
    GO:0008150), return it. Otherwise return None. This is the bridge
    from natural-language question to capsule expansion priority."""
    m = re.search(r"\b([A-Z][A-Za-z0-9]+:[A-Z0-9_]{4,12})\b", query)
    return m.group(1) if m else None


def _is_formal_query(query: str) -> bool:
    """Heuristic: queries with formal-ontology verbs (compare, list,
    axiomatize, get_axioms) get exact_tool_calls pointers."""
    ql = query.lower()
    return any(k in ql for k in ("compare", "axioms of", "get_axioms",
                                  "list children of", "list parents of",
                                  "is_a path", "subsumers of"))


def _budget_for_capsule(c: dict, max_ann: int) -> int:
    """Per-capsule budget from the rendering protocol §3, in tokens."""
    ic = c.get("usage", {}).get("information_content", 0.0) or 0.0
    amb = 0.0          # V3 has no explicit ambiguity signal
    conn = 0.0         # V3 has no explicit connectivity signal
    ann = c.get("usage", {}).get("annotation_frequency", 0) or 0
    if max_ann == 0:
        ann_norm = 0.0
    else:
        ann_norm = math.log(1 + ann) / math.log(1 + max_ann)
    return B_0 + int(ALPHA * ic) + int(BETA * amb) + int(GAMMA * conn) + int(DELTA * ann_norm)


def render(
    query: str, snapshot_id: str, ontology: str, budget_tokens: int = DEFAULT_BUDGET,
) -> dict:
    """Render a query against the world model under a token budget.

    Returns a dict with the §1 RenderedContext fields.

    Priority logic (added for real ontologies with many modules):
    1. If the query mentions a CURIE, reserve a fixed slice of the
       budget for the queried capsule + its module neighbours.
       The remaining budget is used for the global skeleton +
       budget-ranked expansion.
    2. If no CURIE is mentioned, fall back to the previous behaviour
       (global skeleton → per-capsule expansion by Utility).
    """
    modules, entities = _load_world_model(snapshot_id, ontology)
    capsules = _load_capsules_for(snapshot_id, ontology)
    if not modules:
        raise RuntimeError(f"no modules in world model for {snapshot_id}/{ontology}")

    selected_capsules: list[str] = []
    selected_modules: list[str] = []
    expansion_lines: list[str] = []

    # ---- 0) Query-priority expansion (real-ontology fix) ----
    # Reserve 60% of the budget for the queried term (if any) and
    # its module's top-IC neighbours. The remaining 40% is for the
    # global skeleton + secondary expansion.
    #
    # The V3 fix is gated by `ONTO_NO_V3_FIX`: when True, we skip
    # the priority block entirely and the renderer falls back to
    # the pre-V3 (global-skeleton + secondary expansion) behaviour.
    query_curie = _extract_curie_from_query(query)
    prio_text: str = ""
    prio_modules: list[str] = []
    prio_capsules: list[str] = []
    if (not ONTO_NO_V3_FIX) and query_curie and query_curie in capsules:
        target = capsules[query_curie]
        target_module = target.get("coordinates", {}).get("semantic_module")
        # 1) The queried term itself (always)
        seen: set[str] = {query_curie}
        ordered: list[tuple[str, dict]] = [(query_curie, target)]
        # 2) Its selected_ancestors (parents) — these are the gold
        #    for hierarchy questions and should always be on screen.
        for anc_id in target.get("coordinates", {}).get("selected_ancestors", []) or []:
            if anc_id in seen:
                continue
            if anc_id in capsules:
                seen.add(anc_id)
                ordered.append((anc_id, capsules[anc_id]))
        # 3) Module-mates, sorted by IC desc, top-(8-len(ordered))
        if target_module:
            module_terms = [
                (tid, c) for tid, c in capsules.items()
                if tid not in seen
                and c.get("coordinates", {}).get("semantic_module") == target_module
            ]
            module_terms.sort(
                key=lambda kv: -float(kv[1].get("usage", {}).get("information_content", 0.0))
            )
            ordered.extend(module_terms[: max(0, 8 - len(ordered))])
        prio_budget = max(64, int(budget_tokens * 0.60))
        char_budget = prio_budget * int(CHARS_PER_TOKEN)
        prio_lines = ["== QUERY-TARGET CONTEXT =="]
        for tid, c in ordered:
            # Per-capsule char budget is large (1024 chars) so the
            # full cross_ontology + relations blocks fit. The
            # outer char_budget still caps the total priority block.
            summary = _capsule_summary(c, 1024)
            prio_lines.append(summary)
            prio_capsules.append(tid)
        prio_text = "\n".join(prio_lines)[:char_budget]
        if target_module:
            prio_modules = [target_module]

    # ---- 1) Build the global skeleton (rule §2) ----
    # If we used 60% of the budget for query context, the skeleton
    # uses ≤ 40% — and we trim modules to fit.
    skeleton_budget = budget_tokens - _estimate_tokens(prio_text) if prio_text else budget_tokens
    skeleton_lines: list[str] = [
        f"== GLOBAL SKELETON ({len(modules)} modules) ==",
    ]
    for m in sorted(modules, key=lambda x: x["module_id"]):
        skeleton_lines.append(_module_summary(m, entities, 80))
        if m["module_id"] not in selected_modules:
            selected_modules.append(m["module_id"])
    skeleton_text = "\n".join(skeleton_lines)
    tokens_used = _estimate_tokens(skeleton_text)
    if tokens_used > skeleton_budget:
        # Trim modules from the end to fit
        for k in range(len(skeleton_lines) - 1, 0, -1):
            skeleton_text = "\n".join(skeleton_lines[:k])
            tokens_used = _estimate_tokens(skeleton_text)
            if tokens_used <= skeleton_budget:
                selected_modules = selected_modules[:k]
                break

    # ---- 2) Secondary expansion (remaining budget) ----
    remaining = budget_tokens - tokens_used - _estimate_tokens(prio_text) if prio_text \
                else budget_tokens - tokens_used
    max_ann = max((c.get("usage", {}).get("annotation_frequency", 0) or 0)
                  for c in capsules.values()) or 1
    capsule_budget = sorted(
        [c for cid, c in capsules.items() if cid not in prio_capsules],
        key=lambda c: -_budget_for_capsule(c, max_ann),
    )
    for c in capsule_budget:
        if remaining <= 0:
            break
        cb = _budget_for_capsule(c, max_ann)
        char_budget = cb * int(CHARS_PER_TOKEN)
        summary = _capsule_summary(c, char_budget)
        st = _estimate_tokens(summary)
        if st > remaining:
            continue
        expansion_lines.append(summary)
        selected_capsules.append(c["id"])
        remaining -= st

    # ---- 3) Compose final text ----
    text = ""
    if prio_text:
        text += prio_text + "\n\n"
    text += skeleton_text
    if expansion_lines:
        text += "\n\n== EXPANDED CAPSULES ==\n" + "\n".join(expansion_lines)
    if _is_formal_query(query):
        text += "\n\n== EXACT TOOLS ==\n"
        text += (" - get_axioms(term_id) for any term in selected_capsules\n"
                  " - compare_terms(term_id_a, term_id_b) for sibling disambiguation\n"
                  " - list_parents(term_id) and list_children(term_id) for hierarchy\n")
    tokens_used = _estimate_tokens(text)
    if tokens_used > budget_tokens:
        ratio = budget_tokens / max(1, tokens_used)
        text = text[: int(len(text) * ratio)]
        tokens_used = _estimate_tokens(text)

    exact_tool_calls: list[str] = []
    if _is_formal_query(query):
        exact_tool_calls = [
            "get_axioms(term_id)",
            "compare_terms(term_id_a, term_id_b)",
        ]

    return {
        "text": text,
        "tokens_used": tokens_used,
        "selected_capsules": prio_capsules + selected_capsules,
        "selected_modules": prio_modules + selected_modules,
        "exact_tool_calls": exact_tool_calls,
        "synopsis_used": [],
        "query": query,
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "budget_tokens": budget_tokens,
        "schema_version": SCHEMA_VERSION,
        "rendered_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Render logging (one JSON per query, per step 08 spec)
# ---------------------------------------------------------------------------

def _write_render_report(rc: dict, run_id: str) -> Path:
    out_dir = REPORT_DIR / run_id / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    qid = "q_" + rc["query"][:32].replace(" ", "_").replace("?", "")
    out = out_dir / f"{qid}.json"
    out.write_text(json.dumps(rc, indent=2, sort_keys=True), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--diagnose", action="store_true",
                   help="also write a render-trace figure")
    p.add_argument("--run-id", type=str, default="default")
    p.add_argument("--no-v3-fix", action="store_true",
                   help="disable the V3 query-priority expansion block "
                        "(ablation: isolates the V3 fix's contribution)")
    return p.parse_args(argv)


def _infer_ontology(snapshot_id: str) -> str:
    """Infer ontology from the processed manifest, else from snapshot id."""
    proc = Path("data/processed") / snapshot_id
    mf = proc / "manifest_processed.yaml"
    if mf.exists():
        try:
            return yaml.safe_load(mf.read_text())["ontology"]
        except Exception:
            pass
    return snapshot_id.split("_")[0].upper()


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.no_v3_fix:
        # Flip the module-level flag so `render()` sees it without
        # having to thread an extra argument through every call site.
        global ONTO_NO_V3_FIX
        ONTO_NO_V3_FIX = True
    ontology = _infer_ontology(args.snapshot)
    rc = render(args.query, args.snapshot, ontology, args.budget)
    out = _write_render_report(rc, args.run_id)
    print(f"✅ Rendered: tokens={rc['tokens_used']}/{rc['budget_tokens']} "
          f"modules={len(rc['selected_modules'])} capsules={len(rc['selected_capsules'])} "
          f"tools={len(rc['exact_tool_calls'])} → {out}")
    if args.diagnose:
        try:
            # Lazy import: matplotlib only for diagnostic
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from pathlib import Path
            fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
            ax.barh(["global skeleton", "expanded capsules"],
                    [rc['tokens_used'] - _estimate_tokens("\n\n".join(rc.get("text", "").split("\n\n== EXPANDED CAPSULES ==")[1:]) or ""),
                     _estimate_tokens("\n\n".join(rc.get("text", "").split("\n\n== EXPANDED CAPSULES ==")[1:]) or "")],
                    color=["#3b6fb6", "#7a5fb0"])
            ax.set_xlabel("tokens used")
            ax.set_title(f"Step 08 — render distribution ({args.snapshot})")
            fig_dir = Path("figures"); fig_dir.mkdir(parents=True, exist_ok=True)
            base = f"step08_a_render_distribution_{args.snapshot}"
            fig.savefig(fig_dir / f"{base}.png", dpi=200, bbox_inches="tight")
            fig.savefig(fig_dir / f"{base}.pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"   ↪ figures/{base}.{{png,pdf}}")
        except Exception as e:
            print(f"   ! diagnostic failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
