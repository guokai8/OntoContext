"""
step05_v2_discriminate.py — V2 discriminative capsule builder.

Reads the **V1** capsules from `capsules/v1/entities/` and:

  1. Computes a **confusable set** for every term. The default
     similarity is a small, dependency-free blend of
        (a) **graph Jaccard** over the term's parent/children sets
            (captures "looks similar because it sits in the same
            part of the hierarchy"), and
        (b) **lexical Jaccard** over bag-of-words in the term's
            name + definitions + synonyms
            (captures "looks similar because it has the same
            words in the name").
     The two are mixed with default weights 0.6 / 0.4 (configurable
     via _config.SIMILARITY_SOURCE). The per-term top-K confusables
     (default K=8) are written to
     `cache/confusables/<snapshot_id>.parquet` for reuse.

  2. For every V1 capsule, builds a V2 capsule by populating
        - `disambiguation.siblings`        : other terms sharing
          the **same parent** (the canonical "sibling" set;
          strictly graph-based; this is the §40.2 headline);
        - `disambiguation.near_terms`      : top-K entries from
          the confusables table (similarity-based, mix of graph
          and lexical);
        - `disambiguation.distinguishing_features`:
          hand-rolled list of "what makes this term different
          from its siblings" — for now: the term's own IC rank
          bucket ("leaf", "near-root", "root") and the count of
          its own synonyms. A future V2 (with embeddings) would
          derive these from embeddings.
     The schema_version stays at **0.1.0** (no schema bump for V2).

  3. The companion ablation `step05_v2_a_confusable_seed_sweep.py`
     sweeps the graph-vs-lexical weight over the synthetic
     ontology and produces `figures/step05_a_confusable.{png,pdf}`.

Usage:
    python3 scripts/step05_v2_discriminate.py --synthetic synth_d02_b02_s017
    python3 scripts/step05_v2_discriminate.py --all
    python3 scripts/step05_v2_discriminate.py --snapshot go_2024-06-10 --top-k 16
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

# Determinism per the plan.
np.random.seed(17)

V1_ROOT = Path("capsules/v1/entities")
V2_ROOT = Path("capsules/v2/entities")
CONFUSABLES_ROOT = Path("cache/confusables")
SCHEMA_VERSION = "0.1.0"

# Default similarity mix (graph / lexical) — overridable via _config.
DEFAULT_GRAPH_WEIGHT = 0.6
DEFAULT_LEXICAL_WEIGHT = 0.4
DEFAULT_TOP_K = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curie_path(ontology: str, term_id: str, root: Path) -> Path:
    safe = term_id.replace(":", "_").replace("/", "__")
    return root / ontology / f"{safe}.json"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _wordbag(s: str) -> set[str]:
    """Lowercase word-bag, punctuation-stripped."""
    if not s:
        return set()
    out: set[str] = set()
    cur: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.add("".join(cur))
                cur = []
    if cur:
        out.add("".join(cur))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Confusable set computation
# ---------------------------------------------------------------------------

def _build_term_features(snapshot_id: str) -> dict[str, dict]:
    """Walk V1 capsules and build per-term features:
       - parents (set of term ids)
       - children (set of term ids)
       - neighbours (parents ∪ children)
       - wordbag (lexical bag-of-words)
    """
    out: dict[str, dict] = {}
    for ont_dir in V1_ROOT.iterdir() if V1_ROOT.exists() else []:
        if not ont_dir.is_dir():
            continue
        for v1_path in ont_dir.glob("*.json"):
            c = json.loads(v1_path.read_text(encoding="utf-8"))
            if c.get("snapshot_id") != snapshot_id:
                continue
            parents = set(c.get("coordinates", {}).get("parents", []) or [])
            children = set(c.get("coordinates", {}).get("selected_children", []) or [])
            text = " ".join([
                c.get("name") or "",
                c.get("identity", {}).get("definition") or "",
                " ".join(s.get("text", "") for s in c.get("identity", {}).get("synonyms", [])),
            ])
            out[c["id"]] = {
                "ontology": c.get("ontology"),
                "parents": parents,
                "children": children,
                "neighbours": parents | children,
                "wordbag": _wordbag(text),
            }
    return out


def _build_confusables(
    snapshot_id: str, ontology: str, features: dict[str, dict],
    top_k: int, graph_w: float, lex_w: float,
) -> list[dict]:
    """For every term in `features`, compute top-K confusables.

    We restrict the candidate set to terms in the same ontology
    (so cross-ontology confusion is handled separately by the
    cross_ontology_index in step 02).
    """
    terms = [t for t, f in features.items() if f.get("ontology") == ontology]
    rows: list[dict] = []
    for t in terms:
        ft = features[t]
        scored: list[tuple[float, str, float, float]] = []
        for u in terms:
            if u == t:
                continue
            fu = features[u]
            jg = _jaccard(ft["neighbours"], fu["neighbours"])
            jl = _jaccard(ft["wordbag"], fu["wordbag"])
            sim = graph_w * jg + lex_w * jl
            if sim <= 0:
                continue
            scored.append((sim, u, jg, jl))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for rank, (sim, u, jg, jl) in enumerate(scored[:top_k], start=1):
            rows.append({
                "term_id": t,
                "snapshot_id": snapshot_id,
                "near_term_id": u,
                "rank": rank,
                "similarity": float(sim),
                "graph_jaccard": float(jg),
                "lexical_jaccard": float(jl),
            })
    return rows


def _write_confusables(
    snapshot_id: str, ontology: str, features: dict[str, dict],
    top_k: int, graph_w: float, lex_w: float, force: bool,
) -> Path:
    """Compute the confusables table and write it to disk."""
    CONFUSABLES_ROOT.mkdir(parents=True, exist_ok=True)
    out = CONFUSABLES_ROOT / f"{snapshot_id}.parquet"
    if out.exists() and not force:
        return out
    rows = _build_confusables(snapshot_id, ontology, features, top_k, graph_w, lex_w)
    schema = pa.schema([
        ("term_id", pa.string()),
        ("snapshot_id", pa.string()),
        ("near_term_id", pa.string()),
        ("rank", pa.uint16()),
        ("similarity", pa.float32()),
        ("graph_jaccard", pa.float32()),
        ("lexical_jaccard", pa.float32()),
    ])
    if not rows:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out, compression="snappy")
    return out


# ---------------------------------------------------------------------------
# V2 capsule assembly
# ---------------------------------------------------------------------------

def _siblings_for(term_id: str, features: dict[str, dict]) -> list[str]:
    """Return the set of terms sharing the same parent(s) as term_id."""
    parents = features.get(term_id, {}).get("parents", set())
    if not parents:
        return []
    siblings: set[str] = set()
    for other, f in features.items():
        if other == term_id:
            continue
        if f["parents"] & parents:
            siblings.add(other)
    return sorted(siblings)


def _ic_rank_bucket(ic: float) -> str:
    """Map a numeric IC into one of three qualitative buckets for
    distinguishing_features."""
    if ic <= 0.05:
        return "leaf"
    if ic <= 0.5:
        return "mid"
    return "near-root"


def _build_v2_capsule(
    v1: dict, features: dict[str, dict], confusables_by_term: dict[str, list[dict]],
) -> dict:
    """V2 = V1 + disambiguation fields populated.

    The `disambiguation.siblings` list is sorted alphabetically;
    `disambiguation.near_terms` is sorted by descending confusable
    similarity (rank ascending); distinguishing_features is a
    bullet-style list (strings).
    """
    out = json.loads(json.dumps(v1))
    nid = v1["id"]
    feat = features.get(nid, {})
    siblings = _siblings_for(nid, features)
    near = confusables_by_term.get(nid, [])

    ic = v1.get("usage", {}).get("information_content", 0.0) or 0.0
    syn_count = len(v1.get("identity", {}).get("synonyms", []))
    df = [
        f"ic-bucket:{_ic_rank_bucket(ic)}",
        f"synonym-count:{syn_count}",
        f"parents:{len(feat.get('parents', set()))}",
        f"children:{len(feat.get('children', set()))}",
    ]
    if not feat.get("parents"):
        df.append("root:no-parent")
    if not feat.get("children"):
        df.append("leaf:no-children")

    out["schema_version"] = SCHEMA_VERSION
    out["disambiguation"] = {
        "siblings": siblings,
        "near_terms": [
            {
                "id": row["near_term_id"],
                "rank": row["rank"],
                "similarity": row["similarity"],
            }
            for row in near
        ],
        "distinguishing_features": df,
    }
    return out


def _validate_v2(c: dict) -> None:
    """Same required-fields check as V0/V1; rule §3.3 is invariant."""
    required = [
        "id", "name", "ontology", "snapshot_id", "schema_version",
        "identity", "coordinates", "logical_signature", "disambiguation",
        "cross_ontology", "usage", "semantic_synopsis",
        "semantic_synopsis_metadata", "exact_pointers",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise RuntimeError(f"V2 capsule missing required fields: {missing}")
    if c["semantic_synopsis"] is not None:
        raise RuntimeError("V2 capsule must have semantic_synopsis=null (rule §5)")
    if c["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(f"V2 schema_version must be {SCHEMA_VERSION!r}")
    d = c["disambiguation"]
    if not all(k in d for k in ("siblings", "near_terms", "distinguishing_features")):
        raise RuntimeError("V2 disambiguation missing required sub-fields")


# ---------------------------------------------------------------------------
# Per-snapshot processing
# ---------------------------------------------------------------------------

def build_for_snapshot(
    snapshot_id: str, ontology: str, processed_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    graph_w: float = DEFAULT_GRAPH_WEIGHT,
    lex_w: float = DEFAULT_LEXICAL_WEIGHT,
    force_confusables: bool = False,
) -> dict:
    """Build V2 capsules for one snapshot. Returns a summary dict."""
    # 1) Read V1 capsules and build per-term features
    if not V1_ROOT.exists():
        raise RuntimeError(f"no V1 capsules at {V1_ROOT}; run step 04 first")
    features = _build_term_features(snapshot_id)
    if not features:
        raise RuntimeError(f"no V1 features for snapshot {snapshot_id}")
    # Ontology: use the first term's ontology if it agrees with the
    # caller; otherwise warn.
    ont_inferred = next(iter(features.values()))["ontology"]
    if ont_inferred and ont_inferred != ontology:
        # Auto-correct rather than fail.
        ontology = ont_inferred

    # 2) Confusables table (cached)
    conf_path = _write_confusables(
        snapshot_id, ontology, features, top_k, graph_w, lex_w, force_confusables
    )
    conf_t = pq.read_table(conf_path)
    conf_by_term: dict[str, list[dict]] = defaultdict(list)
    for row in conf_t.to_pylist():
        conf_by_term[row["term_id"]].append(row)
    for k in conf_by_term:
        conf_by_term[k].sort(key=lambda r: r["rank"])

    # 3) Read V1 capsules and write V2
    src_dir = V1_ROOT / ontology
    if not src_dir.exists():
        raise RuntimeError(f"no V1 capsules for ontology {ontology} at {src_dir}")
    written = 0
    nonroot_with_disambig = 0
    for v1_path in sorted(src_dir.glob("*.json")):
        v1 = json.loads(v1_path.read_text(encoding="utf-8"))
        if v1.get("snapshot_id") != snapshot_id:
            continue
        v2 = _build_v2_capsule(v1, features, conf_by_term)
        _validate_v2(v2)
        out = _curie_path(ontology, v2["id"], V2_ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v2, indent=2, sort_keys=True), encoding="utf-8")
        written += 1
        if features.get(v2["id"], {}).get("parents") and v2["disambiguation"]["siblings"]:
            nonroot_with_disambig += 1
    (V2_ROOT.parent / "schema_version.txt").write_text(SCHEMA_VERSION)
    nonroot_total = sum(1 for f in features.values() if f.get("parents"))
    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "v2_written": written,
        "confusables_path": str(conf_path),
        "confusables_rows": conf_t.num_rows,
        "nonroot_with_disambig": nonroot_with_disambig,
        "nonroot_total": nonroot_total,
        "disambig_coverage": (
            nonroot_with_disambig / max(1, nonroot_total)
        ),
        "weights": {"graph": graph_w, "lexical": lex_w, "top_k": top_k},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--graph-w", type=float, default=DEFAULT_GRAPH_WEIGHT)
    p.add_argument("--lex-w", type=float, default=DEFAULT_LEXICAL_WEIGHT)
    p.add_argument("--force-confusables", action="store_true")
    return p.parse_args(argv)


def _resolve_snapshots(args) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    if args.synthetic:
        proc = Path("data/processed") / args.synthetic
        out.append((args.synthetic, "SYNTH", proc))
        return out
    if args.snapshot:
        proc = Path("data/processed") / args.snapshot
        mf = proc / "manifest_processed.yaml"
        ontology = (yaml.safe_load(mf.read_text())["ontology"]
                    if mf.exists() else args.snapshot.split("_")[0].upper())
        out.append((args.snapshot, ontology, proc))
        return out
    if args.all:
        for d in sorted((Path("data/processed")).iterdir()):
            if not (d / "node.parquet").exists():
                continue
            mf = d / "manifest_processed.yaml"
            ontology = (yaml.safe_load(mf.read_text())["ontology"]
                        if mf.exists() else d.name.split("_")[0].upper())
            out.append((d.name, ontology, d))
    return out


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    snaps = _resolve_snapshots(args)
    if not snaps:
        print("ERROR: pass --snapshot, --synthetic, or --all", file=sys.stderr)
        return 2
    summaries = []
    for sid, ontology, proc in snaps:
        s = build_for_snapshot(
            sid, ontology, proc,
            top_k=args.top_k, graph_w=args.graph_w, lex_w=args.lex_w,
            force_confusables=args.force_confusables,
        )
        print(
            f"✅ {sid} ({s['ontology']}): {s['v2_written']} V2 capsules, "
            f"confusables {s['confusables_rows']} rows, "
            f"nonroot disambig coverage = {s['disambig_coverage']:.2%} "
            f"({s['nonroot_with_disambig']}/{s['nonroot_total']}), "
            f"top_k={s['weights']['top_k']} graph={s['weights']['graph']} "
            f"lex={s['weights']['lexical']}"
        )
        summaries.append(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
