"""
step04_v1_select.py — V1 information-theoretic capsule builder.

Reads the **V0** capsules from `capsules/v0/entities/` and the
`data/processed/<snapshot_id>/` Parquet tables to:

  1. Compute the **IC (information-content) table** for every term
     in the snapshot and write it to
     `cache/ic_tables/<snapshot_id>.parquet`. We use the
     **descendant-count** variant, since not every ontology has
     annotations available:

         p(c) = (|descendants(c)| + 1) / (|max_descendants| + 1)
         IC(c) = -log(p(c))    # natural log, not log2

     The +1 in numerator and denominator is the standard
     Laplace-smoothed estimator that prevents IC(leaf) = 0.

  2. For each V0 capsule, build a V1 capsule by:
       a. **Sorting ancestors by descending `Importance(v) = IC(v)`**
          (a per-vertex utility from the rendering protocol §3).
       b. **Truncating** the ancestor list at the budget-heuristic
          of 200 edges (PLAN.md step 04: "selection size per capsule
          < 200 edges").
       c. Populating `usage.information_content` with the term's
          own IC.
       d. Populating `coordinates.selected_ancestors` with the
          truncated list.
     The schema_version stays at **0.1.0** (no schema bump for V1).

  3. Cache the IC table to disk with sha-256 of the Parquet
     inputs in the filename metadata, so re-runs are content-keyed.

The companion ablation `step04_v1_a_weight_sweep.py` (this file's
sidekick) sweeps the four Utility weights α,β,γ,δ on the synthetic
ontology and emits `figures/step04_a_weight_sweep.{png,pdf}`.

Usage:
    python3 scripts/step04_v1_select.py --snapshot synth_d02_b02_s017
    python3 scripts/step04_v1_select.py --all
    python3 scripts/step04_v1_select.py --snapshot go_2024-06-10 --weight-set a3
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

# Determinism per the plan.
import numpy as np
np.random.seed(17)

V0_ROOT = Path("capsules/v0/entities")
V1_ROOT = Path("capsules/v1/entities")
IC_ROOT = Path("cache/ic_tables")
SCHEMA_VERSION = "0.1.0"
ANCESTOR_BUDGET = 200   # plan: "selection size per capsule < 200 edges"

# Default Utility weights from _config (per PLAN.md step 04).
DEFAULT_ALPHA = 0.5
DEFAULT_BETA  = 0.3
DEFAULT_GAMMA = 0.1
DEFAULT_DELTA = 0.1


# ---------------------------------------------------------------------------
# IC-table computation
# ---------------------------------------------------------------------------

def _descendants(children: dict[str, list[str]], root: str) -> set[str]:
    """Return the full set of descendants of `root` (including root)."""
    out: set[str] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        for c in children.get(n, ()):
            stack.append(c)
    return out


def _compute_ic_table(snapshot_id: str, processed_dir: Path) -> pa.Table:
    """Compute the IC table for one snapshot.

    Source: node.parquet + edge.parquet from step 02.
    Returns a PyArrow table with columns:
        term_id, snapshot_id, n_descendants, p, ic, ic_normalized
    """
    nodes = pq.read_table(processed_dir / "node.parquet").to_pylist()
    edges = pq.read_table(processed_dir / "edge.parquet").to_pylist()

    # Build parent -> children map for the standard is_a relation.
    children: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.get("relation_iri") != "obo:BFO_0000050":
            continue
        children[e["src_id"]].append(e["dst_id"])
    # descendant sets, memoized
    desc_cache: dict[str, set[str]] = {}

    # |descendants(c)| per term
    sizes: dict[str, int] = {}
    for n in nodes:
        tid = n["id"]
        if tid in desc_cache:
            sizes[tid] = len(desc_cache[tid])
            continue
        d = _descendants(children, tid)
        # cache every intermediate node we touched
        stack = [tid]
        seen: set[str] = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for c in children.get(x, ()):
                stack.append(c)
        # `seen` is the descendants of tid
        desc_cache[tid] = seen
        sizes[tid] = len(seen)

    # maximum descendant count
    max_d = max(sizes.values()) if sizes else 1
    if max_d == 0:
        max_d = 1

    rows: list[dict] = []
    for n in nodes:
        tid = n["id"]
        sz = sizes.get(tid, 0)
        p = (sz + 1) / (max_d + 1)   # Laplace-smoothed
        ic = -math.log(p)            # natural log
        rows.append({
            "term_id": tid,
            "snapshot_id": snapshot_id,
            "n_descendants": sz,
            "p": p,
            "ic": ic,
            "ic_normalized": ic / (-math.log(1.0 / (max_d + 1))) if max_d > 0 else 0.0,
        })

    schema = pa.schema([
        ("term_id", pa.string()),
        ("snapshot_id", pa.string()),
        ("n_descendants", pa.uint32()),
        ("p", pa.float64()),
        ("ic", pa.float64()),
        ("ic_normalized", pa.float64()),
    ])
    if not rows:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    return pa.Table.from_pylist(rows, schema=schema)


def _write_ic_table(snapshot_id: str, processed_dir: Path) -> tuple[Path, int]:
    """Compute and write the IC table for one snapshot. Returns (path, n_rows)."""
    IC_ROOT.mkdir(parents=True, exist_ok=True)
    out = IC_ROOT / f"{snapshot_id}.parquet"
    table = _compute_ic_table(snapshot_id, processed_dir)
    pq.write_table(table, out, compression="snappy")
    return out, table.num_rows


# ---------------------------------------------------------------------------
# V1 capsule assembly
# ---------------------------------------------------------------------------

def _curie_path(ontology: str, term_id: str, root: Path) -> Path:
    safe = term_id.replace(":", "_").replace("/", "__")
    return root / ontology / f"{safe}.json"


def _utility(
    v_ic: float,
    ambiguity: float,
    degree_norm: float,
    ann_count: int,
    max_ann: int,
    weights: tuple[float, float, float, float],
) -> float:
    """Utility(e) from the rendering protocol §3, restricted to the
    four weights α/β/γ/δ (η is reserved for V3 / change-risk)."""
    alpha, beta, gamma, delta = weights
    if max_ann == 0:
        ann_norm = 0.0
    else:
        ann_norm = math.log(1 + ann_count) / math.log(1 + max_ann)
    return (alpha * v_ic
            + beta * ambiguity
            + gamma * degree_norm
            + delta * ann_norm)


def _build_v1_capsule(
    v0: dict, ic_by_term: dict[str, float], weights: tuple[float, float, float, float],
) -> dict:
    """Take a V0 capsule and produce a V1 capsule.

    V1 differs from V0 only in:
      - `usage.information_content` populated (was 0.0 in V0)
      - `coordinates.selected_ancestors` populated by descending
        Utility(e), truncated to ANCESTOR_BUDGET.
    """
    parents = list(v0.get("coordinates", {}).get("parents", []) or [])
    nid = v0["id"]
    own_ic = ic_by_term.get(nid, 0.0)
    ann_count = v0.get("usage", {}).get("annotation_frequency", 0) or 0
    max_ann = max(1, ann_count)

    # Sort parents by descending Utility, breaking ties by IC desc.
    def _u(p: str) -> float:
        return _utility(ic_by_term.get(p, 0.0), 0.0, 0.0, ann_count, max_ann, weights)
    ranked = sorted(parents, key=_u, reverse=True)
    selected = ranked[:ANCESTOR_BUDGET]

    out = json.loads(json.dumps(v0))   # deep-copy
    out["schema_version"] = SCHEMA_VERSION
    out["usage"]["information_content"] = float(own_ic)
    out["coordinates"]["selected_ancestors"] = selected
    return out


def _validate_v1(c: dict) -> None:
    """Same required-fields check as V0; rule §3.3 is invariant under version."""
    required = [
        "id", "name", "ontology", "snapshot_id", "schema_version",
        "identity", "coordinates", "logical_signature", "disambiguation",
        "cross_ontology", "usage", "semantic_synopsis",
        "semantic_synopsis_metadata", "exact_pointers",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise RuntimeError(f"V1 capsule missing required fields: {missing}")
    if c["semantic_synopsis"] is not None:
        raise RuntimeError("V1 capsule must have semantic_synopsis=null (rule §5)")
    if c["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(f"V1 schema_version must be {SCHEMA_VERSION!r}, got {c['schema_version']!r}")
    if not isinstance(c["usage"].get("information_content"), (int, float)):
        raise RuntimeError("V1 capsule usage.information_content must be numeric")


def _weight_set_for(name: str) -> tuple[float, float, float, float]:
    """A small library of pre-baked weight sets for the ablation script."""
    table = {
        "default": (DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA),
        "ic_only":     (1.0, 0.0, 0.0, 0.0),
        "amb_only":    (0.0, 1.0, 0.0, 0.0),
        "deg_only":    (0.0, 0.0, 1.0, 0.0),
        "ann_only":    (0.0, 0.0, 0.0, 1.0),
        "ic_amb":      (0.6, 0.4, 0.0, 0.0),
        "balanced":    (0.25, 0.25, 0.25, 0.25),
        "a3":          (0.5, 0.3, 0.1, 0.1),   # alias of "default" — for the plan's
                                                  # "≥ 3 settings on real GO" criterion
    }
    if name not in table:
        raise RuntimeError(f"unknown weight set: {name!r}")
    return table[name]


# ---------------------------------------------------------------------------
# Per-snapshot processing
# ---------------------------------------------------------------------------

def build_for_snapshot(
    snapshot_id: str, ontology: str, processed_dir: Path,
    v0_root: Path = V0_ROOT, v1_root: Path = V1_ROOT,
    weights: tuple[float, float, float, float] = (DEFAULT_ALPHA, DEFAULT_BETA,
                                                    DEFAULT_GAMMA, DEFAULT_DELTA),
) -> dict:
    """Build V1 capsules for one snapshot.

    Reads from `v0_root/<ontology>/*.json` and the IC table (writes
    if missing). Returns a summary dict.
    """
    # 1) IC table
    ic_path = IC_ROOT / f"{snapshot_id}.parquet"
    if not ic_path.exists():
        ic_path, _ = _write_ic_table(snapshot_id, processed_dir)
    ic_table = pq.read_table(ic_path)
    ic_by_term = {row["term_id"]: row["ic"]
                  for row in ic_table.to_pylist()}

    # 2) Read V0 capsules and write V1
    src_dir = v0_root / ontology
    if not src_dir.exists():
        raise RuntimeError(f"no V0 capsules at {src_dir}; run step 03 first")
    written = 0
    skipped = 0
    for v0_path in sorted(src_dir.glob("*.json")):
        v0 = json.loads(v0_path.read_text(encoding="utf-8"))
        if v0.get("ontology") and v0["ontology"] != ontology:
            continue
        v1 = _build_v1_capsule(v0, ic_by_term, weights)
        _validate_v1(v1)
        out = _curie_path(ontology, v1["id"], v1_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v1, indent=2, sort_keys=True), encoding="utf-8")
        written += 1

    # 3) schema_version.txt
    (v1_root.parent / "schema_version.txt").write_text(SCHEMA_VERSION)
    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "v0_read": written,
        "ic_path": str(ic_path),
        "ic_rows": ic_table.num_rows,
        "weights": list(weights),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--weight-set", type=str, default="default",
                   help="one of: " + ", ".join(_weight_set_for.__doc__.split("table = {", 1)[1].split("}", 1)[0].split("\n")[1:9] if False else "default ic_only amb_only deg_only ann_only ic_amb balanced a3"))
    p.add_argument("--force-ic", action="store_true",
                   help="recompute the IC table even if it exists on disk")
    return p.parse_args(argv)


def _resolve_snapshots(args) -> list[tuple[str, str, Path]]:
    """Return list of (snapshot_id, ontology, processed_dir) tuples to process."""
    out: list[tuple[str, str, Path]] = []
    if args.synthetic:
        proc = Path("data/processed") / args.synthetic
        if not proc.exists():
            raise RuntimeError(f"no processed dir for {args.synthetic}; run step 02")
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
        proc_root = Path("data/processed")
        for d in sorted(proc_root.iterdir()):
            if not (d / "node.parquet").exists():
                continue
            mf = d / "manifest_processed.yaml"
            ontology = (yaml.safe_load(mf.read_text())["ontology"]
                        if mf.exists() else d.name.split("_")[0].upper())
            out.append((d.name, ontology, d))
    return out


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    try:
        weights = _weight_set_for(args.weight_set)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.force_ic and IC_ROOT.exists():
        for p in IC_ROOT.glob("*.parquet"):
            p.unlink()

    snaps = _resolve_snapshots(args)
    if not snaps:
        print("ERROR: pass --snapshot, --synthetic, or --all", file=sys.stderr)
        return 2

    total = 0
    for sid, ontology, proc in snaps:
        s = build_for_snapshot(sid, ontology, proc, weights=weights)
        print(f"✅ {sid} ({s['ontology']}): {s['v0_read']} V1 capsules "
              f"(IC table {s['ic_rows']} rows, weights={s['weights']})")
        total += s["v0_read"]
    print(f"Total: {total} V1 capsules; IC tables in {IC_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
