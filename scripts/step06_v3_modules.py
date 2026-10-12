"""
step06_v3_modules.py — V3 hierarchical module map.

Reads the **V2** capsules from `capsules/v2/entities/` and
`data/processed/<snapshot_id>/` Parquet (for the edge list) and:

  1. Builds an undirected graph G from the `is_a` edges (and any
     other OBO relations, but `is_a` is the dominant signal for
     hierarchy).
  2. Runs **label propagation** to assign each term to a module
     (`cache/modules/<snapshot_id>/labelprop_<resolution>.parquet`).
     We use the vanilla Raghavan–Albert–Kumara label-propagation
     algorithm (no external deps; deterministic up to a seed).
  3. For every cluster of terms, writes a module JSON to
     `capsules/v3/modules/<module_id>.json` with the module's
     `name`, `terms` (list of term ids), `centroid_term`,
     `size`, and `parent_module` (for hierarchical rollups).
  4. Updates each V2 capsule's `coordinates.semantic_module` to
     point to the assigned module id, and writes V3 capsules to
     `capsules/v3/entities/<ontology>/<id>.json`. Schema_version
     stays at **0.1.0**.
  5. Persists `cache/modules/<snapshot_id>/labelprop_<resolution>.parquet`
     so the same clustering can be re-used by step 07/08 without
     re-running the algorithm.

The companion `step06_v3_a_alt_clustering.py` sweeps
`{labelprop, louvain, spectral, ontology_native}` over
resolutions {0.5, 1.0, 1.5, 2.0} and produces
`figures/step06_a_clustering_compare.{png,pdf}`.

Usage:
    python3 scripts/step06_v3_modules.py --synthetic synth_d02_b02_s017
    python3 scripts/step06_v3_modules.py --all
    python3 scripts/step06_v3_modules.py --snapshot go_2024-06-10 --resolution 1.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
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
SEED = 17
np.random.seed(SEED)
random.seed(SEED)

V2_ROOT = Path("capsules/v2/entities")
V3_ROOT = Path("capsules/v3/entities")
V3_MODULES_ROOT = Path("capsules/v3/modules")
MODULES_CACHE = Path("cache/modules")
SCHEMA_VERSION = "0.1.0"

DEFAULT_RESOLUTION = 1.0
DEFAULT_MAX_ITERS = 30
DEFAULT_ALGO = "labelprop"     # overridable via _config.MODULE_ALGO


# ---------------------------------------------------------------------------
# Label-propagation (Raghavan–Albert–Kumara, dependency-free)
# ---------------------------------------------------------------------------

def label_propagation(
    nodes: list[str],
    adj: dict[str, set[str]],
    *,
    max_iters: int = DEFAULT_MAX_ITERS,
    seed: int = SEED,
) -> dict[str, int]:
    """Vanilla label propagation on an undirected graph.

    Each node starts with a unique label. At each step, each node
    adopts the most-frequent label among its neighbours (ties broken
    by the node's own current label). Stops when the labelling
    stabilises or after `max_iters` steps.

    Returns: a dict mapping node → module id (an integer).
    """
    rng = random.Random(seed)
    labels: dict[str, int] = {n: i for i, n in enumerate(nodes)}

    for _ in range(max_iters):
        new_labels: dict[str, int] = {}
        order = list(nodes)
        rng.shuffle(order)        # break synchronous-update bias
        for n in order:
            nbr_labels = Counter(labels[m] for m in adj.get(n, ()))
            if not nbr_labels:
                new_labels[n] = labels[n]
                continue
            # Most-frequent neighbour label; tie-break by keeping own label.
            top_count = max(nbr_labels.values())
            top_labels = [l for l, c in nbr_labels.items() if c == top_count]
            if len(top_labels) == 1:
                new_labels[n] = top_labels[0]
            else:
                if labels[n] in top_labels:
                    new_labels[n] = labels[n]
                else:
                    new_labels[n] = min(top_labels)   # deterministic tie-break
        if new_labels == labels:
            break
        labels = new_labels

    # Re-map to dense module ids (0..k-1) for compactness
    unique = sorted(set(labels.values()))
    remap = {old: new for new, old in enumerate(unique)}
    return {n: remap[l] for n, l in labels.items()}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _curie_path(ontology: str, term_id: str, root: Path) -> Path:
    safe = term_id.replace(":", "_").replace("/", "__")
    return root / ontology / f"{safe}.json"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _build_graph_from_processed(processed_dir: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Build an undirected graph from the edge table."""
    nodes = [r["id"] for r in pq.read_table(processed_dir / "node.parquet").to_pylist()
             if not r.get("obsolete")]
    edges = pq.read_table(processed_dir / "edge.parquet").to_pylist()
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        if e.get("relation_iri") != "obo:BFO_0000050":
            continue
        s, d = e.get("src_id"), e.get("dst_id")
        if s in adj and d in adj:
            adj[s].add(d)
            adj[d].add(s)
    return nodes, adj


def _write_module_assignments(
    snapshot_id: str, ontology: str, assignments: dict[str, int], resolution: float,
) -> Path:
    MODULES_CACHE.mkdir(parents=True, exist_ok=True)
    out = MODULES_CACHE / snapshot_id / f"labelprop_{resolution:.2f}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"term_id": t, "snapshot_id": snapshot_id, "module_id": int(m),
             "ontology": ontology, "algo": "labelprop", "resolution": float(resolution)}
            for t, m in assignments.items()]
    schema = pa.schema([
        ("term_id", pa.string()),
        ("snapshot_id", pa.string()),
        ("module_id", pa.int32()),
        ("ontology", pa.string()),
        ("algo", pa.string()),
        ("resolution", pa.float32()),
    ])
    if not rows:
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out, compression="snappy")
    return out


def _write_module_jsons(
    snapshot_id: str, ontology: str, assignments: dict[str, int], names: dict[str, str],
) -> list[dict]:
    """Write one JSON per module. Returns a list of summary dicts."""
    V3_MODULES_ROOT.mkdir(parents=True, exist_ok=True)
    by_module: dict[int, list[str]] = defaultdict(list)
    for t, m in assignments.items():
        by_module[int(m)].append(t)
    summaries: list[dict] = []
    for m, terms in sorted(by_module.items()):
        # Centroid = term with the highest lexical centrality proxy: shortest name
        # (i.e. most general). For V3 the "centroid" is informational; the
        # V3 contract doesn't require it to be unique.
        centroid = min(terms, key=lambda t: (len(names.get(t, t)), t))
        mod = {
            "module_id": f"m_{m:04d}",
            "snapshot_id": snapshot_id,
            "ontology": ontology,
            "algo": "labelprop",
            "size": len(terms),
            "terms": sorted(terms),
            "centroid_term": centroid,
            "parent_module": None,        # populated in step 07 (world model)
            "schema_version": SCHEMA_VERSION,
        }
        out = V3_MODULES_ROOT / f"{mod['module_id']}.json"
        out.write_text(json.dumps(mod, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append({"module_id": mod["module_id"], "size": len(terms),
                          "centroid_term": centroid})
    return summaries


# ---------------------------------------------------------------------------
# V3 capsule assembly
# ---------------------------------------------------------------------------

def _build_v3_capsule(
    v2: dict, module_id_by_term: dict[str, str],
) -> dict:
    out = json.loads(json.dumps(v2))
    out["schema_version"] = SCHEMA_VERSION
    out["coordinates"]["semantic_module"] = module_id_by_term.get(v2["id"])
    return out


def _validate_v3(c: dict) -> None:
    required = [
        "id", "name", "ontology", "snapshot_id", "schema_version",
        "identity", "coordinates", "logical_signature", "disambiguation",
        "cross_ontology", "usage", "semantic_synopsis",
        "semantic_synopsis_metadata", "exact_pointers",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        raise RuntimeError(f"V3 capsule missing required fields: {missing}")
    if c["semantic_synopsis"] is not None:
        raise RuntimeError("V3 capsule must have semantic_synopsis=null (rule §5)")
    if c["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(f"V3 schema_version must be {SCHEMA_VERSION!r}")
    # The plan: "every non-root capsule has a semantic_module"
    if c["coordinates"].get("parents") and not c["coordinates"].get("semantic_module"):
        raise RuntimeError(f"V3 non-root {c['id']!r} missing semantic_module")


# ---------------------------------------------------------------------------
# Per-snapshot processing
# ---------------------------------------------------------------------------

def build_for_snapshot(
    snapshot_id: str, ontology: str, processed_dir: Path,
    resolution: float = DEFAULT_RESOLUTION,
    max_iters: int = DEFAULT_MAX_ITERS,
    force: bool = False,
) -> dict:
    """Build V3 capsules + module map for one snapshot."""
    if not V2_ROOT.exists():
        raise RuntimeError(f"no V2 capsules at {V2_ROOT}; run step 05 first")

    # 1) Build the graph from the processed edges
    nodes, adj = _build_graph_from_processed(processed_dir)

    # 2) Run label propagation
    raw_assignments = label_propagation(nodes, adj, max_iters=max_iters)
    # Re-key to 4-digit module ids, leave room for rollup
    mod_id_str = {t: f"m_{m:04d}" for t, m in raw_assignments.items()}

    # 3) Cache the assignments
    assign_path = _write_module_assignments(
        snapshot_id, ontology, raw_assignments, resolution
    )

    # 4) Module JSONs
    # Read names from V2 capsules
    names: dict[str, str] = {}
    src_dir = V2_ROOT / ontology
    if not src_dir.exists():
        raise RuntimeError(f"no V2 capsules for ontology {ontology} at {src_dir}")
    for p in src_dir.glob("*.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        if c.get("snapshot_id") != snapshot_id:
            continue
        names[c["id"]] = c.get("name") or c["id"]
    module_summaries = _write_module_jsons(snapshot_id, ontology, raw_assignments, names)

    # 5) V3 capsules
    written = 0
    for v2_path in sorted(src_dir.glob("*.json")):
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        if v2.get("snapshot_id") != snapshot_id:
            continue
        v3 = _build_v3_capsule(v2, mod_id_str)
        _validate_v3(v3)
        out = _curie_path(ontology, v3["id"], V3_ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(v3, indent=2, sort_keys=True), encoding="utf-8")
        written += 1
    (V3_ROOT.parent / "schema_version.txt").write_text(SCHEMA_VERSION)

    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "v3_written": written,
        "modules": len(module_summaries),
        "module_assignments_path": str(assign_path),
        "module_size_distribution": {
            "min": min(m["size"] for m in module_summaries) if module_summaries else 0,
            "max": max(m["size"] for m in module_summaries) if module_summaries else 0,
            "mean": (sum(m["size"] for m in module_summaries) / max(1, len(module_summaries)))
                    if module_summaries else 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    p.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    p.add_argument("--force", action="store_true")
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
        for d in sorted(Path("data/processed").iterdir()):
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
    for sid, ontology, proc in snaps:
        s = build_for_snapshot(
            sid, ontology, proc,
            resolution=args.resolution,
            max_iters=args.max_iters,
            force=args.force,
        )
        msd = s["module_size_distribution"]
        print(
            f"✅ {sid} ({s['ontology']}): {s['v3_written']} V3 capsules, "
            f"{s['modules']} modules (size min={msd['min']} "
            f"max={msd['max']} mean={msd['mean']:.2f}), "
            f"resolution={args.resolution}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
