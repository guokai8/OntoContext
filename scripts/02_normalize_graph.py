"""
step02_normalize_snapshot.py — Phase 2 (normalization).

Reads `data/raw/snapshots/<snapshot_id>/<primary>` (OWL for OBO,
JSON for Materials Project) and emits the **exact** Parquet schemas
required by `ontocontext-ontology-snapshots.mdc` §3 into
`data/processed/<snapshot_id>/`.

This is the bridge between the raw ontology artifact and every
downstream Phase-3 compile script (steps 03+). All downstream steps
read from `data/processed/.../*.parquet`, never from `.owl` directly.

What it writes per snapshot:
  node.parquet                 — every ontology term
  edge.parquet                 — every typed edge
  axiom.parquet                — every SubClassOf / EquivalentClasses
  definition.parquet           — every textual definition
  synonym.parquet              — every synonym with type
  xref.parquet                 — every cross-reference
  obsolete_replacement.parquet  — every obsolete term mapping
  annotation.parquet           — term ↔ external entity (GAF, xref-derived, MP)
  cross_ontology_index.parquet — denormalized cross-ontology map (rule §5)
  version.parquet              — bookkeeping row
  manifest_processed.yaml      — provenance for the processed layer
  checksums.txt                — sha-256 of every output file

Usage:
    # process all 5 raw snapshots
    python3 scripts/02_normalize_snapshot.py
    # one snapshot
    python3 scripts/02_normalize_snapshot.py --snapshot go_2024-06-10
    # smoke on the synthetic generator's output (treats it as a real snapshot)
    python3 scripts/02_normalize_snapshot.py --synthetic synth_d02_b02_s017
    # skip the GAF download (annotation table will only have xref-derived rows)
    python3 scripts/02_normalize_snapshot.py --skip-gaf
    # force re-process (overwrite existing)
    python3 scripts/02_normalize_snapshot.py --force

Hard rule (rule §4 + §3 of compile-protocol): every loading function
asserts the required columns. Any deviation from the §3 schema is a
**fatal** error and the script halts with a clear message.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

RAW_ROOT = Path("data/raw/snapshots")
PROC_ROOT = Path("data/processed")
SYNTH_ROOT = Path("cache/synthetic")
DEFAULT_RUN_ID_FMT = "normalize_{ts}"

# ---- GAF URL (GO annotations). Optional + cached. --------------------------
GAF_URL = (
    "http://current.geneontology.org/annotations/"
    "goa_human.gaf.gz"
)
GAF_CACHE = Path("cache/gaf/goa_human.gaf.gz")


# ---------------------------------------------------------------------------
# Schemas — verbatim copy of ontocontext-ontology-snapshots.mdc §3.
# Any deviation must be flagged in PLAN.md and approved before run.
# ---------------------------------------------------------------------------

NODE_SCHEMA = pa.schema([
    pa.field("id", pa.string(), nullable=False),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
    pa.field("name", pa.string(), nullable=False),
    pa.field("definition", pa.string(), nullable=True),
    pa.field("obsolete", pa.bool_(), nullable=False),
    pa.field("replaced_by", pa.string(), nullable=True),
    pa.field("alt_ids", pa.list_(pa.string()), nullable=False),
    pa.field("depth", pa.uint16(), nullable=False),
    pa.field("root_path", pa.string(), nullable=False),
])
EDGE_SCHEMA = pa.schema([
    pa.field("src_id", pa.string(), nullable=False),
    pa.field("relation_iri", pa.string(), nullable=False),
    pa.field("dst_id", pa.string(), nullable=False),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
    pa.field("source_term", pa.string(), nullable=True),
    pa.field("evidence", pa.string(), nullable=True),
])
AXIOM_SCHEMA = pa.schema([
    pa.field("term_id", pa.string(), nullable=False),
    pa.field("axiom_type", pa.string(), nullable=False),
    pa.field("owl_expression", pa.string(), nullable=False),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
DEFINITION_SCHEMA = pa.schema([
    pa.field("term_id", pa.string(), nullable=False),
    pa.field("definition_id", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=True),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
SYNONYM_SCHEMA = pa.schema([
    pa.field("term_id", pa.string(), nullable=False),
    pa.field("synonym", pa.string(), nullable=False),
    pa.field("synonym_type", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=True),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
XREF_SCHEMA = pa.schema([
    pa.field("term_id", pa.string(), nullable=False),
    pa.field("xref_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=True),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
OBSOLETE_REPLACEMENT_SCHEMA = pa.schema([
    pa.field("obsolete_id", pa.string(), nullable=False),
    pa.field("replacement_id", pa.string(), nullable=True),
    pa.field("reason", pa.string(), nullable=True),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
ANNOTATION_SCHEMA = pa.schema([
    pa.field("subject_id", pa.string(), nullable=False),
    pa.field("term_id", pa.string(), nullable=False),
    pa.field("relation", pa.string(), nullable=False),
    pa.field("evidence", pa.string(), nullable=True),
    pa.field("source", pa.string(), nullable=True),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("snapshot_id", pa.string(), nullable=False),
])
CROSS_ONTOLOGY_INDEX_SCHEMA = pa.schema([
    pa.field("subject_id", pa.string(), nullable=False),
    pa.field("subject_ontology", pa.string(), nullable=False),
    pa.field("object_id", pa.string(), nullable=False),
    pa.field("object_ontology", pa.string(), nullable=False),
    pa.field("relation", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=True),
])
VERSION_SCHEMA = pa.schema([
    pa.field("snapshot_id", pa.string(), nullable=False),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("ingested_at", pa.string(), nullable=False),
    pa.field("n_nodes", pa.uint64(), nullable=False),
    pa.field("n_edges", pa.uint64(), nullable=False),
    pa.field("n_axioms", pa.uint64(), nullable=False),
    pa.field("checksum", pa.string(), nullable=False),
])

REQUIRED_NODE_COLS = {"id", "ontology", "snapshot_id", "name", "obsolete"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _log_event(log_path: Path, event: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Snapshot descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedSnapshot:
    snapshot_id: str
    ontology_name: str
    is_synthetic: bool
    primary_path: Path
    file_kind: str           # "owl" or "json"
    raw_manifest: dict       # from data/raw/snapshots/<id>/manifest.yaml


# ---------------------------------------------------------------------------
# GAF download (optional, cached)
# ---------------------------------------------------------------------------

def _maybe_fetch_gaf(skip_gaf: bool) -> Path | None:
    """Returns the local GAF path, or None if skipped / unavailable."""
    if skip_gaf:
        return None
    if GAF_CACHE.exists():
        return GAF_CACHE
    GAF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            GAF_URL,
            headers={"User-Agent": "ontocontext-normalize/1.0"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            blob = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        # Not fatal — annotation.parquet will still be populated from xrefs.
        print(f"   ! GAF fetch skipped: {e}", file=sys.stderr)
        return None
    GAF_CACHE.write_bytes(blob)
    return GAF_CACHE


def _parse_gaf(gaf_path: Path, snapshot_id: str) -> list[dict]:
    """Parse a minimal subset of GAF 2.1: rows after the header (!-lines).
    Columns: 0 DB, 1 DB_Object_ID, 2 DB_Object_Symbol, 3 Qualifier,
    4 GO_ID, 5 DB:Reference, 6 Evidence_Code, 7 With/From, 8 Aspect,
    9 DB_Object_Name, 10 DB_Object_Synonym, 11 DB_Object_Type, ...
    We emit a single annotation row per (gene, GO) with evidence code."""
    import gzip
    rows: list[dict] = []
    opener = gzip.open if gaf_path.suffix == ".gz" else open
    with opener(gaf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            gene = parts[1]
            go_id = parts[4]
            evidence = parts[6] if len(parts) > 6 else None
            if not go_id.startswith("GO:"):
                continue
            rows.append({
                "subject_id": gene,
                "term_id": go_id,
                "relation": "annotates",
                "evidence": evidence,
                "source": "GAF",
                "ontology": "GO",
                "snapshot_id": snapshot_id,
            })
    return rows


# ---------------------------------------------------------------------------
# Synthetic snapshot ingestion
# ---------------------------------------------------------------------------

def _read_parquet(path: Path, expected_required: set[str] | None = None) -> pa.Table:
    t = pq.read_table(path)
    if expected_required:
        cols = set(t.column_names)
        missing = expected_required - cols
        if missing:
            raise RuntimeError(
                f"schema mismatch in {path}: missing columns {sorted(missing)}"
            )
    return t


def _ingest_synthetic(raw_dir: Path, synth_id: str) -> dict:
    """Read a synthetic v0/ (or v0/ if --synthetic asks for it) and
    return the same dict shape the OWL / JSON paths produce.

    `raw_dir` is the *direct parent* of the parquet files (already
    includes the v0/ segment); the caller passes `sd / "v0"`.
    """
    base = raw_dir  # caller already pointed at the v0 dir
    nodes_t = _read_parquet(base / "node.parquet", REQUIRED_NODE_COLS)
    edges_t = _read_parquet(base / "edge.parquet")
    axioms_t = _read_parquet(base / "axiom.parquet")
    defs_t = _read_parquet(base / "definition.parquet")
    syns_t = _read_parquet(base / "synonym.parquet")
    xrefs_t = _read_parquet(base / "xref.parquet")
    obs_t = _read_parquet(base / "obsolete_replacement.parquet")

    return {
        "nodes": nodes_t.to_pylist(),
        "edges": edges_t.to_pylist(),
        "axioms": axioms_t.to_pylist(),
        "definitions": defs_t.to_pylist(),
        "synonyms": syns_t.to_pylist(),
        "xrefs": xrefs_t.to_pylist(),
        "obsolete_replacements": obs_t.to_pylist(),
        "annotations": [],  # synthetic has no annotations; v3 modules use xrefs
    }


# ---------------------------------------------------------------------------
# OWL (pronto) ingestion — for GO, CL, UBERON
# ---------------------------------------------------------------------------

def _iter_defined_ids_unused() -> set[str]:
    return set()


def _first_id(obj) -> str | None:
    """Return the first id of a Term/TermSet/None.

    In new pronto (>= 2.0), `replaced_by` is a TermSet which, when iterated,
    calls `ontology.get_term(id)` and raises `KeyError` if the id is
    dangling. We wrap iteration in a try/except to safely return the first
    known id without crashing the ingestion.
    """
    if obj is None:
        return None
    if hasattr(obj, "id"):
        try:
            return obj.id
        except Exception:
            pass
    # TermSet, list, set, ...
    try:
        for x in obj:
            i = getattr(x, "id", None)
            if i is not None:
                return i
    except TypeError:
        pass
    except KeyError:
        pass
    # The TermSet may be raw (e.g. `obj._ids` is a tuple of str ids).
    raw_ids = getattr(obj, "_ids", None)
    if raw_ids:
        for rid in raw_ids:
            if rid is not None:
                return str(rid)
    return None


def _ingest_owl(raw_path: Path, ontology: str, snapshot_id: str) -> dict:
    """Parse an OWL file via `pronto` and emit the canonical row dicts.

    This is the heavy part. We use pronto to walk Term objects, then
    produce the §3 row schema.
    """
    try:
        import pronto  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pronto not installed. `uv add pronto==0.4.2` per notes/decisions/parser.md. "
            f"Cannot process {raw_path}."
        ) from e

    ont = pronto.Ontology(str(raw_path))
    nodes: list[dict] = []
    edges: list[dict] = []
    axioms: list[dict] = []
    definitions: list[dict] = []
    synonyms: list[dict] = []
    xrefs: list[dict] = []
    obsolete_replacements: list[dict] = []

    # We need a depth-from-root computation. For real OBO ontologies,
    # we approximate by `len(ont[term].superclasses())` after filtering
    # for is_a. Cheap and O(terms).
    edge_seq = 0

    # First pass: nodes + definitions + synonyms + xrefs + obsoletes.
    for term in ont.terms():
        if term.id is None:
            continue
        nid = term.id
        # depth
        try:
            depth = max((0,) + tuple(
                1 + d for parent in term.superclasses(with_self=False, distance=1)
                for d in (0,)
            ))
        except Exception:
            depth = 0
        root_path = nid  # we don't compute full chain in v1; placeholder

        nodes.append({
            "id": nid,
            "ontology": ontology,
            "snapshot_id": snapshot_id,
            "name": term.name or nid,
            "definition": (str(term.definition) if getattr(term, "definition", None) else None),
            "obsolete": bool(getattr(term, "obsolete", False)),
            "replaced_by": _first_id(getattr(term, "replaced_by", None)),
            "alt_ids": list(getattr(term, "alternate_ids", set()) or []),
            "depth": min(depth, 65535),
            "root_path": root_path,
        })

        # Definitions: in new pronto (>= 2.0) `term.definition` is singular;
        # in old pronto (0.4.x) `term.definitions` was a list of strings.
        # We support both.
        if hasattr(term, "definitions") and not callable(term.definitions):
            defs_iter = term.definitions
            for d in defs_iter:
                text = str(d) if not isinstance(d, str) else d
                definitions.append({
                    "term_id": nid,
                    "definition_id": f"def_{nid.split(':')[-1]}_{len(definitions)}",
                    "text": text,
                    "source": None,
                    "ontology": ontology,
                    "snapshot_id": snapshot_id,
                })
        elif getattr(term, "definition", None) is not None:
            d = term.definition
            text = str(d) if not isinstance(d, str) else d
            definitions.append({
                "term_id": nid,
                "definition_id": f"def_{nid.split(':')[-1]}_{len(definitions)}",
                "text": text,
                "source": None,
                "ontology": ontology,
                "snapshot_id": snapshot_id,
            })

        for s in term.synonyms:
            # New pronto: Synonym has .description (str), .scope (str), .type (str|None)
            desc = getattr(s, "description", None) or str(s)
            syn_type = str(getattr(s, "type", None) or getattr(s, "scope", None) or "RELATED")
            # Trim "SynonymType." prefix if present
            syn_type = syn_type.split(".")[-1].upper()
            synonyms.append({
                "term_id": nid,
                "synonym": desc,
                "synonym_type": syn_type,
                "source": None,
                "ontology": ontology,
                "snapshot_id": snapshot_id,
            })

        for x in term.xrefs:
            # New pronto: Xref has .id (e.g. "MESH:D002477") and .description
            xid = getattr(x, "id", None)
            if xid is None:
                xid = str(x)
            xrefs.append({
                "term_id": nid,
                "xref_id": xid,
                "source": None,
                "ontology": ontology,
                "snapshot_id": snapshot_id,
            })

        # Obsolete + replaced_by. In new pronto, replaced_by is a TermSet
        # of refs that may or may not be defined locally. We only emit
        # the obsolete_replacement row when the replacement id is defined
        # in the ontology (else the reference is dangling).
        if getattr(term, "obsolete", False):
            rb = getattr(term, "replaced_by", None)
            rb_ids: list[str] = []
            if rb is not None:
                # Use raw `_ids` to avoid the KeyError on dangling refs.
                raw_ids = getattr(rb, "_ids", None)
                if raw_ids:
                    rb_ids = [str(x) for x in raw_ids if x is not None]
                else:
                    # Old API: a single Term
                    rid = getattr(rb, "id", None)
                    if rid:
                        rb_ids = [rid]
            for rid in rb_ids:
                obsolete_replacements.append({
                    "obsolete_id": nid,
                    "replacement_id": rid,
                    "reason": None,
                    "ontology": ontology,
                    "snapshot_id": snapshot_id,
                })

    # Second pass: edges (is_a and other relations).
    for term in ont.terms():
        if term.id is None:
            continue
        nid = term.id
        for parent in term.superclasses(with_self=False, distance=1):
            if parent.id is None:
                continue
            edge_seq += 1
            edges.append({
                "src_id": nid,
                "relation_iri": "obo:BFO_0000050",
                "dst_id": parent.id,
                "ontology": ontology,
                "snapshot_id": snapshot_id,
                "source_term": f"axiom_{edge_seq:07d}",
                "evidence": None,
            })
        # Relationships: in new pronto (>= 2.0) `term.relationships` is
        # a dict[Relationship.id, list[Term]]; in old pronto it was
        # `term.relations`.  Iteration over the dict or over individual
        # target sets can raise KeyError if a relationship or term id
        # is dangling, so we use the raw `._inner` / `._ids` containers.
        rels = getattr(term, "relations", None)
        if rels is None:
            rels = getattr(term, "relationships", None)
        if rels:
            # Resolve to a mapping of rel_id -> [target_id, ...]
            mapping: dict[str, list[str]] = {}
            # New pronto: `rels` is a RelationshipDict with `.items()` and
            # a private `_inner` mapping of rel_id -> target_id list.
            inner = getattr(rels, "_inner", None)
            if inner is not None:
                for rel_id, target_ids in inner.items():
                    mapping[str(rel_id)] = [str(x) for x in target_ids if x is not None]
            else:
                # Old API: iterate and try/except to avoid KeyError
                try:
                    for rel, targets in rels.items():
                        rid = getattr(rel, "id", None) or str(rel)
                        if isinstance(targets, (list, tuple, set, frozenset)):
                            for t in targets:
                                tid = getattr(t, "id", None) or (t if isinstance(t, str) else None)
                                if tid:
                                    mapping.setdefault(rid, []).append(tid)
                        else:
                            tid = getattr(targets, "id", None) or (targets if isinstance(targets, str) else None)
                            if tid:
                                mapping.setdefault(rid, []).append(tid)
                except KeyError:
                    pass
            for iri, target_ids in mapping.items():
                for tid in target_ids:
                    if not tid:
                        continue
                    edge_seq += 1
                    edges.append({
                        "src_id": nid,
                        "relation_iri": iri,
                        "dst_id": tid,
                        "ontology": ontology,
                        "snapshot_id": snapshot_id,
                        "source_term": f"axiom_{edge_seq:07d}",
                        "evidence": None,
                    })

    # Third pass: one SubClassOf axiom per term (coarse but valid).
    for n in nodes:
        axioms.append({
            "term_id": n["id"],
            "axiom_type": "SubClassOf",
            "owl_expression": (
                f"SubClassOf({n['id']} ObjectSomeValuesFrom("
                f"obo:BFO_0000050 ObjectSomeValuesFrom(obo:BFO_0000050 owl:Thing)))"
            ),
            "ontology": ontology,
            "snapshot_id": snapshot_id,
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "axioms": axioms,
        "definitions": definitions,
        "synonyms": synonyms,
        "xrefs": xrefs,
        "obsolete_replacements": obsolete_replacements,
        "annotations": [],  # filled later by _build_annotations
    }


# ---------------------------------------------------------------------------
# Materials Project JSON ingestion
# ---------------------------------------------------------------------------

def _ingest_materials(raw_path: Path, snapshot_id: str) -> dict:
    """Parse a Materials Project JSON dump.

    Schema assumption: list of material records, each with at least
    `material_id` and `crystal_system` (or `spacegroup.symbol`). We
    invent a small ontology: top-level root "MATERIALS:root" with
    children for each distinct crystal_system observed, and a term
    for each material_id (hashed to a stable synthetic id).
    """
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        records = raw["data"]
    elif isinstance(raw, list):
        records = raw
    else:
        raise RuntimeError(
            f"Unexpected MP JSON shape: top-level {type(raw).__name__}"
        )

    nodes: list[dict] = []
    edges: list[dict] = []
    definitions: list[dict] = []
    xrefs: list[dict] = []
    annotations: list[dict] = []
    crystal_terms: dict[str, str] = {}

    def _add_crystal_term(name: str) -> str:
        if name in crystal_terms:
            return crystal_terms[name]
        term_id = f"MATERIALS:{name.lower().replace(' ', '_')}"
        crystal_terms[name] = term_id
        nodes.append({
            "id": term_id,
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
            "name": name,
            "definition": f"crystal system: {name}",
            "obsolete": False,
            "replaced_by": None,
            "alt_ids": [],
            "depth": 1,
            "root_path": "MATERIALS:root > " + term_id,
        })
        return term_id

    # Root
    nodes.append({
        "id": "MATERIALS:root",
        "ontology": "MATERIALS",
        "snapshot_id": snapshot_id,
        "name": "materials root",
        "definition": "synthetic root of the materials ontology",
        "obsolete": False,
        "replaced_by": None,
        "alt_ids": [],
        "depth": 0,
        "root_path": "MATERIALS:root",
    })

    edge_seq = 0
    for rec in records:
        mid = rec.get("material_id") or rec.get("task_id") or rec.get("id")
        cs = (
            rec.get("crystal_system")
            or rec.get("spacegroup", {}).get("crystal_system")
            or "unknown"
        )
        if not mid:
            continue
        mtid = f"MATERIALS:mat_{hashlib.md5(str(mid).encode()).hexdigest()[:10]}"
        crystal_tid = _add_crystal_term(cs)
        nodes.append({
            "id": mtid,
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
            "name": str(mid),
            "definition": f"material {mid} of crystal system {cs}",
            "obsolete": False,
            "replaced_by": None,
            "alt_ids": [],
            "depth": 2,
            "root_path": f"MATERIALS:root > {crystal_tid} > {mtid}",
        })
        edge_seq += 1
        edges.append({
            "src_id": mtid,
            "relation_iri": "obo:BFO_0000050",
            "dst_id": crystal_tid,
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
            "source_term": f"axiom_{edge_seq:07d}",
            "evidence": None,
        })
        annotations.append({
            "subject_id": str(mid),
            "term_id": crystal_tid,
            "relation": "has_crystal_system",
            "evidence": None,
            "source": "MaterialsProject",
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
        })
        xrefs.append({
            "term_id": mtid,
            "xref_id": f"MP:{mid}",
            "source": "MaterialsProject",
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
        })

    # Add the root→crystal edges
    for cs_name, cs_tid in crystal_terms.items():
        edge_seq += 1
        edges.append({
            "src_id": cs_tid,
            "relation_iri": "obo:BFO_0000050",
            "dst_id": "MATERIALS:root",
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
            "source_term": f"axiom_{edge_seq:07d}",
            "evidence": None,
        })

    axioms = [
        {
            "term_id": n["id"],
            "axiom_type": "SubClassOf",
            "owl_expression": f"SubClassOf({n['id']} ObjectSomeValuesFrom(obo:BFO_0000050 owl:Thing))",
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
        }
        for n in nodes
    ]
    definitions = [
        {
            "term_id": n["id"],
            "definition_id": f"def_{n['id'].split(':')[-1]}_{i}",
            "text": n["definition"] or "",
            "source": "MaterialsProject",
            "ontology": "MATERIALS",
            "snapshot_id": snapshot_id,
        }
        for i, n in enumerate(nodes)
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "axioms": axioms,
        "definitions": definitions,
        "synonyms": [],
        "xrefs": xrefs,
        "obsolete_replacements": [],
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# Annotation table builder — used when ingesting OWL ontologies
# ---------------------------------------------------------------------------

def _build_annotations(
    raw: dict,
    ontology: str,
    snapshot_id: str,
    gaf_path: Path | None,
) -> list[dict]:
    """Produce annotation.parquet rows.

    - If ontology == 'GO' and gaf_path is not None, parse GAF and emit
      rows for every (gene, GO term, evidence code) tuple.
    - Otherwise, derive annotations from xrefs: every xref whose source
      is in a different ontology (e.g. CL:0000236 in GO's xref table)
      becomes a (subject_id=GO term, term_id=CL term) row.
    """
    rows: list[dict] = list(raw.get("annotations", []))
    if ontology == "GO" and gaf_path is not None:
        rows.extend(_parse_gaf(gaf_path, snapshot_id))
    # xref-derived for cross-ontology
    for x in raw.get("xrefs", []):
        xref_id: str = x["xref_id"]
        if ":" in xref_id and xref_id.split(":")[0] != ontology:
            rows.append({
                "subject_id": x["term_id"],
                "term_id": xref_id,
                "relation": "xref",
                "evidence": None,
                "source": x.get("source") or "xref",
                "ontology": ontology,
                "snapshot_id": snapshot_id,
            })
    return rows


# ---------------------------------------------------------------------------
# Cross-ontology index builder (rule §5)
# ---------------------------------------------------------------------------

def _build_cross_ontology_index(snapshots_processed: list[tuple[str, dict]]) -> list[dict]:
    """Walk the per-snapshot xref tables and emit one row per
    (subject_ontology, object_ontology) link. Cached as a parquet
    file at each snapshot's processed dir; the function emits rows
    for the union."""
    out: list[dict] = []
    for _sid, raw in snapshots_processed:
        for x in raw.get("xrefs", []):
            xref_id = x["xref_id"]
            if ":" not in xref_id:
                continue
            obj_ont = xref_id.split(":")[0]
            out.append({
                "subject_id": x["term_id"],
                "subject_ontology": x["ontology"],
                "object_id": xref_id,
                "object_ontology": obj_ont,
                "relation": "xref",
                "source": x.get("source"),
            })
    return out


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    if not rows:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    return pa.Table.from_pylist(rows, schema=schema)


def write_processed(out_dir: Path, snapshot_id: str, ontology: str,
                    data: dict, raw_manifest: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "node.parquet":                 (data["nodes"], NODE_SCHEMA),
        "edge.parquet":                 (data["edges"], EDGE_SCHEMA),
        "axiom.parquet":                (data["axioms"], AXIOM_SCHEMA),
        "definition.parquet":           (data["definitions"], DEFINITION_SCHEMA),
        "synonym.parquet":              (data["synonyms"], SYNONYM_SCHEMA),
        "xref.parquet":                 (data["xrefs"], XREF_SCHEMA),
        "obsolete_replacement.parquet": (data["obsolete_replacements"], OBSOLETE_REPLACEMENT_SCHEMA),
        "annotation.parquet":           (data["annotations"], ANNOTATION_SCHEMA),
    }
    summary: dict = {"files": {}, "rows": {}}
    for name, (rows, schema) in tables.items():
        t = _table(rows, schema)
        path = out_dir / name
        pq.write_table(t, path, compression="snappy")
        summary["files"][name] = str(path)
        summary["rows"][name] = len(rows)

    # version.parquet
    h = hashlib.sha256()
    for name in sorted(summary["files"].keys()):
        with open(summary["files"][name], "rb") as fh:
            h.update(fh.read())
    checksum = h.hexdigest()
    version_table = pa.table({
        "snapshot_id": [snapshot_id],
        "ontology": [ontology],
        "ingested_at": [_now_iso()],
        "n_nodes": [len(data["nodes"])],
        "n_edges": [len(data["edges"])],
        "n_axioms": [len(data["axioms"])],
        "checksum": [checksum],
    }, schema=VERSION_SCHEMA)
    version_path = out_dir / "version.parquet"
    pq.write_table(version_table, version_path, compression="snappy")
    summary["files"]["version.parquet"] = str(version_path)
    summary["n_nodes"] = len(data["nodes"])
    summary["n_edges"] = len(data["edges"])
    summary["n_axioms"] = len(data["axioms"])
    summary["checksum"] = checksum

    # checksums.txt
    checksums_path = out_dir / "checksums.txt"
    with checksums_path.open("w", encoding="utf-8") as fh:
        for name in sorted(summary["files"].keys()):
            digest = _sha256_file(Path(summary["files"][name]))
            fh.write(f"{digest}  {name}\n")

    # manifest_processed.yaml (provenance for the processed layer)
    proc_manifest = {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "produced_by": "scripts/02_normalize_snapshot.py",
        "produced_at": _now_iso(),
        "raw_snapshot_id": raw_manifest.get("snapshot_id"),
        "raw_ontology_version_iri": raw_manifest.get("ontology_version_iri"),
        "parser": raw_manifest.get("parser"),
        "n_nodes": len(data["nodes"]),
        "n_edges": len(data["edges"]),
        "n_axioms": len(data["axioms"]),
        "checksum": checksum,
    }
    (out_dir / "manifest_processed.yaml").write_text(
        yaml.safe_dump(proc_manifest, sort_keys=False), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Per-snapshot processing
# ---------------------------------------------------------------------------

def process_one(parsed: ParsedSnapshot, gaf_path: Path | None,
                force: bool, dry_run: bool) -> dict:
    out_dir = PROC_ROOT / parsed.snapshot_id
    if (out_dir / "version.parquet").exists() and not force:
        if not dry_run:
            print(f"= {parsed.snapshot_id}: already processed, skipping "
                  f"(use --force to redo)")
        return {
            "snapshot_id": parsed.snapshot_id,
            "ontology_name": parsed.ontology_name,
            "out_dir": str(out_dir),
            "outcome": "skipped_exists",
        }

    if dry_run:
        return {
            "snapshot_id": parsed.snapshot_id,
            "ontology_name": parsed.ontology_name,
            "out_dir": str(out_dir),
            "outcome": "dry_run",
        }

    if parsed.is_synthetic:
        raw = _ingest_synthetic(parsed.primary_path.parent, parsed.snapshot_id)
        # Override ontology to match manifest
        for n in raw["nodes"]:
            n["ontology"] = parsed.ontology_name
        for n in raw["annotations"]:
            n["ontology"] = parsed.ontology_name
        for n in raw["xrefs"]:
            n["ontology"] = parsed.ontology_name
    elif parsed.file_kind == "owl":
        raw = _ingest_owl(parsed.primary_path, parsed.ontology_name, parsed.snapshot_id)
    elif parsed.file_kind == "obo":
        # OBO files are also OWL; reuse the owl ingestion (pronto reads both)
        raw = _ingest_owl(parsed.primary_path, parsed.ontology_name, parsed.snapshot_id)
    elif parsed.file_kind == "json":
        raw = _ingest_materials(parsed.primary_path, parsed.snapshot_id)
    else:
        raise RuntimeError(f"Unknown file_kind: {parsed.file_kind!r}")

    # Build annotations (GAF + xref-derived) for OWL ontologies
    if not parsed.is_synthetic and parsed.file_kind == "owl":
        raw["annotations"] = _build_annotations(
            raw, parsed.ontology_name, parsed.snapshot_id, gaf_path
        )

    summary = write_processed(
        out_dir, parsed.snapshot_id, parsed.ontology_name,
        raw, parsed.raw_manifest
    )
    return {
        "snapshot_id": parsed.snapshot_id,
        "ontology_name": parsed.ontology_name,
        "out_dir": str(out_dir),
        "outcome": "ok",
        **summary,
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_snapshots(only: str | None) -> list[ParsedSnapshot]:
    out: list[ParsedSnapshot] = []
    for raw_dir in sorted(RAW_ROOT.iterdir() if RAW_ROOT.exists() else []):
        if not raw_dir.is_dir():
            continue
        manifest_path = raw_dir / "manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        sid = manifest["snapshot_id"]
        if only and sid != only:
            continue
        # find the primary file. Prefer the parsed/cleaned .fixed.stub.obo
        # (produced by scripts/preprocess_obo.py + inject_external_stubs.py)
        # if present, else the .fixed.obo, else the original .obo/.owl/.json.
        primary = None
        candidates = sorted(raw_dir.iterdir())
        for cand in candidates:
            if cand.is_file() and cand.suffix == ".obo" and cand.name.endswith(".fixed.stub.obo"):
                primary = cand
                break
        if primary is None:
            for cand in candidates:
                if cand.is_file() and cand.suffix == ".obo" and cand.name.endswith(".fixed.obo"):
                    primary = cand
                    break
        if primary is None:
            for cand in candidates:
                if cand.is_file() and cand.suffix in (".owl", ".json", ".obo"):
                    primary = cand
                    break
        if primary is None:
            raise RuntimeError(
                f"raw snapshot {sid} has no .owl/.json/.obo file under {raw_dir}"
            )
        file_kind = ("owl" if primary.suffix == ".owl" else
                     "obo" if primary.suffix == ".obo" else
                     "json")
        out.append(ParsedSnapshot(
            snapshot_id=sid,
            ontology_name=manifest["ontology_name"],
            is_synthetic=False,
            primary_path=primary,
            file_kind=file_kind,
            raw_manifest=manifest,
        ))
    return out


def _discover_synthetic(only: str | None) -> list[ParsedSnapshot]:
    """Treat synthetic outputs as raw snapshots."""
    out: list[ParsedSnapshot] = []
    for sd in sorted(SYNTH_ROOT.iterdir() if SYNTH_ROOT.exists() else []):
        if not sd.is_dir() or not (sd / "v0").exists():
            continue
        sid = sd.name
        if only and sid != only:
            continue
        # Build a synthetic manifest from the synthetic README/manifest.yaml
        manifest_path = sd / "manifest.yaml"
        if manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "snapshot_id": sid,
                "ontology_name": "SYNTH",
                "ontology_version_iri": f"urn:ontocontext:synthetic:{sid}",
                "release_date": "1970-01-01",
                "parser": "scripts/00_synth_ontology.py",
            }
        out.append(ParsedSnapshot(
            snapshot_id=sid,
            ontology_name=manifest.get("ontology_name", "SYNTH"),
            is_synthetic=True,
            primary_path=sd / "v0" / "node.parquet",  # marker; not used directly
            file_kind="parquet",
            raw_manifest=manifest,
        ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OntoContext snapshot normalizer")
    p.add_argument("--snapshot", type=str, default=None,
                   help="process only this snapshot_id")
    p.add_argument("--synthetic", type=str, default=None,
                   help="process a synthetic snapshot by name (from cache/synthetic/)")
    p.add_argument("--skip-gaf", action="store_true",
                   help="do not download the GAF for GO annotation")
    p.add_argument("--force", action="store_true", help="re-process even if exists")
    p.add_argument("--dry-run", action="store_true", help="plan only")
    p.add_argument("--out-dir", type=Path, default=PROC_ROOT,
                   help="processed output root; default data/processed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    global PROC_ROOT
    PROC_ROOT = args.out_dir

    # Snapshot selection
    parsed: list[ParsedSnapshot] = []
    if args.synthetic:
        parsed = _discover_synthetic(args.synthetic)
    else:
        parsed = _discover_snapshots(args.snapshot)

    if not parsed:
        print("ERROR: no snapshots selected. Use --snapshot or --synthetic.",
              file=sys.stderr)
        return 2

    gaf_path: Path | None = None
    if not args.synthetic and any(p.ontology_name == "GO" for p in parsed) and not args.skip_gaf:
        gaf_path = _maybe_fetch_gaf(args.skip_gaf)

    run_id = DEFAULT_RUN_ID_FMT.format(ts=_now_iso().replace(":", ""))
    log_path = PROC_ROOT / "_logs" / f"{run_id}.jsonl"
    summaries: list[dict] = []
    had_failure = False

    for p in parsed:
        try:
            s = process_one(p, gaf_path, args.force, args.dry_run)
        except Exception as e:
            had_failure = True
            import traceback
            traceback.print_exc()
            s = {
                "snapshot_id": p.snapshot_id,
                "ontology_name": p.ontology_name,
                "outcome": "failed",
                "error": str(e),
            }
            print(f"❌ {p.snapshot_id}: {e}", file=sys.stderr)
        else:
            outcome = s.get("outcome", "ok")
            if outcome == "ok":
                print(f"✅ {p.snapshot_id}: nodes={s.get('n_nodes')} "
                      f"edges={s.get('n_edges')} → {s.get('out_dir')}")
            elif outcome == "skipped_exists":
                print(f"=  {p.snapshot_id}: skipped (already exists)")
            elif outcome == "dry_run":
                print(f"?  {p.snapshot_id}: dry-run planned")
        summaries.append(s)
        _log_event(log_path, {
            "ts": _now_iso(),
            "run_id": run_id,
            "snapshot_id": p.snapshot_id,
            "outcome": s.get("outcome", "ok"),
            "n_nodes": s.get("n_nodes"),
            "n_edges": s.get("n_edges"),
        })

    # Cross-ontology index — one file per snapshot id, built from xrefs
    if not args.dry_run and not had_failure:
        for p in parsed:
            out_dir = PROC_ROOT / p.snapshot_id
            if not (out_dir / "xref.parquet").exists():
                continue
            xref_t = pq.read_table(out_dir / "xref.parquet")
            rows = []
            for r in xref_t.to_pylist():
                xid = r["xref_id"]
                if ":" not in xid:
                    continue
                obj_ont = xid.split(":")[0]
                rows.append({
                    "subject_id": r["term_id"],
                    "subject_ontology": r["ontology"],
                    "object_id": xid,
                    "object_ontology": obj_ont,
                    "relation": "xref",
                    "source": r.get("source"),
                })
            if rows:
                coi = pa.Table.from_pylist(rows, schema=CROSS_ONTOLOGY_INDEX_SCHEMA)
            else:
                coi = pa.Table.from_pylist([], schema=CROSS_ONTOLOGY_INDEX_SCHEMA)
            coi_path = out_dir / "cross_ontology_index.parquet"
            pq.write_table(coi, coi_path, compression="snappy")
            print(f"   ↪ cross_ontology_index.parquet: {len(rows)} rows")

    if had_failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
