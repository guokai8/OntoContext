"""
step03_v0_build.py — V0 deterministic capsule builder.

Reads the §3 Parquet tables produced by step 02 and writes one
**V0** capsule per non-obsolete ontology term, in JSON form, under
`capsules/v0/entities/<ontology>/<safe_id>.json`. `safe_id` is the
CURIE-encoded term id (rule §3.2).

V0 = "deterministic" — the capsule carries the *unchangeable* part of
the term's identity: the identity block, the axioms attached to the
term, the cross-ontology relations, and the exact_pointers. It is the
ground truth that V1 (information-theoretic) and V2 (discriminative)
extend.

Per PLAN.md step 03:
  - Inputs:  data/processed/<snapshot_id>/{node,edge,axiom,definition,
            synonym,xref,annotation}.parquet
            scripts/_config.py
  - Outputs: capsules/v0/entities/<ontology>/<id>.json
            capsules/v0/schema_version.txt = 0.1.0
  - Verification: every capsule passes jsonschema.validate(capsule, schema);
            capsule count = number of non-obsolete terms; smoke run on
            cache/synthetic/synth_seed_17/v0/ produces ≥ 1 capsule in < 30 s.

Usage:
    python3 scripts/step03_v0_build.py --snapshot go_2024-06-10
    python3 scripts/step03_v0_build.py --synthetic synth_d02_b02_s017
    python3 scripts/step03_v0_build.py --all
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

CAPSULES_ROOT = Path("capsules/v0/entities")
SCHEMA_VERSION = "0.1.0"
DEFAULT_RUN_ID_FMT = "v0_{ts}"


# Capsule JSON-Schema (rule §3.3). Kept inline so the script is
# self-contained; matches the locked schema_version 0.1.0.

CAPSULE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OntoContext capsule v0.1.0",
    "type": "object",
    "required": [
        "id", "name", "ontology", "snapshot_id", "schema_version",
        "identity", "coordinates", "logical_signature",
        "disambiguation", "cross_ontology", "usage",
        "semantic_synopsis", "semantic_synopsis_metadata",
        "exact_pointers",
    ],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "ontology": {"type": "string"},
        "snapshot_id": {"type": "string"},
        "schema_version": {"type": "string"},
        "identity": {
            "type": "object",
            "required": ["alt_ids", "definition", "definition_xrefs",
                         "synonyms", "xrefs"],
        },
        "coordinates": {
            "type": "object",
            "required": ["parents", "selected_ancestors",
                         "selected_children", "semantic_module"],
        },
        "logical_signature": {
            "type": "object",
            "required": ["defining_relations", "necessary_conditions",
                         "selected_axioms"],
        },
        "disambiguation": {
            "type": "object",
            "required": ["siblings", "near_terms",
                         "distinguishing_features"],
        },
        "cross_ontology": {
            "type": "object",
            "required": ["relations"],
        },
        "usage": {
            "type": "object",
            "required": ["annotation_frequency", "information_content"],
        },
        "semantic_synopsis": {"type": ["string", "null"]},
        "semantic_synopsis_metadata": {"type": ["object", "null"]},
        "exact_pointers": {
            "type": "object",
            "required": ["source_term", "license", "checksum"],
        },
    },
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _curie_path(ontology: str, term_id: str, root: Path) -> Path:
    """CURIE-encode the term id for filesystem safety (rule §3.2)."""
    safe = term_id.replace(":", "_").replace("/", "__")
    return root / ontology / f"{safe}.json"


def _read_parquet(path: Path) -> dict:
    """Return a list-of-dicts view of a parquet file."""
    return pq.read_table(path).to_pylist()


def _validate_capsule(c: dict) -> None:
    """Validate against CAPSULE_SCHEMA using jsonschema if available,
    else fall back to the §3.3 required-fields check (rule §3.3)."""
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(c, CAPSULE_SCHEMA)
    except ImportError:
        required = set(CAPSULE_SCHEMA["required"])
        missing = required - set(c.keys())
        if missing:
            raise RuntimeError(f"capsule missing required fields: {sorted(missing)}")
        # Coarse type check on the four top-level strings
        for k in ("id", "name", "ontology", "snapshot_id", "schema_version"):
            if not isinstance(c.get(k), str):
                raise RuntimeError(f"capsule field {k!r} must be a string")


def _build_capsule(
    node: dict,
    edges: list[dict],
    axioms: list[dict],
    defs: list[dict],
    syns: list[dict],
    xrefs: list[dict],
    annotations: list[dict],
    snapshot_id: str,
) -> dict:
    """Assemble one V0 capsule from the per-table rows for this term."""
    nid = node["id"]
    out_edges = [e for e in edges if e.get("src_id") == nid or e.get("dst_id") == nid]
    # In an is_a edge (src is_a dst), `src` is the child and `dst` is the parent.
    # So if nid is the source of an edge, the destination is a parent.
    #    And if nid is the destination of an edge, the source is a child.
    parents = [e["dst_id"] for e in out_edges
               if e.get("src_id") == nid and e.get("relation_iri") == "obo:BFO_0000050"]
    children = [e["src_id"] for e in out_edges
                if e.get("dst_id") == nid and e.get("relation_iri") == "obo:BFO_0000050"]

    defining_relations = sorted({e.get("relation_iri", "?")
                                  for e in out_edges if e.get("relation_iri")})
    selected_axioms = [a.get("owl_expression", "")
                       for a in axioms if a.get("term_id") == nid][:8]
    defn_text = (defs[0]["text"] if defs and (defs[0].get("text") or "").strip()
                 else node.get("definition") or "")
    syn_list = [{"text": s["synonym"], "type": s.get("synonym_type", "RELATED")}
                for s in syns if s.get("term_id") == nid]
    xref_list = [{"id": x["xref_id"], "source": x.get("source") or ""}
                 for x in xrefs if x.get("term_id") == nid]
    cross_rels = [{"to": x["xref_id"], "relation": "xref",
                   "source": x.get("ontology") or ""}
                  for x in xrefs if x.get("term_id") == nid and ":" in x["xref_id"]
                  and x["xref_id"].split(":")[0] != node.get("ontology")]
    ann_count = sum(1 for a in annotations if a.get("term_id") == nid)

    return {
        "id": nid,
        "name": node.get("name") or nid,
        "ontology": node.get("ontology") or "",
        "snapshot_id": snapshot_id,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "alt_ids": node.get("alt_ids") or [],
            "definition": defn_text,
            "definition_xrefs": [],
            "synonyms": syn_list,
            "xrefs": xref_list,
        },
        "coordinates": {
            "parents": parents,
            "selected_ancestors": parents[:5],
            "selected_children": children[:8],
            "semantic_module": None,  # populated in step 06 (V3)
        },
        "logical_signature": {
            "defining_relations": defining_relations,
            "necessary_conditions": [],
            "selected_axioms": selected_axioms,
        },
        "disambiguation": {
            "siblings": [],          # populated in step 05 (V2)
            "near_terms": [],
            "distinguishing_features": [],
        },
        "cross_ontology": {
            "relations": cross_rels,
        },
        "usage": {
            "annotation_frequency": ann_count,
            "information_content": 0.0,   # filled in step 04 (V1)
        },
        "semantic_synopsis": None,         # rule §5: only filled via pipeline
        "semantic_synopsis_metadata": None,
        "exact_pointers": {
            "source_term": (
                f"http://purl.obolibrary.org/obo/{nid.replace(':', '_')}"
                if node.get("ontology") in {"GO", "CL", "UBERON"} else nid
            ),
            "source_axioms": [a.get("owl_expression", "")
                              for a in axioms if a.get("term_id") == nid][:4],
            "license": "CC-BY-4.0" if node.get("ontology") in {"GO", "CL", "UBERON"} else "see-manifest",
            "checksum": "sha256:" + hashlib.sha256(
                json.dumps(node, sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
    }


def build_for_snapshot(snapshot_id: str, ontology: str, processed_dir: Path) -> dict:
    """Build V0 capsules for one snapshot. Returns a summary dict."""
    node_t = pq.read_table(processed_dir / "node.parquet")
    edge_t = pq.read_table(processed_dir / "edge.parquet")
    axiom_t = pq.read_table(processed_dir / "axiom.parquet")
    def_t = pq.read_table(processed_dir / "definition.parquet")
    syn_t = pq.read_table(processed_dir / "synonym.parquet")
    xref_t = pq.read_table(processed_dir / "xref.parquet")
    ann_path = processed_dir / "annotation.parquet"
    if ann_path.exists():
        ann_t = pq.read_table(ann_path)
        annotations = ann_t.to_pylist()
    else:
        annotations = []

    edges = edge_t.to_pylist()
    axioms = axiom_t.to_pylist()
    defs = def_t.to_pylist()
    syns = syn_t.to_pylist()
    xrefs = xref_t.to_pylist()

    written = 0
    skipped_obsolete = 0
    for node in node_t.to_pylist():
        if node.get("obsolete"):
            skipped_obsolete += 1
            continue
        cap = _build_capsule(
            node, edges, axioms, defs, syns, xrefs, annotations, snapshot_id
        )
        _validate_capsule(cap)
        out = _curie_path(ontology, node["id"], CAPSULES_ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cap, indent=2, sort_keys=True), encoding="utf-8")
        written += 1

    (CAPSULES_ROOT.parent / "schema_version.txt").write_text(SCHEMA_VERSION)
    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "capsules_written": written,
        "skipped_obsolete": skipped_obsolete,
    }


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--all", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        proc = Path("data/processed") / args.synthetic
        if not proc.exists():
            proc = Path("cache/synthetic") / args.synthetic / "v0"
        s = build_for_snapshot(args.synthetic, "SYNTH", proc)
        print(f"✅ {args.synthetic}: {s['capsules_written']} V0 capsules")
        return 0
    if args.all:
        # Discover processed snapshots
        proc_root = Path("data/processed")
        if not proc_root.exists():
            print("ERROR: no data/processed/ snapshots; run step 02 first.",
                  file=sys.stderr)
            return 2
        total = 0
        for d in sorted(proc_root.iterdir()):
            if not (d / "node.parquet").exists():
                continue
            # Pick ontology from the manifest if available
            mf = d / "manifest_processed.yaml"
            ontology = (yaml.safe_load(mf.read_text())["ontology"]
                        if mf.exists() else d.name.split("_")[0].upper())
            s = build_for_snapshot(d.name, ontology, d)
            print(f"✅ {d.name}: {s['capsules_written']} V0 capsules")
            total += s["capsules_written"]
        print(f"Total: {total}")
        return 0
    if args.snapshot:
        proc = Path("data/processed") / args.snapshot
        if not proc.exists():
            print(f"ERROR: {proc} not found; run step 02 first.", file=sys.stderr)
            return 2
        mf = proc / "manifest_processed.yaml"
        ontology = (yaml.safe_load(mf.read_text())["ontology"]
                    if mf.exists() else args.snapshot.split("_")[0].upper())
        s = build_for_snapshot(args.snapshot, ontology, proc)
        print(f"✅ {args.snapshot}: {s['capsules_written']} V0 capsules")
        return 0
    print("ERROR: pass --snapshot, --synthetic, or --all", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
