"""
build_synth_rich.py — Build a richer synthetic ontology.

A `synth_d05_b10_s042` ontology with:
    - 5 depth levels
    - 3 branching per level
    - 1+3+9+27+9=49 nodes (1 root + 3 + 9 + 27 + 9 deeper)
    - shared parents (siblings) — needed for sibling-discrimination
    - lexical surface variation (synonyms)
    - axioms and definitions

Output:
    /tmp/mini_synth/synth_d05_b10_s042/v0/{node,edge,axiom,definition,synonym,xref,obsolete_replacement}.parquet

Usage:
    python3 scripts/build_synth_rich.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

random.seed(42)


def _new_id(prefix: str, n: int) -> str:
    return f"SYNTH:{prefix}{n:09d}"


def build() -> dict[str, pa.Table]:
    """Build a 5-level synth with 49 nodes."""
    snapshot_id = "synth_d05_b10_s042"
    ontology = "SYNTH"
    nodes: list[dict] = []
    edges: list[dict] = []
    axioms: list[dict] = []
    definitions: list[dict] = []
    synonyms: list[dict] = []
    xrefs: list[dict] = []
    obsolete_replacements: list[dict] = []

    def add_node(nid: str, name: str, definition: str, depth: int, root: str) -> None:
        nodes.append({
            "id": nid, "ontology": ontology, "snapshot_id": snapshot_id,
            "name": name, "definition": definition, "obsolete": False,
            "replaced_by": None, "alt_ids": [], "depth": depth, "root_path": root,
        })
        definitions.append({"term_id": nid, "definition": definition,
                             "ontology": ontology, "snapshot_id": snapshot_id})

    def add_edge(src: str, dst: str, rel: str = "obo:BFO_0000050") -> None:
        edges.append({"src_id": src, "dst_id": dst, "relation_iri": rel,
                       "ontology": ontology, "snapshot_id": snapshot_id})

    # 1 root at level 0
    root = _new_id("R", 0)
    add_node(root, "synthetic_root", "The synthetic root of the test ontology.",
              0, root)
    axioms.append({"subject": root, "predicate": "rdfs:subClassOf",
                    "object": "owl:Thing",
                    "ontology": ontology, "snapshot_id": snapshot_id})

    # Level 1: 3 domains under the root
    l1 = []
    for i in range(3):
        tid = _new_id("R", i + 1)
        l1.append(tid)
        add_node(tid, f"domain_{i}",
                  f"Top-level domain {i} of the synthetic ontology.",
                  1, root)
        add_edge(tid, root)
        axioms.append({"subject": tid, "predicate": "rdfs:subClassOf",
                        "object": root, "ontology": ontology, "snapshot_id": snapshot_id})

    # Level 2: 3 children of each level-1 node (9 total)
    l2 = []
    for parent in l1:
        for j in range(3):
            tid = _new_id("N", len(l2) + 1)
            l2.append(tid)
            add_node(tid, f"subdomain_{parent[-3:]}_{j}",
                      f"A subdomain of branch {j}.",
                      2, root)
            add_edge(tid, parent)
            # Add a synonym for some
            if j == 0 and random.random() < 0.6:
                synonyms.append({"term_id": tid, "text": f"alt_{tid[-3:]}",
                                  "synonym_type": "RELATED",
                                  "ontology": ontology, "snapshot_id": snapshot_id})
            axioms.append({"subject": tid, "predicate": "rdfs:subClassOf",
                            "object": parent, "ontology": ontology, "snapshot_id": snapshot_id})

    # Level 3: 3 children of each level-2 node (27 total)
    l3 = []
    for parent in l2:
        for k in range(3):
            tid = _new_id("N", 10 + len(l3))
            l3.append(tid)
            add_node(tid, f"subsubdomain_{parent[-3:]}_{k}",
                      f"A sub-subdomain, branch {k}.",
                      3, root)
            add_edge(tid, parent)
            axioms.append({"subject": tid, "predicate": "rdfs:subClassOf",
                            "object": parent, "ontology": ontology, "snapshot_id": snapshot_id})

    # Level 4: 1 child of each level-3 node (27 leaves)
    leaves = []
    for parent in l3:
        for m in range(1):
            tid = _new_id("N", 100 + len(leaves))
            leaves.append(tid)
            add_node(tid, f"leaf_{parent[-3:]}_{m}",
                      f"A leaf term in the synthetic ontology.",
                      4, root)
            add_edge(tid, parent)
            # Add a synonym for some leaves
            if m == 0 and random.random() < 0.5:
                synonyms.append({"term_id": tid, "text": f"alt_{tid[-3:]}",
                                  "synonym_type": "RELATED",
                                  "ontology": ontology, "snapshot_id": snapshot_id})
            axioms.append({"subject": tid, "predicate": "rdfs:subClassOf",
                            "object": parent, "ontology": ontology, "snapshot_id": snapshot_id})

    tables: dict[str, pa.Table] = {
        "node": pa.Table.from_pylist(nodes, schema=pa.schema([
            ("id", pa.string()), ("ontology", pa.string()),
            ("snapshot_id", pa.string()), ("name", pa.string()),
            ("definition", pa.string()), ("obsolete", pa.bool_()),
            ("replaced_by", pa.string()), ("alt_ids", pa.list_(pa.string())),
            ("depth", pa.uint16()), ("root_path", pa.string()),
        ])),
        "edge": pa.Table.from_pylist(edges),
        "axiom": pa.Table.from_pylist(axioms),
        "definition": pa.Table.from_pylist(definitions),
        "synonym": pa.Table.from_pylist(synonyms),
        "xref": pa.Table.from_pylist(xrefs, schema=pa.schema([
            ("subject", pa.string()), ("object", pa.string()),
            ("object_ontology", pa.string()), ("predicate", pa.string()),
            ("ontology", pa.string()), ("snapshot_id", pa.string()),
        ])),
        "obsolete_replacement": pa.Table.from_pylist(obsolete_replacements, schema=pa.schema([
            ("term_id", pa.string()), ("replaced_by", pa.string()),
            ("ontology", pa.string()), ("snapshot_id", pa.string()),
        ])),
        "annotation": pa.Table.from_pylist([], schema=pa.schema([
            ("term_id", pa.string()), ("property", pa.string()),
            ("value", pa.string()), ("ontology", pa.string()),
            ("snapshot_id", pa.string()),
        ])),
        "cross_ontology_index": pa.Table.from_pylist([], schema=pa.schema([
            ("subject_id", pa.string()), ("object_id", pa.string()),
            ("relation", pa.string()), ("snapshot_id", pa.string()),
        ])),
        "version": pa.Table.from_pylist([{
            "snapshot_id": snapshot_id, "ontology": ontology,
            "version_iri": f"http://purl.obolibrary.org/ontology/SYNTH/synth_d05_b10_s042",
            "release_date": "2026-10-09", "license": "CC-BY-4.0",
            "checksum": "rich_synth_seed",
        }]),
    }
    return tables


def main():
    out = Path("/tmp/mini_synth/synth_d05_b10_s042/v0")
    out.mkdir(parents=True, exist_ok=True)
    # Symlink so the project picks it up
    proj_link = Path("cache/synthetic/synth_d05_b10_s042")
    proj_link.parent.mkdir(parents=True, exist_ok=True)
    if not proj_link.exists():
        proj_link.symlink_to(out.parent)
    tables = build()
    for name, t in tables.items():
        pq.write_table(t, out / f"{name}.parquet")
    n_nodes = tables["node"].num_rows
    n_edges = tables["edge"].num_rows
    n_defs = tables["definition"].num_rows
    n_syns = tables["synonym"].num_rows
    print(f"✅ wrote {n_nodes} nodes, {n_edges} edges, {n_defs} defs, {n_syns} synonyms → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
