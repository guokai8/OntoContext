"""
step00a_synth_ontology.py — Phase 0 (sub-step).

Implements the synthetic-ontology generator. The synthetic ontology is the
**primary development surface** for V0–V3 ablations (route-A bias — A9 of
the Phase-1 gate). It must produce output in the EXACT same Parquet
schema as `data/processed/<snapshot_id>/`, so downstream Phase-3 scripts
(steps 03, 04, 05, 06) consume it through the same code path.

Graph shape (per user decision, 2026-10-08): **layered random graph**.
- A single synthetic root, plus 1..3 secondary roots so the structure
  matches the multi-root nature of GO/CL/UBERON.
- Each layer is a `k`-ary skeleton (default `k = branching`).
- For each non-root layer, every node additionally has a small number
  (`--cross-density` of `1/branching`) of "shortcut" edges to a random
  earlier-layer node. This emulates the cross-ontology / cross-relation
  is-a density seen in real ontologies.
- Definitions are short, templated, with a `--ambiguity` parameter
  controlling how many other terms in the same layer share a near-identical
  definition prefix (so V2 discriminative capsules have real
  siblings to discriminate against).

Usage:
    # smoke test (200 nodes)
    python scripts/00_synth_ontology.py --small
    # default sweep (depth=6, branching=4, ambiguity=0.2)
    python scripts/00_synth_ontology.py --seed 17 --depth 6 --branching 4
    # explicit (depth, branching) matrix; each cell creates a v0/..v<n-1>/ tree
    python scripts/00_synth_ontology.py \\
        --depths 3 6 9 12 --branchings 2 4 8 --n-versions 2

Outputs (per the §3 schema of ontocontext-ontology-snapshots.mdc):
    cache/synthetic/<synthetic_id>/v0/
        node.parquet
        edge.parquet
        axiom.parquet
        definition.parquet
        synonym.parquet
        xref.parquet
        obsolete_replacement.parquet
        version.parquet
        manifest.yaml
        checksums.txt
    cache/synthetic/<synthetic_id>/v0/_logs/<run_id>.jsonl

Reproducibility:
    - All randomness goes through `random.Random(seed)`. NOT numpy.
    - Two runs with the same flags produce byte-identical Parquet and
      identical checksums (verified by `step00a_self_check.py`).
    - Default seed 17 (A14, A16).

Invariants honoured (from the umbrella rule):
    I   (LLM != DL): not relevant at this step (no LLM involved).
    II  (compress semantic, not identity): the snapshot_id and term id
         space are constructed to be deterministic and structural; the
         schema is the contract; the generator is the producer.
    III (grounded synopsis): not relevant at this step.

See also: PLAN.md §Step 00a, Purpose_and_Feasibility_Gate.md §4 rows A9, A14, A16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# numpy is a build-time dep; not strictly required. We use only stdlib.
import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # used for manifest.yaml emission at synthetic root


# ---------------------------------------------------------------------------
# Constants — single source of truth for synthetic id shape.
# ---------------------------------------------------------------------------

SYNTH_PREFIX = "SYNTH"  # so it is unmistakable in any logs / figures.
RELATION_IS_A = "obo:BFO_0000050"  # 'is_a' is what the synthetic edge graph encodes
RELATION_PART_OF = "obo:BFO_0000050"  # used as cross-relation flavour in v0
DEFAULT_ONTOLOGY_NAME = "SYNTH"
DEFAULT_OUTPUT_ROOT = "cache/synthetic"
DEFAULT_RUN_ID = "synth_seed_17"


# ---------------------------------------------------------------------------
# Logging — JSONL one line per snapshot, per run.
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def log_event(log_path: Path, event: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Schemas — verbatim copy of ontocontext-ontology-snapshots.mdc §3.
# Any change here MUST be reflected in PLAN.md and the capsule schema version.
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

VERSION_SCHEMA = pa.schema([
    pa.field("snapshot_id", pa.string(), nullable=False),
    pa.field("ontology", pa.string(), nullable=False),
    pa.field("ingested_at", pa.string(), nullable=False),
    pa.field("n_nodes", pa.uint64(), nullable=False),
    pa.field("n_edges", pa.uint64(), nullable=False),
    pa.field("n_axioms", pa.uint64(), nullable=False),
    pa.field("checksum", pa.string(), nullable=False),
])


# ---------------------------------------------------------------------------
# Generator — layered random graph, with cross-layer shortcuts.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynthConfig:
    seed: int
    depth: int                # max depth from any root (inclusive)
    branching: int            # mean children per non-leaf node
    ambiguity: float          # [0, 1] — fraction of intra-layer definitions that
                              #           are near-duplicates for V2 testing
    n_secondary_roots: int    # default 2 — emulates multi-root ontologies
    cross_density: float      # shortcut-edge probability per node
    n_secondary_relations: int  # how many of {part_of, regulates, ...} to
                                # include in the edge table for realism
    obsolete_rate: float      # default 0.0; tune up for the invalidation study
    snapshot_id: str
    version_label: str        # "v0", "v1", ...

    @property
    def n_layers(self) -> int:
        return self.depth

    @property
    def expected_nodes(self) -> int:
        # Very rough; real count depends on the random draw.
        # Useful only for progress messages.
        return 1 + self.n_secondary_roots + self.branching * (self.depth - 1)


def _id_for(prefix: str, layer: int, slot: int) -> str:
    """Format: SYNTH:0001_0001 (layer_slot). Locked to a fixed-width
    numeric encoding so all downstream scripts can sort strings."""
    return f"{SYNTH_PREFIX}:{prefix}{layer:04d}{slot:05d}"


def _short_letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=n))


def _name_for(layer: int, slot: int, rng: random.Random) -> str:
    # A 3-letter mnemonic + a slot. Deterministic given layer/slot.
    # We do not use rng here so the same (layer, slot) yields the same
    # name across runs (we want the *names* to be stable; the *edges*
    # are the only random part).
    return f"entity_{layer:02d}_{slot:04d}"


def _definition_for(
    layer: int,
    slot: int,
    layer_buckets: list[list[int]],
    ambiguity: float,
    rng: random.Random,
) -> str:
    """Generate a definition.

    With probability `ambiguity`, a term in this layer shares its
    definition's first 5 words with a randomly-chosen earlier
    term in the same layer. This is the V2 discrimination signal.
    """
    base = (
        f"a synthetic {layer:02d}-level entity of slot {slot:04d} "
        f"that participates in is_a relations and has a definition "
        f"used by V2 discriminative capsule builders."
    )
    if ambiguity > 0 and rng.random() < ambiguity and layer_buckets[layer]:
        # Pick a sibling and copy the first sentence.
        peer = rng.choice(layer_buckets[layer])
        peer_def = _definition_for(layer, peer, layer_buckets, ambiguity=0.0, rng=rng)
        head = " ".join(peer_def.split()[:8])
        tail = base
        return f"{head} {tail}"
    return base


def _synonyms_for(name: str, n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return list of (synonym, synonym_type) tuples.

    Synonym types follow OBO: EXACT, RELATED, BROAD, NARROW.
    """
    types = ["RELATED", "BROAD", "NARROW", "EXACT"]
    out: list[tuple[str, str]] = []
    for _ in range(n):
        suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=4))
        out.append((f"{name}_{suffix}", rng.choice(types)))
    return out


def _xrefs_for(name: str, n: int, rng: random.Random) -> list[str]:
    """Return list of CURIEs in unrelated namespaces — for cross-ontology
    index testing later. Names are stable enough that we can rebuild
    deterministically from the seed."""
    namespaces = ["PMID", "DOI", "CHEBI", "NCBITaxon", "EFO"]
    out: list[str] = []
    for _ in range(n):
        ns = rng.choice(namespaces)
        token = "".join(rng.choices(string.digits, k=7))
        out.append(f"{ns}:{token}")
    return out


def generate_graph(cfg: SynthConfig) -> dict:
    """Generate one synthetic version (`v0/`, `v1/`, …).

    Returns a dict with keys: nodes, edges, axioms, definitions, synonyms,
    xrefs, obsolete_replacements. All list-of-dict rows ready for the
    Parquet writer.
    """
    rng = random.Random(cfg.seed)

    # ---- Nodes ------------------------------------------------------------
    nodes: list[dict] = []
    node_id_by_layer_slot: dict[tuple[int, int], str] = {}
    layer_buckets: list[list[int]] = [[] for _ in range(cfg.n_layers)]
    children_by_layer: list[list[tuple[int, int]]] = [[] for _ in range(cfg.n_layers)]

    # Roots (layer 0)
    roots: list[tuple[int, int]] = []
    for s in range(1 + cfg.n_secondary_roots):
        nid = _id_for("R", 0, s)
        node_id_by_layer_slot[(0, s)] = nid
        roots.append((0, s))
        layer_buckets[0].append(s)
        nodes.append({
            "id": nid,
            "ontology": DEFAULT_ONTOLOGY_NAME,
            "snapshot_id": cfg.snapshot_id,
            "name": "synthetic_root" if s == 0 else f"synthetic_secondary_root_{s}",
            "definition": f"synthetic root entity number {s}",
            "obsolete": False,
            "replaced_by": None,
            "alt_ids": [],
            "depth": 0,
            "root_path": nid,
        })

    # Body nodes: per layer
    slot_counter = 0
    for layer in range(1, cfg.n_layers):
        # Branching may vary slightly per layer for realism.
        n_children = max(1, int(round(cfg.branching * (1 + 0.1 * rng.random() - 0.05))))
        # Parents: pick a random subset of previous layer (weighted by id
        # to keep ordering stable).
        parents = [ps for (pl, ps) in node_id_by_layer_slot.keys() if pl == layer - 1]
        rng.shuffle(parents)
        for child_index in range(n_children):
            slot_counter += 1
            nid = _id_for("N", layer, child_index)
            node_id_by_layer_slot[(layer, child_index)] = nid
            layer_buckets[layer].append(child_index)
            children_by_layer[layer].append((layer, child_index))
            depth = layer  # tree depth in a layered random graph = layer
            root_path = (
                nodes[0]["root_path"] + " > " + nid if nodes else nid
            )
            nodes.append({
                "id": nid,
                "ontology": DEFAULT_ONTOLOGY_NAME,
                "snapshot_id": cfg.snapshot_id,
                "name": _name_for(layer, child_index, rng),
                "definition": _definition_for(
                    layer, child_index, layer_buckets, cfg.ambiguity, rng
                ),
                "obsolete": False,
                "replaced_by": None,
                "alt_ids": [],
                "depth": depth,
                "root_path": root_path,
            })

    # ---- Edges ------------------------------------------------------------
    edges: list[dict] = []
    edge_seq = 0
    for (layer, slot), nid in node_id_by_layer_slot.items():
        if layer == 0:
            # Roots have no is_a parents in this synthetic convention.
            continue
        # Primary parent: the slot-th "child slot" was assigned a parent
        # at construction time. We use the deterministic mapping by
        # dividing the body node's global slot_counter by parents count
        # to keep the mapping stable without re-running construction.
        # Simpler approach: assign parent as node_id_by_layer_slot[
        #   (layer - 1, slot % len(parents_in_prev_layer))
        # ]. This is deterministic given the RNG ordering of `parents`.
        prev_layer_keys = [
            (pl, ps) for (pl, ps) in node_id_by_layer_slot.keys() if pl == layer - 1
        ]
        # We deliberately reshuffle parents for every layer to mimic
        # the layer-bucket order, but we need a deterministic mapping.
        # Trick: derive a seed from the layer and pick parent index
        # as (slot * 31 + layer) % len(prev_layer_keys). Stable.
        prev_layer_keys_sorted = sorted(prev_layer_keys, key=lambda x: x[1])
        parent_idx = (slot * 31 + layer) % len(prev_layer_keys_sorted)
        parent_key = prev_layer_keys_sorted[parent_idx]
        parent_id = node_id_by_layer_slot[parent_key]
        edge_seq += 1
        edges.append({
            "src_id": parent_id,
            "relation_iri": RELATION_IS_A,
            "dst_id": nid,
            "ontology": DEFAULT_ONTOLOGY_NAME,
            "snapshot_id": cfg.snapshot_id,
            "source_term": f"axiom_{edge_seq:06d}",
            "evidence": "ECO:0000501",  # curator inference
        })

    # Cross-layer shortcut edges (layered random graph signature feature)
    if cfg.cross_density > 0:
        for (layer, slot), nid in node_id_by_layer_slot.items():
            if layer < 2:
                continue
            n_shortcuts = max(0, int(round(cfg.cross_density / max(1, cfg.branching))))
            for _ in range(n_shortcuts):
                target_layer = rng.randint(0, layer - 1)
                target_keys = [
                    (pl, ps) for (pl, ps) in node_id_by_layer_slot.keys()
                    if pl == target_layer
                ]
                if not target_keys:
                    continue
                target_key = rng.choice(target_keys)
                target_id = node_id_by_layer_slot[target_key]
                if target_id == nid:
                    continue
                edge_seq += 1
                edges.append({
                    "src_id": target_id,
                    "relation_iri": RELATION_IS_A,
                    "dst_id": nid,
                    "ontology": DEFAULT_ONTOLOGY_NAME,
                    "snapshot_id": cfg.snapshot_id,
                    "source_term": f"axiom_{edge_seq:06d}",
                    "evidence": "ECO:0000501",
                })

    # Secondary relation flavour (used by V3 module analysis)
    secondary_relations = [
        "BFO:0000050",   # part_of
        "RO:0002211",    # regulates
        "RO:0002212",    # negatively regulates
        "RO:0002213",    # positively regulates
    ][: cfg.n_secondary_relations]
    if secondary_relations:
        for (layer, slot), nid in node_id_by_layer_slot.items():
            if layer < 1:
                continue
            for rel in secondary_relations:
                if rng.random() < 0.05:  # 5% density
                    target_keys = [
                        (pl, ps) for (pl, ps) in node_id_by_layer_slot.keys()
                        if pl == layer
                    ]
                    if not target_keys:
                        continue
                    target_key = rng.choice(target_keys)
                    target_id = node_id_by_layer_slot[target_key]
                    if target_id == nid:
                        continue
                    edge_seq += 1
                    edges.append({
                        "src_id": nid,
                        "relation_iri": rel,
                        "dst_id": target_id,
                        "ontology": DEFAULT_ONTOLOGY_NAME,
                        "snapshot_id": cfg.snapshot_id,
                        "source_term": f"axiom_{edge_seq:06d}",
                        "evidence": "ECO:0000501",
                    })

    # ---- Axioms -----------------------------------------------------------
    axioms: list[dict] = []
    for n in nodes:
        axioms.append({
            "term_id": n["id"],
            "axiom_type": "SubClassOf",
            "owl_expression": f"SubClassOf({n['id']} ObjectSomeValuesFrom({RELATION_IS_A} ObjectSomeValuesFrom({RELATION_IS_A} owl:Thing)))",
            "ontology": DEFAULT_ONTOLOGY_NAME,
            "snapshot_id": cfg.snapshot_id,
        })

    # ---- Definitions (one per node, the node's "definition" field) -------
    definitions: list[dict] = []
    for n in nodes:
        definitions.append({
            "term_id": n["id"],
            "definition_id": f"def_{n['id'].split(':')[1]}",
            "text": n["definition"] or "",
            "source": "synthetic_generator",
            "ontology": DEFAULT_ONTOLOGY_NAME,
            "snapshot_id": cfg.snapshot_id,
        })

    # ---- Synonyms --------------------------------------------------------
    synonyms: list[dict] = []
    for n in nodes:
        n_syn = max(0, int(rng.gauss(1.0, 0.5)))
        for syn_text, syn_type in _synonyms_for(n["name"], n_syn, rng):
            synonyms.append({
                "term_id": n["id"],
                "synonym": syn_text,
                "synonym_type": syn_type,
                "source": "synthetic_generator",
                "ontology": DEFAULT_ONTOLOGY_NAME,
                "snapshot_id": cfg.snapshot_id,
            })

    # ---- Xrefs -----------------------------------------------------------
    xrefs: list[dict] = []
    for n in nodes:
        n_xref = max(0, int(rng.gauss(0.4, 0.5)))
        for x in _xrefs_for(n["name"], n_xref, rng):
            xrefs.append({
                "term_id": n["id"],
                "xref_id": x,
                "source": "synthetic_generator",
                "ontology": DEFAULT_ONTOLOGY_NAME,
                "snapshot_id": cfg.snapshot_id,
            })

    # ---- Obsolete replacements (sparse; used for invalidation study) -----
    obsolete_replacements: list[dict] = []
    if cfg.obsolete_rate > 0:
        n_obs = max(0, int(len(nodes) * cfg.obsolete_rate))
        for n in rng.sample(nodes, k=n_obs):
            replacement = rng.choice(nodes)
            if replacement["id"] == n["id"]:
                continue
            obsolete_replacements.append({
                "obsolete_id": n["id"],
                "replacement_id": replacement["id"],
                "reason": "synthetic_merge",
                "ontology": DEFAULT_ONTOLOGY_NAME,
                "snapshot_id": cfg.snapshot_id,
            })
            # Mutate the node to obsolete=True.
            n["obsolete"] = True
            n["replaced_by"] = replacement["id"]

    return {
        "nodes": nodes,
        "edges": edges,
        "axioms": axioms,
        "definitions": definitions,
        "synonyms": synonyms,
        "xrefs": xrefs,
        "obsolete_replacements": obsolete_replacements,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    if not rows:
        # Empty table but with the locked schema.
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
    return pa.Table.from_pylist(rows, schema=schema)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_version(
    out_dir: Path,
    cfg: SynthConfig,
    data: dict,
) -> dict:
    """Write one version (v0/, v1/, …) and return a per-version summary
    dict with file paths, row counts, and content hashes."""
    version_dir = out_dir / cfg.version_label
    version_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "node.parquet":                (data["nodes"], NODE_SCHEMA),
        "edge.parquet":                (data["edges"], EDGE_SCHEMA),
        "axiom.parquet":               (data["axioms"], AXIOM_SCHEMA),
        "definition.parquet":          (data["definitions"], DEFINITION_SCHEMA),
        "synonym.parquet":             (data["synonyms"], SYNONYM_SCHEMA),
        "xref.parquet":                (data["xrefs"], XREF_SCHEMA),
        "obsolete_replacement.parquet":(data["obsolete_replacements"], OBSOLETE_REPLACEMENT_SCHEMA),
    }
    summary: dict = {"files": {}, "rows": {}}
    for name, (rows, schema) in tables.items():
        t = _table(rows, schema)
        path = version_dir / name
        pq.write_table(t, path, compression="snappy", use_dictionary=False)
        summary["files"][name] = str(path)
        summary["rows"][name] = len(rows)

    # version.parquet (bookkeeping)
    n_nodes = len(data["nodes"])
    n_edges = len(data["edges"])
    n_axioms = len(data["axioms"])
    # Checksum: hash all data tables together, deterministic.
    h = hashlib.sha256()
    for name in sorted(summary["files"].keys()):
        with open(summary["files"][name], "rb") as fh:
            h.update(fh.read())
    checksum = h.hexdigest()
    version_table = pa.table({
        "snapshot_id": [cfg.snapshot_id],
        "ontology": [DEFAULT_ONTOLOGY_NAME],
        "ingested_at": [_now_iso()],
        "n_nodes": [n_nodes],
        "n_edges": [n_edges],
        "n_axioms": [n_axioms],
        "checksum": [checksum],
    }, schema=VERSION_SCHEMA)
    version_path = version_dir / "version.parquet"
    pq.write_table(version_table, version_path, compression="snappy")
    summary["files"]["version.parquet"] = str(version_path)

    # checksums.txt
    checksums_path = version_dir / "checksums.txt"
    with checksums_path.open("w", encoding="utf-8") as fh:
        for name in sorted(summary["files"].keys()):
            rel = name
            digest = _sha256_file(Path(summary["files"][rel]))
            fh.write(f"{digest}  {rel}\n")

    summary["checksum"] = checksum
    summary["n_nodes"] = n_nodes
    summary["n_edges"] = n_edges
    summary["n_axioms"] = n_axioms
    return summary


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------

def _synthetic_id(seed: int, depth: int, branching: int) -> str:
    return f"synth_d{depth:02d}_b{branching:02d}_s{seed:03d}"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OntoContext synthetic ontology generator")
    p.add_argument("--seed", type=int, default=17, help="RNG seed (default 17, A14/A16).")
    p.add_argument("--depth", type=int, default=6, help="max depth (layers), 1..20")
    p.add_argument("--branching", type=int, default=4, help="avg children per non-leaf")
    p.add_argument("--ambiguity", type=float, default=0.2,
                   help="intra-layer near-duplicate definition rate [0,1]")
    p.add_argument("--cross-density", type=float, default=1.0,
                   help="shortcut edges per node (scaled by 1/branching)")
    p.add_argument("--n-secondary-relations", type=int, default=3)
    p.add_argument("--obsolete-rate", type=float, default=0.0)
    p.add_argument("--n-versions", type=int, default=1,
                   help="number of successive versions (v0..vN-1) to emit")
    p.add_argument("--update-rate", type=float, default=0.0,
                   help="fraction of nodes to mutate between successive versions")
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUTPUT_ROOT),
                   help="output root; default cache/synthetic")
    p.add_argument("--small", action="store_true",
                   help="smoke test: depth=2, branching=2, n-versions=1")
    p.add_argument("--dry-run", action="store_true",
                   help="plan + count + log; do not write any files")
    p.add_argument("--depths", type=int, nargs="*", default=None,
                   help="sweep: generate one synthetic per depth value")
    p.add_argument("--branchings", type=int, nargs="*", default=None,
                   help="sweep: generate one synthetic per branching value")
    return p.parse_args(argv)


def _mutate_for_version(base: dict, cfg: SynthConfig, v_index: int, rng: random.Random) -> dict:
    """Return a new dict of tables for a successor version.

    Strategy (controlled by --update-rate):
    - With probability `update_rate` per node, change its name
      (suffix bump), definition (small edit), and obsolete status.
    - Add `update_rate * n_edges` new shortcut edges.
    The new version's checksum will differ; the new version's
    *shape* (node/edge counts) is within ±5% of the base.
    """
    new = {
        "nodes": [dict(n) for n in base["nodes"]],
        "edges": [dict(e) for e in base["edges"]],
        "axioms": [dict(a) for a in base["axioms"]],
        "definitions": [dict(d) for d in base["definitions"]],
        "synonyms": [dict(s) for s in base["synonyms"]],
        "xrefs": [dict(x) for x in base["xrefs"]],
        "obsolete_replacements": [dict(o) for o in base["obsolete_replacements"]],
    }
    if cfg.update_rate <= 0 or v_index == 0:
        return new

    n_to_change = max(1, int(len(new["nodes"]) * cfg.update_rate))
    chosen = rng.sample(new["nodes"], k=n_to_change)
    edge_seq_offset = max(int(e["source_term"].split("_")[1]) for e in new["edges"]) + 1
    for n in chosen:
        n["name"] = n["name"] + f"_v{v_index + 1}"
        if n["definition"]:
            n["definition"] = n["definition"] + f" [rev {v_index + 1}]"
        # Find and update matching definition row
        for d in new["definitions"]:
            if d["term_id"] == n["id"]:
                d["text"] = n["definition"]
                break
    # Add some new edges for the new version
    n_new_edges = max(1, int(len(new["edges"]) * cfg.update_rate))
    nodes_ids = [n["id"] for n in new["nodes"]]
    for i in range(n_new_edges):
        a = rng.choice(nodes_ids)
        b = rng.choice(nodes_ids)
        if a == b:
            continue
        new["edges"].append({
            "src_id": a,
            "relation_iri": RELATION_IS_A,
            "dst_id": b,
            "ontology": DEFAULT_ONTOLOGY_NAME,
            "snapshot_id": cfg.snapshot_id,
            "source_term": f"axiom_{edge_seq_offset + i:06d}",
            "evidence": "ECO:0000501",
        })
    return new


def run_one(args: argparse.Namespace, depth: int, branching: int) -> dict:
    """Build one synthetic per (depth, branching) and write all versions."""
    if args.small:
        depth = 2
        branching = 2
        n_versions = 1
        ambiguity = 0.1
        obsolete_rate = 0.0
    else:
        n_versions = max(1, args.n_versions)
        ambiguity = args.ambiguity
        obsolete_rate = args.obsolete_rate

    syn_id = _synthetic_id(args.seed, depth, branching)
    out_root = args.out_dir / syn_id
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_logs" / f"{syn_id}.jsonl"

    summary = {
        "synthetic_id": syn_id,
        "depth": depth,
        "branching": branching,
        "seed": args.seed,
        "ambiguity": ambiguity,
        "n_versions": n_versions,
        "out_dir": str(out_root),
        "versions": [],
    }
    if args.dry_run:
        return summary

    base_cfg = SynthConfig(
        seed=args.seed,
        depth=depth,
        branching=branching,
        ambiguity=ambiguity,
        n_secondary_roots=2,
        cross_density=args.cross_density,
        n_secondary_relations=args.n_secondary_relations,
        obsolete_rate=obsolete_rate,
        snapshot_id=syn_id,
        version_label="v0",
    )

    base = generate_graph(base_cfg)
    v_summary = write_version(out_root, base_cfg, base)
    summary["versions"].append({"label": "v0", **v_summary})
    log_event(log_path, {
        "ts": _now_iso(), "event": "version_written", "version": "v0",
        "n_nodes": v_summary["n_nodes"], "n_edges": v_summary["n_edges"],
        "checksum": v_summary["checksum"],
    })

    for vi in range(1, n_versions):
        rng = random.Random(args.seed + vi)
        cfg_v = SynthConfig(
            seed=args.seed + vi,
            depth=depth,
            branching=branching,
            ambiguity=ambiguity,
            n_secondary_roots=2,
            cross_density=args.cross_density,
            n_secondary_relations=args.n_secondary_relations,
            obsolete_rate=obsolete_rate,
            snapshot_id=syn_id,
            version_label=f"v{vi}",
        )
        data_v = _mutate_for_version(base, cfg_v, vi, rng)
        v_sum = write_version(out_root, cfg_v, data_v)
        summary["versions"].append({"label": f"v{vi}", **v_sum})
        log_event(log_path, {
            "ts": _now_iso(), "event": "version_written", "version": f"v{vi}",
            "n_nodes": v_sum["n_nodes"], "n_edges": v_sum["n_edges"],
            "checksum": v_sum["checksum"],
        })

    # manifest.yaml at the synthetic root, mirrors the real snapshot manifest
    manifest = {
        "snapshot_id": syn_id,
        "ontology_name": DEFAULT_ONTOLOGY_NAME,
        "ontology_version_iri": f"urn:ontocontext:synthetic:{syn_id}",
        "release_date": _now_iso()[:10],
        "source": {
            "type": "synthetic_generator",
            "url": "local",
            "mirror": "local",
        },
        "license": "CC0-1.0",
        "checksum": {
            "algorithm": "sha256",
            "value": v_summary["checksum"],
        },
        "parser": "scripts/00_synth_ontology.py",
        "owl_release_url": "",
        "notes": "Synthetic layered-random-graph ontology for OntoContext V0-V3 ablations.",
    }
    (out_root / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    # README
    (out_root / "README.md").write_text(
        f"# Synthetic ontology: {syn_id}\n\n"
        f"- Seed: {args.seed}\n- Depth: {depth}\n- Branching: {branching}\n"
        f"- Ambiguity: {ambiguity}\n- Versions: {n_versions}\n"
        f"- Total nodes (v0): {v_summary['n_nodes']}\n"
        f"- Total edges (v0): {v_summary['n_edges']}\n"
        f"- Layout: layered random graph (per plan §Step 00a).\n",
        encoding="utf-8",
    )

    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.small:
        summaries = [run_one(args, depth=2, branching=2)]
    elif args.depths or args.branchings:
        depths = args.depths or [args.depth]
        branchings = args.branchings or [args.branching]
        summaries = []
        for d in depths:
            for b in branchings:
                summaries.append(run_one(args, depth=d, branching=b))
    else:
        summaries = [run_one(args, depth=args.depth, branching=args.branching)]

    # Report
    print(f"Generated {len(summaries)} synthetic ontology(ies):")
    for s in summaries:
        for v in s["versions"]:
            print(
                f"  {s['synthetic_id']}/{v['label']}: "
                f"nodes={v['n_nodes']} edges={v['n_edges']} "
                f"checksum={v['checksum'][:12]}…"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
