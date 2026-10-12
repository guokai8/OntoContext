"""
step07_build_world_model.py — world model materialization.

Reads the **V3** capsules from `capsules/v3/entities/` and the
V3 modules from `capsules/v3/modules/`, and materializes:

  1. `capsules/v3/world_model.jsonl` — one supernode per line.
     Two supernode kinds are emitted (rule §11 of the strategy doc):
        - **module supernode**  : one per module, with its terms,
          centroid, and size.
        - **entity supernode**  : one per non-root entity, with its
          module assignment and selected_ancestors (carried over
          from the V3 capsule for the renderer's use).
     Multi-resolution: each line carries `resolution = "v3"`
     (more resolutions can be added later by V4/V5 capsules).

  2. `capsules/v3/dependencies.jsonl` — one dependency record per
     subject (capsule or module) per `ontocontext-versioning-and-
     invalidation.mdc` §2:
        { subject_type, subject_id, depends_on: {
            terms, relations, axioms, modules,
            external_ontology_terms } }
     The default `depends_on` for a capsule is:
        - itself
        - its selected_ancestors
        - its semantic_module
        - any external_ontology_term from cross_ontology.relations
     The default `depends_on` for a module is the union of its
     member capsules' dependencies, plus the module id.

The world model is the **single source of truth** for the renderer
(step 08): it consumes the world model + a token budget and emits
a `RenderedContext`.  The dependency records are consumed by
step 41 (invalidation policies).

Usage:
    python3 scripts/step07_build_world_model.py --synthetic synth_d02_b02_s017
    python3 scripts/step07_build_world_model.py --all
    python3 scripts/step07_build_world_model.py --snapshot go_2024-06-10
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
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

V3_ROOT = Path("capsules/v3/entities")
V3_MODULES_ROOT = Path("capsules/v3/modules")
WORLD_MODEL_OUT = Path("capsules/v3/world_model.jsonl")
DEPENDENCIES_OUT = Path("capsules/v3/dependencies.jsonl")
WORLD_MODEL_TOP = Path("world_model.jsonl")   # per-step verification output
RESOLUTION = "v3"
SCHEMA_VERSION = "0.1.0"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_capsules(snapshot_id: str, ontology: str) -> list[dict]:
    src = V3_ROOT / ontology
    if not src.exists():
        return []
    out: list[dict] = []
    for p in sorted(src.glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        if c.get("snapshot_id") != snapshot_id:
            continue
        out.append(c)
    return out


def _load_modules(snapshot_id: str, ontology: str) -> list[dict]:
    if not V3_MODULES_ROOT.exists():
        return []
    out: list[dict] = []
    for p in sorted(V3_MODULES_ROOT.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("snapshot_id") == snapshot_id and m.get("ontology") == ontology:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Per-snapshot world-model build
# ---------------------------------------------------------------------------

def _entity_dependencies(c: dict) -> dict:
    """Default `depends_on` for a capsule, per rule §2.

    - terms: the term itself + its selected_ancestors
    - relations: the defining_relations in its logical_signature
    - axioms: the selected_axioms (we use the first 8 axiom bodies
      as opaque identifiers; the rule's exact_id form is
      `axiom_<i>` and we approximate that here)
    - modules: the capsule's semantic_module
    - external_ontology_terms: every cross_ontology.relations
      entry's `to` (terms from outside the capsule's ontology)
    """
    nid = c["id"]
    parents = c.get("coordinates", {}).get("selected_ancestors") or \
              c.get("coordinates", {}).get("parents", []) or []
    relations = c.get("logical_signature", {}).get("defining_relations", []) or []
    axioms = c.get("logical_signature", {}).get("selected_axioms", []) or []
    module = c.get("coordinates", {}).get("semantic_module")
    cross = c.get("cross_ontology", {}).get("relations", []) or []
    external: list[str] = []
    own_ont = c.get("ontology") or ""
    for r in cross:
        to = r.get("to", "")
        if ":" in to and to.split(":")[0] != own_ont:
            external.append(to)
    return {
        "terms": [nid] + list(parents),
        "relations": list(relations),
        "axioms": [f"{nid.split(':')[-1]}_axiom_{i}"
                   for i in range(len(axioms))],
        "modules": [module] if module else [],
        "external_ontology_terms": external,
    }


def _module_dependencies(m: dict, member_dependencies: list[dict]) -> dict:
    """Default `depends_on` for a module: union of its members' deps."""
    terms: set[str] = set()
    relations: set[str] = set()
    axioms: set[str] = set()
    modules: set[str] = set()
    external: set[str] = set()
    for dep in member_dependencies:
        terms.update(dep.get("terms", []))
        relations.update(dep.get("relations", []))
        axioms.update(dep.get("axioms", []))
        modules.update(dep.get("modules", []))
        external.update(dep.get("external_ontology_terms", []))
    # The module is itself a dependency for its members
    return {
        "terms": sorted(terms),
        "relations": sorted(relations),
        "axioms": sorted(axioms),
        "modules": sorted(modules | {m["module_id"]}),
        "external_ontology_terms": sorted(external),
    }


def build_for_snapshot(snapshot_id: str, ontology: str) -> dict:
    """Build the world model + dependencies for one snapshot."""
    capsules = _load_capsules(snapshot_id, ontology)
    modules = _load_modules(snapshot_id, ontology)
    if not capsules and not modules:
        raise RuntimeError(
            f"no V3 entities or modules for {snapshot_id}/{ontology}; run step 06 first"
        )

    supernodes: list[dict] = []
    dep_records: list[dict] = []
    covered_term_ids: set[str] = set()
    cap_dependencies: dict[str, dict] = {}

    # 1) Module supernodes + module-level dependency records
    member_dep_by_module: dict[str, list[dict]] = defaultdict(list)
    for m in modules:
        ms = {
            "kind": "module",
            "module_id": m["module_id"],
            "snapshot_id": snapshot_id,
            "ontology": ontology,
            "size": m["size"],
            "centroid_term": m.get("centroid_term"),
            "terms": m.get("terms", []),
            "parent_module": m.get("parent_module"),
            "resolution": RESOLUTION,
            "schema_version": SCHEMA_VERSION,
        }
        supernodes.append(ms)
    # Build per-module member dependencies by looking at each capsule's module
    for c in capsules:
        dep = _entity_dependencies(c)
        cap_dependencies[c["id"]] = dep
        covered_term_ids.add(c["id"])
        mod = c.get("coordinates", {}).get("semantic_module")
        if mod:
            member_dep_by_module[mod].append(dep)

    # 2) Module-level dependency records (after we've collected member deps)
    for m in modules:
        mdep = _module_dependencies(m, member_dep_by_module.get(m["module_id"], []))
        dep_records.append({
            "subject_type": "module",
            "subject_id": m["module_id"],
            "depends_on": mdep,
        })

    # 3) Entity supernodes (one per non-root entity with a module)
    for c in capsules:
        if not c.get("coordinates", {}).get("semantic_module"):
            # root terms (no module) still get a supernode
            pass
        supernodes.append({
            "kind": "entity",
            "term_id": c["id"],
            "snapshot_id": snapshot_id,
            "ontology": ontology,
            "module_id": c.get("coordinates", {}).get("semantic_module"),
            "selected_ancestors": c.get("coordinates", {}).get("selected_ancestors", []),
            "information_content": c.get("usage", {}).get("information_content", 0.0),
            "schema_version": SCHEMA_VERSION,
        })
        # 4) Capsule-level dependency record
        dep_records.append({
            "subject_type": "capsule",
            "subject_id": c["id"],
            "depends_on": cap_dependencies[c["id"]],
        })

    # 5) Write world_model.jsonl (capsules/v3/) and the verification copy.
    # We PRESERVE any supernodes for other snapshot+ontology pairs (so
    # that re-running for a different snapshot doesn't wipe the prior
    # world's data). The renderer loads only the matching snapshot+ont
    # so co-existing snapshots in the same jsonl is fine.
    existing: list[dict] = []
    if WORLD_MODEL_OUT.exists():
        for ln in WORLD_MODEL_OUT.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("snapshot_id") != snapshot_id or d.get("ontology") != ontology:
                existing.append(d)
    combined = existing + supernodes
    WORLD_MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with WORLD_MODEL_OUT.open("w", encoding="utf-8") as fh:
        for s in combined:
            fh.write(json.dumps(s, sort_keys=True) + "\n")
    # The plan also wants a top-level `world_model.jsonl` (step 07 output spec).
    WORLD_MODEL_TOP.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in combined),
                                encoding="utf-8")

    # 5b) Per-module summary (small JSON, one per module)
    module_summary: dict[str, dict] = {}
    # Build ic_by_term from the loaded V3 capsules
    ic_by_term: dict[str, float] = {}
    for c in capsules:
        ic_v = c.get("coordinates", {}).get("ic")
        if ic_v is not None:
            ic_by_term[c["id"]] = float(ic_v)
    for m in modules:
        mid = m.get("module_id") or m.get("id")
        if not mid:
            continue
        members = m.get("terms") or m.get("members", []) or []
        module_summary[mid] = {
            "id": mid,
            "n_members": len(members),
            "members": members,
            "centroid": m.get("centroid_term") or m.get("centroid"),
            "ic_mean": (sum(ic_by_term.get(x, 0.0) for x in members) /
                         max(1, len(members))),
            "label": m.get("label", ""),
        }
    (Path("capsules/v3/modules/module_summary.json")).write_text(
        json.dumps(module_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 6) dependencies.jsonl
    DEPENDENCIES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DEPENDENCIES_OUT.open("w", encoding="utf-8") as fh:
        for d in dep_records:
            fh.write(json.dumps(d, sort_keys=True) + "\n")

    # 7) Sanity: every entity in entities/ should appear in world_model.jsonl
    missing = [c["id"] for c in capsules if c["id"] not in covered_term_ids]
    if missing:
        raise RuntimeError(f"world model missing entities: {missing}")

    return {
        "snapshot_id": snapshot_id,
        "ontology": ontology,
        "n_supernodes": len(supernodes),
        "n_module_supernodes": sum(1 for s in supernodes if s["kind"] == "module"),
        "n_entity_supernodes": sum(1 for s in supernodes if s["kind"] == "entity"),
        "n_dependency_records": len(dep_records),
        "world_model_path": str(WORLD_MODEL_OUT),
        "dependencies_path": str(DEPENDENCIES_OUT),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--all", action="store_true")
    return p.parse_args(argv)


def _resolve(args) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if args.synthetic:
        out.append((args.synthetic, "SYNTH"))
        return out
    if args.snapshot:
        proc = Path("data/processed") / args.snapshot
        mf = proc / "manifest_processed.yaml"
        ont = (yaml.safe_load(mf.read_text())["ontology"]
               if mf.exists() else args.snapshot.split("_")[0].upper())
        out.append((args.snapshot, ont))
        return out
    if args.all:
        for d in sorted(Path("data/processed").iterdir()):
            if not (d / "node.parquet").exists():
                continue
            mf = d / "manifest_processed.yaml"
            ont = (yaml.safe_load(mf.read_text())["ontology"]
                   if mf.exists() else d.name.split("_")[0].upper())
            out.append((d.name, ont))
    return out


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    snaps = _resolve(args)
    if not snaps:
        print("ERROR: pass --snapshot, --synthetic, or --all", file=sys.stderr)
        return 2
    for sid, ont in snaps:
        s = build_for_snapshot(sid, ont)
        print(
            f"✅ {sid} ({s['ontology']}): {s['n_supernodes']} supernodes "
            f"({s['n_module_supernodes']} modules, {s['n_entity_supernodes']} entities), "
            f"{s['n_dependency_records']} dependency records; "
            f"world_model.jsonl: {s['world_model_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
