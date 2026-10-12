"""
step41_invalidation.py — AOCI dependency invalidation.

Given a delta (from step 40), read the dependency records
(`capsules/v3/dependencies.jsonl`) and emit a list of capsules
that need to be invalidated.

A capsule c is invalidated if:
    - c.id is in delta.added, delta.removed, or delta.modified
    - c is a parent or child of any modified term
    - c is in the same module as a modified term (semantic_module)

Output:
    cache/invalidation/<old>__<new>.json

Usage:
    python3 scripts/step41_invalidation.py --old synth_d02_b02_s017 --new synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step07_build_world_model import _load_capsules
from step08_render import _infer_ontology

# Determinism per the plan.
np.random.seed(17)


def _load_dependencies(snapshot_id: str) -> list[dict]:
    """Load AOCI dependency records from the V3 world model."""
    dep_path = Path("capsules/v3") / "dependencies.jsonl"
    if not dep_path.exists():
        return []
    out: list[dict] = []
    for line in dep_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("snapshot_id") == snapshot_id or d.get("snapshot") == snapshot_id:
            out.append(d)
    return out


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.old or not args.new:
        print("ERROR: --old and --new required", file=sys.stderr)
        return 2
    delta_path = Path("cache/delta") / f"{args.old}__{args.new}.json"
    if not delta_path.exists():
        print(f"ERROR: {delta_path} not found; run step 40 first", file=sys.stderr)
        return 2
    delta = json.loads(delta_path.read_text())
    ontology = delta.get("ontology") or _infer_ontology(args.new)
    # Build a set of directly-affected term IDs
    directly_affected: set[str] = set(delta.get("added", [])) \
        | set(delta.get("removed", [])) \
        | set(delta.get("modified", {}).keys())
    # Find modules containing any affected term
    new_caps = {c["id"]: c for c in _load_capsules(args.new, ontology)}
    affected_modules: set[str] = set()
    for tid in directly_affected:
        c = new_caps.get(tid)
        if c:
            sm = c.get("coordinates", {}).get("semantic_module")
            if sm:
                affected_modules.add(sm)
    # Invalidation cascade: parent / child of any modified term
    cascade: set[str] = set()
    for tid in directly_affected:
        c = new_caps.get(tid)
        if not c:
            continue
        cascade.update(c.get("coordinates", {}).get("parents", []) or [])
        cascade.update(c.get("coordinates", {}).get("selected_children", []) or [])
    invalidation_set = directly_affected | cascade
    # Add all capsules whose semantic_module is affected
    for c in new_caps.values():
        sm = c.get("coordinates", {}).get("semantic_module")
        if sm and sm in affected_modules:
            invalidation_set.add(c["id"])

    # Read dependency records
    deps = _load_dependencies(args.new)
    dep_invalidated: list[dict] = [d for d in deps if d.get("target_id") in invalidation_set]

    out = {
        "old": args.old, "new": args.new,
        "directly_affected": sorted(directly_affected),
        "cascade_affected": sorted(cascade),
        "affected_modules": sorted(affected_modules),
        "n_invalidated_capsules": len(invalidation_set),
        "n_invalidated_dependencies": len(dep_invalidated),
        "invalidated_dependencies": dep_invalidated,
    }
    out_path = Path("cache/invalidation")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{args.old}__{args.new}.json"
    out_file.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ invalidation {args.old} → {args.new}: "
          f"{len(invalidation_set)} capsules, {len(dep_invalidated)} dependencies; "
          f"{out_file}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
