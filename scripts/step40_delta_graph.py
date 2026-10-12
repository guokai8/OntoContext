"""
step40_delta_graph.py — delta graph between two snapshots.

Compares two V3 capsule sets and emits the per-term delta:
    added     — term in new but not old
    removed   — term in old but not new
    modified  — term in both, with field-level changes
    unchanged — term in both, no changes

Output:
    cache/delta/<old>__<new>.json

Usage:
    python3 scripts/step40_delta_graph.py --old synth_d02_b02_s017 --new synth_d02_b02_s017
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


def _load_caps_set(snapshot_id: str, ontology: str) -> dict[str, dict]:
    return {c["id"]: c for c in _load_capsules(snapshot_id, ontology)}


def _field_diff(old: dict, new: dict) -> dict[str, dict]:
    """Compute field-level diffs between two capsules."""
    diffs: dict[str, dict] = {}
    # scalar fields
    for k in ("name", "obsolete", "version"):
        if old.get(k) != new.get(k):
            diffs[k] = {"old": old.get(k), "new": new.get(k)}
    # nested fields
    coords_old = old.get("coordinates", {})
    coords_new = new.get("coordinates", {})
    for k in ("parents", "selected_children"):
        old_v = set(coords_old.get(k, []) or [])
        new_v = set(coords_new.get(k, []) or [])
        if old_v != new_v:
            diffs[k] = {"added": sorted(new_v - old_v),
                        "removed": sorted(old_v - new_v)}
    relations_old = old.get("logical_signature", {}).get("defining_relations", []) or []
    relations_new = new.get("logical_signature", {}).get("defining_relations", []) or []
    if set(relations_old) != set(relations_new):
        diffs["defining_relations"] = {"added": sorted(set(relations_new) - set(relations_old)),
                                        "removed": sorted(set(relations_old) - set(relations_new))}
    return diffs


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.old or not args.new:
        print("ERROR: --old and --new required", file=sys.stderr)
        return 2
    ontology = _infer_ontology(args.new)
    old = _load_caps_set(args.old, ontology)
    new = _load_caps_set(args.new, ontology)
    all_ids = sorted(set(old) | set(new))
    added: list[str] = []
    removed: list[str] = []
    modified: dict[str, dict] = {}
    unchanged: list[str] = []
    for tid in all_ids:
        if tid in old and tid not in new:
            removed.append(tid)
        elif tid not in old and tid in new:
            added.append(tid)
        else:
            d = _field_diff(old[tid], new[tid])
            if d:
                modified[tid] = d
            else:
                unchanged.append(tid)
    delta = {
        "old": args.old, "new": args.new,
        "ontology": ontology,
        "n_added": len(added), "added": added,
        "n_removed": len(removed), "removed": removed,
        "n_modified": len(modified), "modified": modified,
        "n_unchanged": len(unchanged), "unchanged": unchanged,
    }
    out = Path("cache/delta")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{args.old}__{args.new}.json"
    out_path.write_text(json.dumps(delta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ delta {args.old} → {args.new}: "
          f"+{len(added)} -{len(removed)} ~{len(modified)} ={len(unchanged)}; "
          f"{out_path}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
