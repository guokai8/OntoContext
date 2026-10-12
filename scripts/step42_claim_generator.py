"""
step42_claim_generator.py — Project-1 cell-state claim generator.

Given an invalidation set, generate Project-1 cell-state claims
(e.g. "term GO:0008150 changed in v2: definition rewritten").

Each claim is a (subject, predicate, object) triple ready to
ingest into the Project-1 cell-state ledger.

Output:
    cache/claims/<old>__<new>.jsonl

Usage:
    python3 scripts/step42_claim_generator.py --old synth_d02_b02_s017 --new synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Determinism per the plan.
np.random.seed(17)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.old or not args.new:
        print("ERROR: --old and --new required", file=sys.stderr)
        return 2
    delta_path = Path("cache/delta") / f"{args.old}__{args.new}.json"
    invalid_path = Path("cache/invalidation") / f"{args.old}__{args.new}.json"
    if not delta_path.exists():
        print(f"ERROR: {delta_path} not found; run step 40 first", file=sys.stderr)
        return 2
    if not invalid_path.exists():
        print(f"ERROR: {invalid_path} not found; run step 41 first", file=sys.stderr)
        return 2
    delta = json.loads(delta_path.read_text())
    invalid = json.loads(invalid_path.read_text())
    claims: list[dict] = []
    # Claims for added terms
    for tid in delta.get("added", []):
        claims.append({
            "subject": f"ontocontext:{tid}",
            "predicate": "ontocontext:added_in",
            "object": args.new,
            "kind": "added",
            "snapshot_old": args.old,
            "snapshot_new": args.new,
        })
    # Claims for removed terms
    for tid in delta.get("removed", []):
        claims.append({
            "subject": f"ontocontext:{tid}",
            "predicate": "ontocontext:removed_in",
            "object": args.new,
            "kind": "removed",
            "snapshot_old": args.old,
            "snapshot_new": args.new,
        })
    # Claims for modified terms
    for tid, field_diffs in delta.get("modified", {}).items():
        for field, d in field_diffs.items():
            claims.append({
                "subject": f"ontocontext:{tid}",
                "predicate": f"ontocontext:{field}_changed",
                "object": json.dumps(d, sort_keys=True),
                "kind": "modified",
                "field": field,
                "snapshot_old": args.old,
                "snapshot_new": args.new,
            })
    # Cascade claims
    for tid in invalid.get("cascade_affected", []):
        claims.append({
            "subject": f"ontocontext:{tid}",
            "predicate": "ontocontext:invalidated_due_to_cascade",
            "object": args.new,
            "kind": "cascade",
            "snapshot_old": args.old,
            "snapshot_new": args.new,
        })
    out = Path("cache/claims")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{args.old}__{args.new}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for c in claims:
            fh.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"✅ {len(claims)} claims for {args.old} → {args.new} → {out_path}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
