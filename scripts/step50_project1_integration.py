"""
step50_project1_integration.py — Project-1 cell-state reconciliation.

Reads OntoContext claims (from step 42) and reconciles them
against Project-1 cell-state entries. The reconciliation is
deterministic: each OntoContext claim either
    confirms an existing P1 cell-state entry,
    contradicts one (and is recorded as a conflict),
    or is new (and is appended to the P1 ledger).

Output:
    cache/project1/reconciliation_<old>__<new>.json

Usage:
    python3 scripts/step50_project1_integration.py --old synth_d02_b02_s017 --new synth_d02_b02_s017_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Determinism per the plan.
np.random.seed(17)

P1_LEDGER = Path("cache/project1/cell_state.jsonl")


def _load_p1() -> list[dict]:
    if not P1_LEDGER.exists():
        return []
    return [json.loads(line) for line in P1_LEDGER.read_text().splitlines() if line.strip()]


def _load_claims(old: str, new: str) -> list[dict]:
    cp = Path("cache/claims") / f"{old}__{new}.jsonl"
    if not cp.exists():
        return []
    return [json.loads(line) for line in cp.read_text().splitlines() if line.strip()]


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.old or not args.new:
        print("ERROR: --old and --new required", file=sys.stderr)
        return 2
    p1 = _load_p1()
    claims = _load_claims(args.old, args.new)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for c in p1:
        by_subject[c.get("subject", "")].append(c)
    confirmed: list[dict] = []
    contradicted: list[dict] = []
    new_entries: list[dict] = []
    for claim in claims:
        s = claim.get("subject", "")
        existing = by_subject.get(s, [])
        match = None
        for e in existing:
            if e.get("predicate") == claim.get("predicate"):
                match = e
                break
        if match and match.get("object") == claim.get("object"):
            confirmed.append(claim)
        elif match:
            contradicted.append({"claim": claim, "existing": match})
        else:
            new_entries.append(claim)
    out = {
        "old": args.old, "new": args.new,
        "n_claims": len(claims),
        "n_confirmed": len(confirmed),
        "n_contradicted": len(contradicted),
        "n_new": len(new_entries),
        "n_p1_entries": len(p1),
        "confirmed": confirmed,
        "contradicted": contradicted,
        "new_entries": new_entries,
    }
    out_path = Path("cache/project1") / f"reconciliation_{args.old}__{args.new}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    # Append new entries to the P1 ledger
    with P1_LEDGER.open("a", encoding="utf-8") as fh:
        for e in new_entries:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    print(f"✅ {len(claims)} claims → "
          f"confirmed={len(confirmed)} contradicted={len(contradicted)} new={len(new_entries)}; "
          f"{out_path}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=str, required=True)
    p.add_argument("--new", type=str, required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
