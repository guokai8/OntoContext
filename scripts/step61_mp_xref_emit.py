"""
step61_mp_xref_emit.py — Materials-Project cross-ontology relations.

For each MP-xref candidate (from step 60), synthesize a *placeholder*
MP xref relation (since the real MP ontology is not bundled in this
project). The placeholder is marked with a "_seed": true flag so
the user knows it is a seed, not a real relation.

Output:
    cache/mp_xref_relations.jsonl

Usage:
    python3 scripts/step61_mp_xref_emit.py --synthetic synth_d02_b02_s017
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
    cand_path = Path("cache/mp_xref_candidates.json")
    if not cand_path.exists():
        print(f"ERROR: {cand_path} not found; run step 60 first", file=sys.stderr)
        return 2
    candidates = json.loads(cand_path.read_text())["candidates"]
    out_path = Path("cache/mp_xref_relations.jsonl")
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for c in candidates:
            rel = {
                "subject": c["term_id"],
                "object_id": f"MP:SEED:{c['term_id']}",
                "object_ontology": "MP",
                "relation": "ontocontext:cross_domain_placeholder",
                "_seed": True,
                "snapshot_id": c.get("snapshot_id"),
            }
            fh.write(json.dumps(rel, sort_keys=True) + "\n")
            n += 1
    print(f"✅ {n} MP xref relations emitted → {out_path}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
