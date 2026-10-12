"""
step60_mp_xref_seed.py — Materials-Project cross-domain expansion seed.

Cross-domain step 60: given an OntoContext snapshot, find terms
that have *no* cross-ontology relations and tag them as
"candidates for Materials-Project (MP) xref expansion".

Output:
    cache/mp_xref_candidates.json

Usage:
    python3 scripts/step60_mp_xref_seed.py --synthetic synth_d02_b02_s017
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


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR: pass --synthetic or --snapshot", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    capsules = _load_capsules(snapshot, ontology)
    candidates = []
    for c in capsules:
        cross = c.get("cross_ontology", {}).get("relations", []) or []
        if not cross:
            candidates.append({
                "term_id": c["id"],
                "name": c.get("name"),
                "snapshot_id": snapshot,
                "ontology": ontology,
                "ic": c.get("coordinates", {}).get("ic"),
            })
    out = {"snapshot_id": snapshot, "ontology": ontology,
           "n_candidates": len(candidates), "candidates": candidates}
    out_path = Path("cache/mp_xref_candidates.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ {len(candidates)} MP xref candidates for {snapshot} → {out_path}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
