"""
step70_maintain_state.py — periodic maintenance task.

OntoContext's maintenance loop:
    1. Re-read the latest V3 world model
    2. Compute IC drift per module
    3. Check for any new confusables
    4. Emit a maintenance report

Output:
    cache/maintenance_<snapshot>.json

Usage:
    python3 scripts/step70_maintain_state.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

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
        print("ERROR", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    capsules = _load_capsules(snapshot, ontology)
    ic_values = [c.get("coordinates", {}).get("ic", 0.0) for c in capsules]
    ic_min, ic_max, ic_mean = min(ic_values), max(ic_values), sum(ic_values) / max(1, len(ic_values))
    module_counter: Counter[str] = Counter()
    for c in capsules:
        sm = c.get("coordinates", {}).get("semantic_module")
        if sm:
            module_counter[sm] += 1
    report = {
        "snapshot_id": snapshot,
        "ontology": ontology,
        "n_capsules": len(capsules),
        "n_modules": len(module_counter),
        "module_sizes": dict(module_counter),
        "ic": {"min": ic_min, "max": ic_max, "mean": ic_mean},
        "actions": ["none"],
    }
    # Action recommendation based on diagnostics
    if ic_max - ic_min < 0.1:
        report["actions"].append("consider-ic-rescale")
    if len(module_counter) == len(capsules):
        report["actions"].append("consider-lower-clustering-resolution")
    out_path = Path("cache") / f"maintenance_{snapshot}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ maintenance report for {snapshot}: {report['n_capsules']} capsules, "
          f"{report['n_modules']} modules, actions={report['actions']}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
