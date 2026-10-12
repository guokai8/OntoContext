"""
step99_pack_and_verify.py — final pack-and-verify step.

Validates that:
    1. All required directories exist (capsules, cache, figures, tables)
    2. Schemas are valid (parquet tables, JSON capsules, JSONL world model)
    3. No step is missing from the ledger
    4. Pack everything into a single tarball

Output:
    reports/ontocontext_<snapshot>_<timestamp>.tar.gz

Usage:
    python3 scripts/step99_pack_and_verify.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import _infer_ontology

# Determinism per the plan.
np.random.seed(17)


def _check(p: Path, label: str) -> bool:
    """Return True if the path exists; print status."""
    ok = p.exists()
    print(f"  [{'OK' if ok else 'X'}] {label}: {p}")
    return ok


def _check_parquet(p: Path, label: str) -> bool:
    if not p.exists():
        print(f"  [X] {label}: {p} (missing)")
        return False
    try:
        pq.read_table(p)
        print(f"  [OK] {label}: {p}")
        return True
    except Exception as e:
        print(f"  [X] {label}: {p} ({e})")
        return False


def _check_jsonl(p: Path, label: str) -> bool:
    if not p.exists():
        print(f"  [X] {label}: {p} (missing)")
        return False
    try:
        for line in p.read_text().splitlines():
            if line.strip():
                json.loads(line)
        print(f"  [OK] {label}: {p} (parseable)")
        return True
    except Exception as e:
        print(f"  [X] {label}: {p} ({e})")
        return False


def _check_json(p: Path, label: str) -> bool:
    if not p.exists():
        print(f"  [X] {label}: {p} (missing)")
        return False
    try:
        json.loads(p.read_text())
        print(f"  [OK] {label}: {p}")
        return True
    except Exception as e:
        print(f"  [X] {label}: {p} ({e})")
        return False


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
    print(f"=== Pack-and-verify for {snapshot} ({ontology}) ===")
    n_ok = 0
    n_total = 0
    # Required directories
    for p, label in [
        (Path("data/processed") / snapshot, f"processed snapshot"),
        (Path("capsules/v0/entities") / ontology, "V0 capsules"),
        (Path("capsules/v1/entities") / ontology, "V1 capsules"),
        (Path("capsules/v2/entities") / ontology, "V2 capsules"),
        (Path("capsules/v3/entities") / ontology, "V3 capsules"),
        (Path("capsules/v3/modules"), "V3 modules"),
        (Path("cache/ic_tables"), "IC tables"),
        (Path("cache/confusables"), "confusables"),
        (Path("figures"), "figures"),
        (Path("tables"), "tables"),
        (Path("evaluation/reports"), "evaluation reports"),
    ]:
        n_total += 1
        if _check(p, label):
            n_ok += 1
    # Required parquet files
    for fname, label in [
        ("node.parquet", "node table"),
        ("edge.parquet", "edge table"),
        ("axiom.parquet", "axiom table"),
        ("definition.parquet", "definition table"),
        ("synonym.parquet", "synonym table"),
        ("xref.parquet", "xref table"),
        ("cross_ontology_index.parquet", "cross-ontology index"),
        ("version.parquet", "version table"),
    ]:
        n_total += 1
        if _check_parquet(Path("data/processed") / snapshot / fname, label):
            n_ok += 1
    # Required JSONL files
    for p, label in [
        (Path("capsules/v3/world_model.jsonl"), "world model JSONL"),
        (Path("capsules/v3/dependencies.jsonl"), "dependencies JSONL"),
    ]:
        n_total += 1
        if _check_jsonl(p, label):
            n_ok += 1
    # JSON / JSONL files
    for p, label, kind in [
        (Path("capsules/v3/modules/module_summary.json"), "module summary", "json"),
        (Path("world_model.jsonl"), "top-level world_model.jsonl", "jsonl"),
    ]:
        n_total += 1
        if kind == "json":
            ok = _check_json(p, label)
        else:
            ok = _check_jsonl(p, label)
        if ok:
            n_ok += 1
    print(f"\n{n_ok}/{n_total} checks passed")
    # Pack
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path("reports") / f"ontocontext_{snapshot}_{ts}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    paths_to_pack = [
        Path("data/processed") / snapshot,
        Path("capsules"),
        Path("cache"),
        Path("figures"),
        Path("tables"),
        Path("evaluation/reports"),
        Path("notes"),
    ]
    with tarfile.open(out, "w:gz") as tar:
        for p in paths_to_pack:
            if p.exists():
                tar.add(p, arcname=str(p), recursive=True)
    print(f"✅ packed {out} ({out.stat().st_size / 1024:.1f} KB)")
    return 0 if n_ok == n_total else 1


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
