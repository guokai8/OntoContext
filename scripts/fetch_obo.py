"""
fetch_obo.py — fetch a real OBO ontology from purl.obolibrary.org.

Downloads the .obo file to data/raw/snapshots/<snap_id>/<snap_id>.obo
and writes a minimal manifest.yaml that step 02 needs.

Output:
    data/raw/snapshots/<snap_id>/<snap_id>.obo
    data/raw/snapshots/<snap_id>/manifest.yaml

Usage:
    python3 scripts/fetch_obo.py --snap cl_2024-06-10 --onto CL --url http://purl.obolibrary.org/obo/cl/cl.obo
    python3 scripts/fetch_obo.py --snap go_2024-06-10 --onto GO --url http://purl.obolibrary.org/obo/go/go.obo
    python3 scripts/fetch_obo.py --snap uberon_2024-06-10 --onto UBERON --url http://purl.obolibrary.org/obo/uberon/uberon.obo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

RAW_ROOT = Path("data/raw/snapshots")


def _download(url: str, dest: Path) -> tuple[int, str]:
    """Download url to dest, return (size_bytes, sha256)."""
    req = urllib.request.Request(url, headers={"User-Agent": "ontocontext/0.1"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    dest.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    return len(data), h


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--snap", type=str, required=True,
                   help="snapshot id, e.g. go_2024-06-10")
    p.add_argument("--onto", type=str, required=True,
                   help="ontology name, e.g. GO, CL, UBERON")
    p.add_argument("--url", type=str, required=True,
                   help="URL of the .obo file")
    p.add_argument("--license", type=str, default="CC-BY-4.0")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out_dir = RAW_ROOT / args.snap
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obo = out_dir / f"{args.snap}.obo"
    if out_obo.exists():
        print(f"⚠️ {out_obo} already exists; skip download")
    else:
        print(f"downloading {args.url} → {out_obo}")
        size, sha = _download(args.url, out_obo)
        print(f"  {size/1024/1024:.1f} MB, sha256={sha[:16]}…")
    manifest = {
        "snapshot_id": args.snap,
        "ontology_name": args.onto,
        "primary_filename": f"{args.snap}.obo",
        "license": args.license,
        "source_url": args.url,
        "release_date": args.snap.split("_")[-1] if "_" in args.snap else "2024-06-10",
        "checksum_alg": "sha256",
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
    )
    print(f"✅ manifest written → {out_dir / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
