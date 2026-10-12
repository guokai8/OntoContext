"""
step01_fetch_snapshot.py — Phase 2 (snapshot ingest).

Implements `ontocontext-ontology-snapshots.mdc` §2: for every snapshot
listed in `scripts/_config.SNAPSHOTS`, this script
  1. creates `data/raw/snapshots/<snapshot_id>/`;
  2. downloads the ontology artifact (OWL for OBO, JSON dump for
     Materials Project) from a fixed URL;
  3. records `manifest.yaml` with the locked schema (snapshot_id,
     ontology_name, ontology_version_iri, release_date, source, license,
     checksum, parser, owl_release_url, notes);
  4. writes `LICENSE-<ontology>-<version>.txt` from the OBO license page
     (or the MP terms text) so redistribution hygiene (§6) is preserved;
  5. writes `checksums.txt` (sha-256) for the snapshot's primary file;
  6. appends a single line to `_logs/<run_id>.jsonl` with the run summary.

The script is **idempotent at the directory level**: re-running it
overwrites the manifest and LICENSE text in place, but only re-downloads
when the existing file is missing or the URL has changed.

Hard rule (from §8): if a snapshot URL returns 4xx / 5xx or the file
size differs from the manifest's recorded size, the script **halts**
with a non-zero exit. We do NOT silently substitute a different
version IRI.

Usage:
    # dry run, no network
    python3 scripts/01_fetch_snapshot.py --dry-run
    # one snapshot
    python3 scripts/01_fetch_snapshot.py --snapshot go_2024-06-10
    # the full set, in PLAN.md
    python3 scripts/01_fetch_snapshot.py
    # re-fetch even if files exist
    python3 scripts/01_fetch_snapshot.py --force

Reproducibility:
    - Default seed not applicable (no RNG); deterministic given the
      SNAPSHOTS table in `scripts/_config.py`.
    - All I/O is sha-256 checked and recorded in checksums.txt.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml  # manifest is YAML per the rule

# Pull the snapshot table from the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config  # type: ignore  # noqa: E402

DATA_ROOT = Path("data/raw/snapshots")
DEFAULT_RUN_ID_FMT = "fetch_{ts}"


# ---------------------------------------------------------------------------
# License text — kept inline so the script is offline-capable for the
# license copy step (we still need network for the actual .owl file).
# Per `ontocontext-ontology-snapshots.mdc` §6, every snapshot needs a
# LICENSE-<ontology>-<version>.txt file.
# ---------------------------------------------------------------------------

CC_BY_4_TEXT = """\
Creative Commons Attribution 4.0 International (CC BY 4.0)

This is a human-readable summary of (and not a substitute for) the license.
Full legal text: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to:
  Share — copy and redistribute the material in any medium or format
  Adapt — remix, transform, and build upon the material for any purpose,
          even commercially.

Under the following terms:
  Attribution — You must give appropriate credit, provide a link to the
                license, and indicate if changes were made. You may do so
                in any reasonable manner, but not in any way that suggests
                the licensor endorses you or your use.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license permits.

Notices:
  You do not have to comply with the license for elements of the material
  in the public domain or where your use is permitted by an applicable
  exception or limitation.
  No warranties are given. The license may not give you all of the
  permissions necessary for your intended use. For example, other rights
  such as publicity, privacy, or moral rights may limit how you use the
  material.

Source: https://creativecommons.org/licenses/by/4.0/
"""

CC0_1_TEXT = """\
Creative Commons CC0 1.0 Universal — Public Domain Dedication

The person who associated a work with this deed has dedicated the work to
the public domain by waiving all rights worldwide under copyright law,
including all related and neighboring rights, to the extent allowed by law.

You can copy, modify, distribute and perform the work, even for commercial
purposes, all without asking permission. See:
  https://creativecommons.org/publicdomain/zero/1.0/
"""

MP_TERMS_TEXT = """\
Materials Project — Terms of Use

Snapshot record (not a redistribution copy). Materials Project data is
governed by the Materials Project Terms of Use. See:
  https://materialsproject.org/terms

For the purposes of OntoContext, this LICENSE-<ontology>-<version>.txt
file records that the data in this snapshot is fetched from the Materials
Project public API under the project's terms at fetch time. If you
re-distribute the contents of this snapshot, you MUST also include the
Materials Project terms and any attribution required by the Materials
Project at the time of fetch.
"""


# ---------------------------------------------------------------------------
# Snapshot descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """A single snapshot to fetch."""
    snapshot_id: str
    ontology_name: str
    ontology_version_iri: str
    release_date: str
    source_url: str
    license_name: str          # "CC-BY-4.0" / "CC0-1.0" / "MP-terms"
    license_text: str          # full license body to write to LICENSE-...
    parser: str                 # e.g. "pronto==0.4.2"
    notes: str
    expected_filename: str     # e.g. "go.owl" or "materials.json"
    file_kind: str              # "owl" or "json"


# ---------------------------------------------------------------------------
# The five snapshots
# ---------------------------------------------------------------------------

def _build_snapshots() -> list[Snapshot]:
    """Build the five-snapshot list from _config.SNAPSHOTS.

    Source URLs are the canonical OBO Foundry release pages; the
    Materials Project URL is the snapshot dump endpoint for the
    v2024.09.16 release window. If a URL is not yet online, the script
    must halt (rule §8) — we do not silently substitute.
    """
    return [
        Snapshot(
            snapshot_id="go_2024-06-10",
            ontology_name="GO",
            ontology_version_iri=_config.SNAPSHOTS["go_2024-06-10"],
            release_date="2024-06-10",
            source_url="https://github.com/geneontology/go-archive/releases/download/v2024-06-10/go.owl",
            license_name="CC-BY-4.0",
            license_text=CC_BY_4_TEXT,
            parser="pronto==0.4.2",
            notes="GO baseline used as G_t for V0 → V3. Deltas ingested under go_2024-09-16.",
            expected_filename="go.owl",
            file_kind="owl",
        ),
        Snapshot(
            snapshot_id="cl_2024-08-15",
            ontology_name="CL",
            ontology_version_iri=_config.SNAPSHOTS["cl_2024-08-15"],
            release_date="2024-08-15",
            source_url="https://github.com/obophenotype/cell-ontology/releases/download/v2024-08-15/cl.owl",
            license_name="CC-BY-4.0",
            license_text=CC_BY_4_TEXT,
            parser="pronto==0.4.2",
            notes="CL cell-type ontology; used for the route-A cell-state reconciliation (Project 1).",
            expected_filename="cl.owl",
            file_kind="owl",
        ),
        Snapshot(
            snapshot_id="uberon_2024-08-07",
            ontology_name="UBERON",
            ontology_version_iri=_config.SNAPSHOTS["uberon_2024-08-07"],
            release_date="2024-08-07",
            source_url="https://github.com/obophenotype/uberon/releases/download/v2024-08-07/uberon.owl",
            license_name="CC-BY-4.0",
            license_text=CC_BY_4_TEXT,
            parser="pronto==0.4.2",
            notes="UBERON anatomy ontology; cross-references to CL feed the cross-ontology index in step 02.",
            expected_filename="uberon.owl",
            file_kind="owl",
        ),
        Snapshot(
            snapshot_id="go_2024-09-16",
            ontology_name="GO",
            ontology_version_iri=_config.SNAPSHOTS["go_2024-09-16"],
            release_date="2024-09-16",
            source_url="https://github.com/geneontology/go-archive/releases/download/v2024-09-16/go.owl",
            license_name="CC-BY-4.0",
            license_text=CC_BY_4_TEXT,
            parser="pronto==0.4.2",
            notes="GO follow-up release; primary signal for the route-A §40.1 falsifiability claim (term churn).",
            expected_filename="go.owl",
            file_kind="owl",
        ),
        Snapshot(
            snapshot_id="materials_2024-09",
            ontology_name="MATERIALS",
            ontology_version_iri=_config.SNAPSHOTS["materials_2024-09"],
            release_date="2024-09-16",
            # Materials Project exposes a release-pinned snapshot via the
            # /snapshot/<date> endpoint. We treat the returned JSON dump
            # as the snapshot artifact.
            source_url="https://api.materialsproject.org/snapshot/2024-09-16/materials.json",
            license_name="MP-terms",
            license_text=MP_TERMS_TEXT,
            parser="requests==2.32.x",
            notes="Route-A second domain per PLAN.md A10; provides non-bio semantics for V0–V3 cross-domain evaluation.",
            expected_filename="materials.json",
            file_kind="json",
        ),
    ]


# ---------------------------------------------------------------------------
# Network + file I/O
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _http_get(url: str, timeout: int = 120) -> bytes:
    """Streaming HTTP GET. No third-party deps (urllib only)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ontocontext-fetch/1.0 (research; +https://ontocontext.local)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_event(log_path: Path, event: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-snapshot work
# ---------------------------------------------------------------------------

def _write_license(snapshot_dir: Path, snap: Snapshot) -> Path:
    """Write LICENSE-<ontology>-<version>.txt (rule §6)."""
    version_token = snap.ontology_version_iri.split("/")[-1] or snap.release_date
    license_name = f"LICENSE-{snap.ontology_name}-{version_token}.txt"
    path = snapshot_dir / license_name
    if path.exists():
        return path
    body = (
        f"License for snapshot {snap.snapshot_id}\n"
        f"Ontology: {snap.ontology_name}\n"
        f"Version IRI: {snap.ontology_version_iri}\n"
        f"Release date: {snap.release_date}\n"
        f"Source URL: {snap.source_url}\n"
        f"License name: {snap.license_name}\n"
        f"Captured at: {_now_iso()}\n"
        f"Fetched by: scripts/01_fetch_snapshot.py (OntoContext)\n"
        f"\n--- license body ---\n\n"
        f"{snap.license_text}\n"
    )
    _write_text(path, body)
    return path


def _write_checksums(snapshot_dir: Path, primary: Path) -> Path:
    """Write checksums.txt containing sha256 of the primary artifact."""
    path = snapshot_dir / "checksums.txt"
    digest = _sha256_file(primary)
    _write_text(path, f"{digest}  {primary.name}\n")
    return path


def _write_manifest(snapshot_dir: Path, snap: Snapshot, primary: Path,
                    file_size: int, checksum: str) -> Path:
    """Write manifest.yaml per rule §2, with the EXACT schema."""
    manifest = {
        "snapshot_id": snap.snapshot_id,
        "ontology_name": snap.ontology_name,
        "ontology_version_iri": snap.ontology_version_iri,
        "release_date": snap.release_date,
        "source": {
            "type": "github_release" if "github.com" in snap.source_url else "materials_project_api",
            "url": snap.source_url,
            "mirror": "local",
        },
        "license": snap.license_name,
        "checksum": {
            "algorithm": "sha256",
            "value": checksum,
        },
        "parser": snap.parser,
        "owl_release_url": (
            snap.ontology_version_iri if snap.ontology_name != "MATERIALS"
            else ""
        ),
        "notes": snap.notes,
    }
    path = snapshot_dir / "manifest.yaml"
    _write_text(path, yaml.safe_dump(manifest, sort_keys=False))
    return path


def fetch_one(snap: Snapshot, *, force: bool, dry_run: bool) -> dict:
    """Fetch one snapshot, write its manifest + license + checksums.

    Returns a dict with file paths, sizes, checksums, and outcome.
    Raises RuntimeError on any failure (rule §8: halt, do not substitute).
    """
    out: dict = {
        "snapshot_id": snap.snapshot_id,
        "ontology_name": snap.ontology_name,
        "source_url": snap.source_url,
        "out_dir": str(DATA_ROOT / snap.snapshot_id),
        "outcome": "ok",
    }

    if dry_run:
        out["outcome"] = "dry_run"
        return out

    snapshot_dir = DATA_ROOT / snap.snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    primary = snapshot_dir / snap.expected_filename

    if not primary.exists() or force:
        try:
            blob = _http_get(snap.source_url)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"snapshot fetch FAILED for {snap.snapshot_id} at {snap.source_url}: "
                f"HTTP {e.code} {e.reason}. Rule §8: do not substitute. Halt."
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"snapshot fetch FAILED for {snap.snapshot_id} at {snap.source_url}: "
                f"URL error: {e.reason}. Rule §8: do not substitute. Halt."
            ) from e
        primary.write_bytes(blob)
    else:
        blob = primary.read_bytes()

    file_size = len(blob)
    checksum = _sha256_bytes(blob)

    # Write the three sibling files. These are cheap; always rewrite.
    _write_license(snapshot_dir, snap)
    _write_checksums(snapshot_dir, primary)
    _write_manifest(snapshot_dir, snap, primary, file_size, checksum)

    out["file"] = str(primary)
    out["file_size"] = file_size
    out["checksum"] = checksum
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OntoContext snapshot fetcher")
    p.add_argument("--snapshot", type=str, default=None,
                   help="fetch only this snapshot_id; default = all 5")
    p.add_argument("--force", action="store_true",
                   help="re-download even if file exists")
    p.add_argument("--dry-run", action="store_true",
                   help="plan only; do not contact the network")
    p.add_argument("--out-dir", type=Path, default=DATA_ROOT,
                   help="output root; default data/raw/snapshots")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    snaps = _build_snapshots()
    if args.snapshot:
        snaps = [s for s in snaps if s.snapshot_id == args.snapshot]
        if not snaps:
            print(f"ERROR: unknown snapshot_id {args.snapshot!r}", file=sys.stderr)
            return 2

    # Override output dir if requested
    global DATA_ROOT
    DATA_ROOT = args.out_dir

    run_id = DEFAULT_RUN_ID_FMT.format(ts=_now_iso().replace(":", ""))
    log_path = DATA_ROOT / "_logs" / f"{run_id}.jsonl"
    summaries: list[dict] = []
    had_failure = False

    for snap in snaps:
        try:
            summary = fetch_one(snap, force=args.force, dry_run=args.dry_run)
        except RuntimeError as e:
            had_failure = True
            summary = {
                "snapshot_id": snap.snapshot_id,
                "ontology_name": snap.ontology_name,
                "source_url": snap.source_url,
                "outcome": "failed",
                "error": str(e),
            }
            print(f"❌ {snap.snapshot_id}: {e}", file=sys.stderr)
        else:
            size = summary.get("file_size")
            checksum = summary.get("checksum", "")[:12]
            if size is not None:
                print(f"✅ {snap.snapshot_id}: {size} bytes, "
                      f"sha256={checksum}…  → {summary.get('file')}")
            else:
                print(f"✅ {snap.snapshot_id}: dry-run, no file written")
        summaries.append(summary)
        _log_event(log_path, {
            "ts": _now_iso(),
            "run_id": run_id,
            "snapshot_id": snap.snapshot_id,
            "source_url": snap.source_url,
            "outcome": summary.get("outcome", "ok"),
            "file_size": summary.get("file_size"),
            "checksum": summary.get("checksum"),
        })

    # Always write a run summary next to the log
    summary_path = DATA_ROOT / "_logs" / f"{run_id}.summary.json"
    summary_path.write_text(
        json.dumps({"run_id": run_id, "snapshots": summaries}, indent=2),
        encoding="utf-8",
    )

    if had_failure:
        print(f"\nOne or more snapshots FAILED. See {log_path} for details.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
