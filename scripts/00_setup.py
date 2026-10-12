"""
step00_setup.py — Phase 0 (scaffold).

Implements the directory layout from `ontocontext/rules/ontocontext-repo-layout.mdc`
§2 (Universal OntoContext folder layout). Idempotent at the layout level:
running it twice with the same arguments produces the same on-disk structure.

Usage:
    python scripts/00_setup.py
        # create the layout if missing
    python scripts/00_setup.py --check
        # verify the layout matches the spec without writing; exit 1 on mismatch
    python scripts/00_setup.py --force
        # create or rebuild .gitkeep / .gitignore files even if the dirs exist

Outputs:
    - folders listed in §2 of the rule
    - `.gitignore` at the project root (cache/, _logs/, world_model/, synthesis/generated/)
    - placeholder `notes/RUN_LEDGER.md` (header only; filled by later steps)
    - placeholder `notes/decisions/README.md`
    - placeholder `bibliography/references.bib`, `bibliography/ontologies.bib`,
      `bibliography/search/.gitkeep`
    - placeholder `derivations/claims/.gitkeep`, `derivations/proofs/.gitkeep`
    - placeholder `evaluation/{benchmarks,baselines,models,reports,integration}/.gitkeep`

Capsule schema version: 0.1.0 (locked for this MVP release line — A15).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Per the rule: capsule schema is locked at 0.1.0 for the MVP release line.
# Any new field requires a schema bump + a migration script under capsules/_migrations/.
CAPSULE_SCHEMA_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Layout — exactly mirrors ontocontext-repo-layout.mdc §2.
# ---------------------------------------------------------------------------

REQUIRED_DIRS: list[str] = [
    # data
    "data/raw/snapshots",
    "data/raw/annotations",
    "data/processed",
    # embeddings / caches
    "cache/embeddings",
    "cache/ic_tables",
    "cache/lms",
    # capsules (the deliverable)
    "capsules",
    "modules",
    "world_model",
    # synthesis
    "synthesis/generated",
    "synthesis/verified",
    "synthesis/rejected",
    # numeric + visual deliverables
    "results",
    "figures",
    "tables",
    # bibliography + papers + derivations
    "bibliography/search",
    "bibliography/library",
    "papers/draft",
    "papers/revisions",
    "papers/submitted",
    "derivations/claims",
    "derivations/proofs",
    "derivations/notebooks",
    "derivations/reviews",
    # evaluation harness
    "evaluation/benchmarks",
    "evaluation/baselines",
    "evaluation/models",
    "evaluation/reports",
    "evaluation/integration",
    # notes + scripts
    "notes/decisions",
    "scripts",
    "_logs",
]


GITIGNORED_DIRS = ("cache", "_logs", "world_model", "synthesis/generated")
GITIGNORE_CONTENTS = "\n".join(
    f"{d}/" for d in sorted(GITIGNORED_DIRS)
) + "\n# python\n__pycache__/\n*.pyc\n.venv/\n\n# project scratch\n_logs/*\n!_logs/.gitkeep\n\n"


PLACEHOLDERS: list[tuple[str, str]] = [
    (
        "notes/RUN_LEDGER.md",
        "# Run ledger\n\n"
        "| Step | Phase | Status | Output | Schema-ver | Capsule-ver | Verified by | Notes |\n"
        "|------|-------|--------|--------|------------|-------------|-------------|-------|\n"
        "| 00   | scaffold | ⬜ not started | scripts/00_setup.py | n/a | n/a |  |  |\n",
    ),
    (
        "notes/decisions/README.md",
        "# Decisions\n\n"
        "Non-obvious choices, each in one MD file named `<topic>.md`.\n\n"
        "Initial decisions seeded at plan-approval time:\n\n"
        "- `route-a.md` — publication bias = ICLR / NeurIPS / ICML.\n"
        "- `ic-def.md` — V1 IC = descendant-frequency (assumption A8).\n"
        "- `parser.md` — primary parser = `pronto` (assumption A4).\n"
        "- `tokenizer.md` — `tiktoken/cl100k_base` is canonical (assumption A6).\n",
    ),
    (
        "bibliography/references.bib",
        "% references.bib — populated by step 18 (evidence-and-citations) and onward.\n",
    ),
    (
        "bibliography/ontologies.bib",
        "% ontologies.bib — every snapshot version IRI gets a BibTeX entry.\n",
    ),
    (
        "bibliography/search/.gitkeep",
        "",
    ),
    (
        "bibliography/library/.gitkeep",
        "",
    ),
    (
        "scripts/_config.py",
        "\"\"\"scripts/_config.py — project-level constants (loaded by every step).\n\n"
        "Per `ontocontext-naming.mdc` §5, this file is infra, not part of the\n"
        "numbered DAG. Every magic number lives here or is read from here.\n"
        "\"\"\"\n"
        "from __future__ import annotations\n\n"
        "# ---- Run-reproducibility (A14, A16) ----\n"
        "DEFAULT_SEED = 17\n\n"
        "# ---- V1 information-theoretic Utility(e) weights (A7) ----\n"
        "UTILITY_WEIGHTS = {\"alpha\": 0.5, \"beta\": 0.3, \"gamma\": 0.1, \"delta\": 0.1}\n\n"
        "# ---- V4 renderer Budget allocation (A7) ----\n"
        "# Reference: ontocontext-rendering-protocol.mdc §3\n"
        "RENDER_BUDGET = {\n"
        "    \"B_0\": 64,\n"
        "    \"alpha\": 256,  # multiplied by Importance (normalized IC)\n"
        "    \"beta\": 192,   # multiplied by Ambiguity\n"
        "    \"gamma\": 48,   # multiplied by Connectivity (degree normalized)\n"
        "    \"delta\": 64,   # multiplied by log(1 + annotation_count) / log(1 + max_anno)\n"
        "    \"eta\": 64,     # multiplied by ChangeRisk\n"
        "}\n\n"
        "# ---- Token budgets (A6) ----\n"
        "DEFAULT_TOKEN_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)\n"
        "CANONICAL_TOKENIZER = \"tiktoken.cl100k_base\"\n\n"
        "# ---- Capsule schema version (A15) ----\n"
        "CAPSULE_SCHEMA_VERSION = \"0.1.0\"\n\n"
        "# ---- Snapshot table (A1–A3) ----\n"
        "SNAPSHOTS = {\n"
        "    \"go_2024-06-10\":       \"http://purl.obolibrary.org/obo/go/releases/2024-06-10/go.owl\",\n"
        "    \"cl_2024-08-15\":       \"http://purl.obolibrary.org/obo/cl/releases/2024-08-15/cl.owl\",\n"
        "    \"uberon_2024-08-07\":   \"http://purl.obolibrary.org/obo/uberon/releases/2024-08-07/uberon.owl\",\n"
        "    \"materials_2024-09\":   \"https://materialsproject.org/\",\n"
        "    \"go_2024-09-16\":       \"http://purl.obolibrary.org/obo/go/releases/2024-09-16/go.owl\",\n"
        "}\n",
    ),
]


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Project root = parent of `scripts/`."""
    return Path(__file__).resolve().parent.parent


def create_layout(root: Path, force: bool = False) -> tuple[list[str], list[str]]:
    """Create required directories and placeholder files.

    Returns (created, already_existed) — used for the report.
    """
    created: list[str] = []
    existing: list[str] = []

    for rel in REQUIRED_DIRS:
        p = root / rel
        if p.exists():
            existing.append(rel)
        else:
            p.mkdir(parents=True, exist_ok=True)
            created.append(rel)

    # .gitkeep for dirs whose canonical file is not yet written
    for rel in REQUIRED_DIRS:
        if rel.endswith(("decisions", ".gitkeep")):
            continue
        p = root / rel / ".gitkeep"
        if not p.exists() or force:
            p.touch(exist_ok=True)
            created.append(f"{rel}/.gitkeep")

    # .gitignore at root
    gi = root / ".gitignore"
    if not gi.exists() or force:
        gi.write_text(GITIGNORE_CONTENTS)
        created.append(".gitignore")
    else:
        existing.append(".gitignore")

    # Placeholder contents
    for rel, content in PLACEHOLDERS:
        p = root / rel
        if p.exists() and not force:
            existing.append(rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        created.append(rel)

    return created, existing


def check_layout(root: Path) -> tuple[list[str], list[str]]:
    """Return (missing, present) — used for --check."""
    missing: list[str] = []
    present: list[str] = []
    for rel in REQUIRED_DIRS:
        if (root / rel).is_dir():
            present.append(rel)
        else:
            missing.append(rel)
    return missing, present


def main() -> int:
    parser = argparse.ArgumentParser(description="OntoContext project scaffolding")
    parser.add_argument("--check", action="store_true",
                        help="verify layout matches the rule without writing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild placeholder files even when the dir exists")
    args = parser.parse_args()

    root = project_root()

    if args.check:
        missing, present = check_layout(root)
        print(f"Project root: {root}")
        print(f"Present:      {len(present)}/{len(REQUIRED_DIRS)} required dirs")
        if missing:
            print(f"MISSING: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
            return 1
        # Also check key files
        key_files = [".gitignore", "PLAN.md"]
        missing_files = [f for f in key_files if not (root / f).exists()]
        if missing_files:
            print(f"MISSING FILES: {missing_files}")
            return 1
        print("OK.")
        return 0

    created, existing = create_layout(root, force=args.force)
    print(f"Project root: {root}")
    print(f"Created:      {len(created)} entries")
    print(f"Already present: {len(existing)} entries")

    # Post-write self-check.
    missing, present = check_layout(root)
    if missing:
        print(f"FATAL: layout post-check failed, still missing: {missing}", file=sys.stderr)
        return 1
    print(f"Post-check:   {len(present)}/{len(REQUIRED_DIRS)} required dirs present.")
    print("OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
