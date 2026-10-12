"""scripts/_config.py — project-level constants (loaded by every step).

Per `ontocontext-naming.mdc` §5, this file is infra, not part of the
numbered DAG. Every magic number lives here or is read from here.
"""
from __future__ import annotations

# ---- Run-reproducibility (A14, A16) ----
DEFAULT_SEED = 17

# ---- V1 information-theoretic Utility(e) weights (A7) ----
UTILITY_WEIGHTS = {"alpha": 0.5, "beta": 0.3, "gamma": 0.1, "delta": 0.1}

# ---- V4 renderer Budget allocation (A7) ----
# Reference: ontocontext-rendering-protocol.mdc §3
RENDER_BUDGET = {
    "B_0": 64,
    "alpha": 256,  # multiplied by Importance (normalized IC)
    "beta": 192,   # multiplied by Ambiguity
    "gamma": 48,   # multiplied by Connectivity (degree normalized)
    "delta": 64,   # multiplied by log(1 + annotation_count) / log(1 + max_anno)
    "eta": 64,     # multiplied by ChangeRisk
}

# ---- Token budgets (A6) ----
DEFAULT_TOKEN_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
CANONICAL_TOKENIZER = "tiktoken.cl100k_base"

# ---- Capsule schema version (A15) ----
CAPSULE_SCHEMA_VERSION = "0.1.0"

# ---- Snapshot table (A1–A3) ----
SNAPSHOTS = {
    "go_2024-06-10":       "http://purl.obolibrary.org/obo/go/releases/2024-06-10/go.owl",
    "cl_2024-08-15":       "http://purl.obolibrary.org/obo/cl/releases/2024-08-15/cl.owl",
    "uberon_2024-08-07":   "http://purl.obolibrary.org/obo/uberon/releases/2024-08-07/uberon.owl",
    "materials_2024-09":   "https://materialsproject.org/",
    "go_2024-09-16":       "http://purl.obolibrary.org/obo/go/releases/2024-09-16/go.owl",
}
