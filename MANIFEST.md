# OntoContext — Release Manifest

This folder is a snapshot of the OntoContext codebase prepared for
public release. It contains only the research code and reproducibility
artifacts; all private working documents, intermediate caches, and
submission-internal scripts have been removed.

## Contents

| Path | Purpose |
|---|---|
| `README.md` | Project overview, install/run instructions, design notes |
| `LICENSE` | MIT license |
| `requirements.txt` | Python dependencies |
| `MANIFEST.md` | This file |
| `scripts/` | 58 Python scripts: 40 pipeline steps + 18 utility/sweep scripts |
| `bibliography/references.bib` | 11 BibTeX entries cited in the OntoContext paper |
| `bibliography/ontologies.bib` | Ontology IRIs used in the benchmarks |
| `notes/RUN_LEDGER.md` | Reproducibility ledger (per-step pass/fail and headline numbers) |

## Scripts (`scripts/`)

| Group | Files | Purpose |
|---|---|---|
| Setup | `00_setup.py`, `00_synth_ontology.py`, `00_run_all.py`, `01_fetch_snapshot.py`, `02_normalize_graph.py` | Phase 1 — snapshot ingest and synth generation |
| Compile | `step03_v0_build.py`, `step04_v1_select.py`, `step05_v2_discriminate.py`, `step06_v3_modules.py`, `step07_build_world_model.py` | V0 capsule, V1 information-theoretic, V2 discriminative, V3 modules, world model |
| Render | `step08_render.py`, `step08_render_a_diagnostic.py`, `step09_agent.py` | Adaptive renderer and benchmark agent |
| Benchmarks | `step10–step18` (12 files) | 8 task families + ablations (localization, hierarchy, sibling, relation, multi-hop, cross-ontology, version-change, insufficient-info) |
| Baselines | `step20–step24` (5 files) | BM25, dense, GraphRAG, raw-neighborhood, ontology-embedding |
| Plots | `step30_plot_pareto.py`, `step31_loss_decomposition.py` | Pareto frontier and consolidated loss decomposition |
| Maintain | `step40_delta_graph.py`, `step41_invalidation.py`, `step42_claim_generator.py` | Incremental updates and AOCI invalidation |
| Integrate | `step50_project1_integration.py`, `step60_mp_xref_seed.py`, `step61_mp_xref_emit.py`, `step70_maintain_state.py` | Downstream consumer integration (Project 1) and Materials-Project cross-reference seed |
| Verify | `step99_b_budget_sweep.py`, `step99_pack_and_verify.py` | Final budget sweep and pack-and-verify |
| Utilities | `_config.py`, `add_n_questions_cap.py`, `build_synth_rich.py`, `fetch_obo.py`, `inject_external_stubs.py`, `preprocess_obo.py`, `quick_pareto_cl.py` | Project config, helpers, OBO fetchers |

## Quick start

```bash
pip install -r requirements.txt
python3 scripts/00_setup.py            # create the directory layout
python3 scripts/00_run_all.py --synthetic synth_d06_b05_s017
                                   # full pipeline (synthetic ontology, ~21.5 s)
```

For the real Cell Ontology (CL 2024-06-10):

```bash
python3 scripts/01_fetch_snapshot.py cl_2024-06-10
python3 scripts/00_run_all.py --snapshot cl_2024-06-10
```

## Headline result (from the run ledger)

- OntoContext reaches 100 % on synth at every budget ≥ 64 tokens
- OntoContext reaches 96.67 % on CL (29/30, 4-family average) at every
  budget ≥ 256 tokens
- The 3.33 pp gap to the strongest baselines is paid back by a
  1 000–30 000× reduction in context tokens
- Removing the V3 fix collapses all four CL families to 0 % at
  budgets 8–512

See `notes/RUN_LEDGER.md` for the full per-step pass/fail and
per-budget tables.

## What is **not** in this release

- Intermediate caches (`cache/`, `_logs/`, `world_model.jsonl`)
- Generated outputs (`figures/`, `tables/`) — these are produced by
  running the pipeline
- Submission-internal scripts (`build_submission.py`,
  `story_figures.py`) — these hardcode a private path and were
  removed for the public release
- The Cursor rule pack (`ontocontext/`) — this is a developer
  tool, not the research contribution; it lives in a separate
  project
- The OntoContext paper submission folder
  (`papers/submitted/OntoContext_v1/`) — this is a separate artifact
  prepared for NeurIPS 2027
- Private working documents (planning notes, audit reports,
  feasibility gates)
