# OntoContext — Release Manifest (camera-ready, NeurIPS 2027)

This folder is a snapshot of the OntoContext codebase prepared for
**public release alongside the NeurIPS 2027 camera-ready submission**.
It contains the research code, the v2 (camera-ready) scripts, the
reproducibility artifacts cited in the paper, and the data-integrity
fixes. All private working documents, intermediate caches, and
submission-internal scripts have been removed.

## Headline numbers (camera-ready, post-F1/F2)

| Claim | Number | Source artifact |
|-------|--------|-----------------|
| OntoContext default-mode hallucination (n=90 across 3 models) | **0/90 (0.0 %)** | `tables/step63_model_comparison.csv` |
| GraphRAG default-mode hallucination (post-F1 fix, n=90) | 36/90 (40.0 %) | `tables/step63_model_comparison.csv` |
| Fisher exact, OntoContext vs GraphRAG hallucination | **p = 3.6 × 10⁻¹³** | scipy `fisher_exact` on `[[0,90],[36,54]]` |
| Matched-budget on CL (4-family avg @ 64–1024 tok) | **95.0 %** | `tables/step30_matched_budget_cl.csv` |
| BM25 on CL @ full corpus (~950 K tok), same 40 q | 80.0 % | `tables/step30_matched_budget_cl.csv` |
| Δ (matched-budget OntoContext vs BM25 full-corpus) | **+15.0 pp** | derived |
| OntoContext accuracy on CL, 4-family avg @ ≥ 256 tok | 96.67 % (29/30 q) | `notes/RUN_LEDGER.md` |
| v3-fix ablation collapse (CL, 4 families, budgets 8–512) | 0 % | `figures/story_01_on_off_comparison.png` |
| Token-cost reduction vs full-corpus render | 1 000–30 000× | `tables/quick_pareto_cl_2024-06-10.csv` |

## Contents (this folder)

| Path | Purpose | v2 status |
|------|---------|-----------|
| `README.md` | Project overview, install/run instructions, headline numbers | **rewritten for v2** — leads with the camera-ready table |
| `LICENSE` | MIT license | unchanged |
| `requirements.txt` | Python dependencies | unchanged |
| `MANIFEST.md` | This file | **rewritten for v2** |
| `PLAN.md` | 6-phase, 1-line-per-step execution plan | unchanged |
| `Project_2_OntoContext_Ontology_Compilation.md` | Hypothesis / MVP | unchanged |
| `Purpose_and_Feasibility_Gate.md` | Phase-1 gate | unchanged |
| `bibliography/references.bib` | **16** BibTeX entries (5 new for the camera-ready) | **updated v2** |
| `bibliography/ontologies.bib` | OBO IRIs used in the benchmarks | unchanged |
| `bibliography/library/` | Curated reference text | unchanged |
| `bibliography/search/` | Related-work archive | unchanged |
| `notes/RUN_LEDGER.md` | Reproducibility ledger (per-step pass/fail + headline numbers, post-F1/F2) | **updated v2** |
| `scripts/` | **63** Python scripts: 40 pipeline + 23 utility/sweep/aggregate | **+6 v2 scripts** |
| `tables/` | **6** camera-ready CSVs + per-model `step63_runs/` | **new in v2** |
| `figures/` | **14** camera-ready PNG + PDF (story 01–04, step30, step62) | **new in v2** |

## Scripts (`scripts/`)

| Group | Files (v2) | Purpose |
|-------|-----------|---------|
| Setup | `00_setup.py`, `00_synth_ontology.py`, `00_run_all.py`, `01_fetch_snapshot.py`, `02_normalize_graph.py` | Phase 1 — snapshot ingest and synth generation |
| Compile | `step03_v0_build.py`, `step04_v1_select.py`, `step04_v1_a_weight_sweep.py`, `step05_v2_discriminate.py`, `step05_v2_a_confusable_seed_sweep.py`, `step06_v3_modules.py`, `step06_v3_a_alt_clustering.py`, `step07_build_world_model.py` | V0 capsule, V1 information-theoretic, V2 discriminative, V3 modules, world model |
| Render | `step08_render.py`, `step08_render_a_diagnostic.py`, `step09_agent.py` | Adaptive renderer and benchmark agent |
| Benchmarks | `step10_bench_localization.py` … `step18_hard_neg_sweep.py` (16 files) | 8 task families + 8 ablations |
| Baselines | `step20_baseline_bm25.py` … `step24_baseline_ontology_embedding.py` (5 files) | BM25, dense, GraphRAG, raw-neighborhood, ontology-embedding |
| Plots | `step30_plot_pareto.py`, `step31_loss_decomposition.py` | Pareto + loss |
| **Matched-budget (v2)** | `step30_matched_budget.py`, `step30_matched_budget_cl.py` | **Camera-ready Story 2A, +15.0 pp headline** |
| **Cross-ontology (v2)** | `step62_xref_benchmark.py` | **Non-ontology benchmark, 60 % reach** |
| **LLM-agent (v2)** | `step63_llm_agent_eval.py`, `step63_aggregate_runs.py`, `step63_aggregate_models.py` | **Headline hallucination eval across 3 model families** |
| Maintain | `step40_delta_graph.py`, `step41_invalidation.py`, `step42_claim_generator.py`, `step50_project1_integration.py` | Incremental updates, AOCI invalidation, Project-1 integration |
| Integrate | `step60_mp_xref_seed.py`, `step61_mp_xref_emit.py`, `step70_maintain_state.py` | Materials-Project cross-reference seed/emit + state maintenance |
| Verify | `step99_b_budget_sweep.py`, `step99_pack_and_verify.py` | Final budget sweep + pack-and-verify |
| Utilities | `_config.py`, `add_n_questions_cap.py`, `build_synth_rich.py`, `fetch_obo.py`, `inject_external_stubs.py`, `preprocess_obo.py`, `quick_pareto_cl.py` | Project config + helpers |

## Tables (`tables/`)

| File | What |
|------|------|
| `quick_pareto_cl_2024-06-10.csv` | Pareto frontier on CL (anchor for the 1 000–30 000× token reduction) |
| `step30_matched_budget_cl.csv` | 4-family × 5-budget matched-budget sweep (the +15.0 pp headline) |
| `step62_xref_benchmark.csv` | 10-q cross-ontology xref benchmark (60 % @ 256 tok) |
| `step63_model_comparison.csv` | 3-model × 8-cell cross-model LLM-agent table (the 0/90 hallucination headline) |
| `step63_llm_agent_eval_cross_run.csv` | Aggregated cross-run statistics (n=90/cell across 3 models) |
| `step63_llm_agent_eval_cross_run_summary.txt` | Human-readable cross-run summary |
| `step63_runs/gpt_4o_mini/{run1,run_v2_clean,run_v3_patched}.csv` | gpt-4o-mini per-run archive (pre-F1, post-F1, post-F3) |
| `step63_runs/gpt_5_mini/{run1,run_v2_clean,run_v3_patched}.csv` | gpt-5-mini per-run archive |
| `step63_runs/gpt_6_astra/{run1,run_v2_clean,run_v3_patched}.csv` | gpt-6-astra per-run archive |

## Figures (`figures/`)

| File | What |
|------|------|
| `story_01_on_off_comparison.{png,pdf}` | v3-fix on vs off ablation (the 4/4 families → 0 % headline) |
| `story_02_pareto_methods.{png,pdf}` | Pareto frontier on CL (OntoContext vs 4 baselines) |
| `story_03_loss_decomposition.{png,pdf}` | Single-elbow budget curve on synth |
| `story_04_summary_dashboard.{png,pdf}` | §40 scoreboard with 11 numeric rows + 3-model LLM-agent row |
| `step30_matched_budget_cl.{png,pdf}` | Camera-ready matched-budget bar chart |
| `step62_xref_benchmark.{png,pdf}` | Camera-ready cross-ontology xref bar chart |
| `quick_pareto_cl_2024-06-10.png` | Baseline Pareto anchor |

## Quick start

```bash
pip install -r requirements.txt
python3 scripts/00_setup.py            # create the directory layout
python3 scripts/00_run_all.py --synthetic synth_d06_b05_s017
# → 40/40 pipeline scripts pass in ~21.5 s

# (Camera-ready eval requires OpenAI API key)
export OPENAI_API_KEY=...
python3 scripts/step63_llm_agent_eval.py
python3 scripts/step63_aggregate_models.py
```

For the real Cell Ontology (CL 2024-06-10):

```bash
python3 scripts/01_fetch_snapshot.py cl_2024-06-10
python3 scripts/00_run_all.py --snapshot cl_2024-06-10
```

## What is **not** in this release

- Intermediate caches (`cache/`, `_logs/`, `world_model.jsonl`)
- The OntoContext paper submission folder (`papers/submitted/OntoContext_v2/`) — this is a separate artifact
- Submission-internal scripts (`build_pdf.py`, `build_latex.sh`, `story_figures.py` in `scripts/`) — these live in the submission folder, hardcode a private path
- The CCFA review reports (`ccfa-review-reports/`) — internal audit trail
- Private working documents (planning notes, audit reports, feasibility gates, integrity-audit)
- The Cursor rule pack (`ontocontext/`) is **not** in this release because it is a developer tool, not the research contribution (it lives in a separate project; the README cross-references the three invariants it enforces)
