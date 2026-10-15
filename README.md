# OntoContext

**Loss-aware, budgeted compilation of large structured knowledge systems
into persistent hierarchical world models for LLM and agent reasoning.**

> Camera-ready release for the NeurIPS 2027 submission
> *OntoContext: Loss-Aware, Budgeted Compilation of Large Structured
> Knowledge Systems into Persistent Hierarchical World Models for
> LLM Reasoning* (https://github.com/guokai8/OntoContext).

## Headline results (camera-ready, n=90/cell × 3 model families)

OntoContext is the only system in the comparison that **prevents
CURIE hallucination** while staying within a 64–256-token render
budget. The defaults below are reproducible from the bundled CSVs in
`tables/` and the script in `scripts/step63_llm_agent_eval.py`.

| Claim | Number | Source |
|-------|--------|--------|
| OntoContext default-mode accuracy on CL (n=90 across 3 models) | **100.0 % ± 0.0 %** | `tables/step63_llm_agent_eval_cross_run.csv` |
| OntoContext default-mode hallucination (0/90, 3 models) | **0.0 %** | `tables/step63_model_comparison.csv` |
| GraphRAG default-mode hallucination (36/90, post-F1 fix) | **40.0 %** | `tables/step63_model_comparison.csv` |
| Fisher exact, OntoContext vs GraphRAG hallucination (0/90 vs 36/90) | **p = 3.6 × 10⁻¹³** | scipy `fisher_exact` on `[[0,90],[36,54]]` |
| Matched-budget on CL (4-family avg @ 64–1024 tok) | **95.0 %** | `tables/step30_matched_budget_cl.csv` |
| BM25 on CL @ full corpus (~950 K tok), same 40 q | 80.0 % | `tables/step30_matched_budget_cl.csv` |
| Δ (matched-budget OntoContext vs BM25 full-corpus) | **+15.0 pp** | derived |
| OntoContext accuracy on CL, 4-family avg @ ≥ 256 tok | **96.67 %** (29/30 q) | `notes/RUN_LEDGER.md` |
| v3-fix ablation collapse (CL, 4 families, budgets 8–512) | **0 %** | `figures/story_01_on_off_comparison.png` |
| Token-cost reduction vs full-corpus render | **1 000–30 000×** | `tables/quick_pareto_cl_2024-06-10.csv` |

The hallucination gap (0 % vs 40 %) is **decisive** at the p < 10⁻¹²
level and is preserved across all three model families tested
(gpt-4o-mini, gpt-5-mini, gpt-6-astra), which is the evidence that
the OntoContext advantage is a property of the **rendered context**,
not the model that reads it.

## What's in this project

63 Python scripts implementing the **full OntoContext pipeline**
end-to-end, plus the camera-ready CSVs and figures cited in the
paper. Run the entire pipeline on any snapshot with one command:

```bash
pip install -r requirements.txt
python3 scripts/00_setup.py
python3 scripts/00_run_all.py --synthetic synth_d06_b05_s017
# → 40/40 pipeline scripts pass in ~21.5 s on synth
```

For the real Cell Ontology (CL 2024-06-10, 18 964 terms, 51 001 edges):

```bash
python3 scripts/01_fetch_snapshot.py cl_2024-06-10
python3 scripts/00_run_all.py --snapshot cl_2024-06-10
# → full CL sweep runs in ~22 min
```

The headline LLM-agent numbers are produced by the v2 camera-ready scripts
`step63_llm_agent_eval.py` + `step63_aggregate_models.py`, which
require an OpenAI API key. See "Reproducing the camera-ready
numbers" below.

## The eight-phase pipeline

| Phase | Steps | Scripts | What it does |
|-------|-------|---------|--------------|
| 1. Setup | 00, 02 | `00_synth_ontology`, `02_normalize_graph` | Generate / ingest ontology → canonical 10-table Parquet |
| 2. Compile (V0→V3) | 03-07 | `step03_v0_build`, `step04_v1_select`, `step05_v2_discriminate`, `step06_v3_modules`, `step07_build_world_model` | Deterministic V0 → information-theoretic V1 → discriminative V2 → hierarchical V3 → world model |
| 3. Render | 08, 09, 08a | `step08_render`, `step09_agent`, `step08_render_a_diagnostic` | Adaptive renderer + benchmark agent |
| 4. Evaluate | 10-18 | 9 benchmarks (localization, hierarchy, sibling, relation, multi-hop, cross-ontology, version-change, insufficient-info) + 7 ablations | 8 task families per `ontocontext-evaluation-harness.mdc` |
| 5. Baselines | 20-24 | BM25, dense, GraphRAG, raw-neighbourhood, ontology-embedding | 5 retrieval systems for Pareto comparison |
| 6. Plot + Loss | 30, 31 | `step30_plot_pareto`, `step31_loss_decomposition` + `step30_matched_budget*` (camera-ready) | Pareto, loss decomposition, matched-budget sweep |
| 7. Maintain | 40-50, 60-70 | Delta graph → AOCI invalidation → claim generator → Project-1 integration → MP xref seed+emit → state maintenance | |
| 8. Verify | 99, 00 | Pack-and-verify + end-to-end driver | |
| 9. **Camera-ready eval (v2)** | 62, 63 | `step62_xref_benchmark`, `step63_llm_agent_eval` + `step63_aggregate_{runs,models}` | Cross-ontology xref + LLM-agent end-to-end eval (3 model families) |

## Reproducing the camera-ready numbers

Every headline number in the table above is reproducible from the
bundled artifacts in `tables/` and `figures/`, without re-running
the LLM-agent eval (which costs OpenAI credits). To re-run from
scratch:

```bash
# (a) Matched-budget sweep on CL (the +15.0 pp headline)
python3 scripts/step30_matched_budget_cl.py
#   → tables/step30_matched_budget_cl.csv
#   → figures/step30_matched_budget_cl.png / .pdf

# (b) Cross-ontology xref benchmark (the 60% non-ontology headline)
python3 scripts/step62_xref_benchmark.py
#   → tables/step62_xref_benchmark.csv
#   → figures/step62_xref_benchmark.png / .pdf

# (c) LLM-agent end-to-end evaluation (the 0/90 hallucination headline)
export OPENAI_API_KEY=...
python3 scripts/step63_llm_agent_eval.py
python3 scripts/step63_aggregate_runs.py
python3 scripts/step63_aggregate_models.py
#   → tables/step63_runs/<model>/run_v3_patched.csv (3 model families)
#   → tables/step63_llm_agent_eval_cross_run.csv
#   → tables/step63_model_comparison.csv
#   → figures/step63_llm_agent_eval_model_comparison.png / .pdf
```

The default LLM-agent protocol is **gpt-4o-mini / gpt-5-mini /
gpt-6-astra** at n=30 questions × 4 conditions (OntoContext,
full-corpus, BM25, GraphRAG) × 2 modes (default, hard) per model
for n=90/cell across the 3 model families. Hard mode hides the
queried CURIE from the prompt; the OntoContext render is a
`tool_call` payload the model can read out of context.

## Outputs (per snapshot)

| Path | Contents |
|------|----------|
| `data/processed/<snap>/` | 10-table Parquet (node, edge, axiom, definition, synonym, xref, obsolete_replacement, annotation, cross_ontology_index, version) + manifest + checksums |
| `capsules/v0..v3/entities/<ont>/` | V0→V3 capsules (JSON) |
| `capsules/v3/modules/` | V3 module assignments + `module_summary.json` |
| `capsules/v3/world_model.jsonl` | Multi-resolution supernode graph (module + entity) |
| `capsules/v3/dependencies.jsonl` | AOCI dependency records |
| `cache/ic_tables/` | Laplace-smoothed descendant-count IC |
| `cache/confusables/` | V2 confusables (graph + lexical Jaccard) |
| `cache/delta/`, `cache/invalidation/`, `cache/claims/` | Version-diff artifacts |
| `cache/project1/` | Project-1 cell-state reconciliation |
| `figures/`, `tables/` | PNG+PDF, CSV+TeX (per script) — camera-ready figures pre-bundled in `figures/` |
| `evaluation/reports/` | Per-question benchmark JSON |
| `reports/` | Packed tarballs from step 99 |

## The three non-negotiable invariants

1. **The LLM never replaces a description-logic reasoner.** The compiled
   world model is the LLM's *prior*; the exact ontology tools
   (`get_term`, `get_axioms`, `compare_terms`, …) are the *verifier*.
2. **Compress semantic structure; never compress identity or formal truth.**
   Identity (CURIE, version IRI, relation type, obsolete / replacement
   status) is preserved exactly.
3. **(V4 contract) Synopses must be grounded.** A hallucinated persistent
   summary is more dangerous than a one-time hallucination. The V4
   pipeline (out of scope for the camera-ready paper) is
   exact facts → constrained generation → claim extraction → graph
   verification → accepted synopsis (or none), with verification
   mandatory.

## Status

- ✅ **All 9 phases complete and runnable end-to-end** (40 pipeline + 23 utility scripts).
- ✅ **63 Python scripts** in `scripts/` (40 pipeline + 23 v2 utility / sweep / aggregate).
- ✅ **Camera-ready CSVs and figures pre-bundled** in `tables/` and `figures/` — a reviewer can verify the headline numbers without re-running the LLM-agent eval.
- ✅ **40/40 pipeline scripts pass** in 16-22 seconds on the synthetic snapshots.
- ✅ **23/23 verification checks** pass via `step99_pack_and_verify.py`.
- ✅ **All cited file paths exist in this release** (verified at release time; see `notes/RUN_LEDGER.md` for the per-step pass/fail and headline numbers).
- ⚠️ **V3 limitations** (expected, called out in the paper §8): no
  hallucination refusal (V4 feature), no discriminative embeddings,
  cross-ontology limited to bundled OBO Foundry xrefs (60 % reach,
  the remaining 4/10 are outside the snapshot, not a renderer limit).

## The iron order

`PLAN.md` is a one-line-per-step table in execution order. The pipeline
**must** be run in this order; each step consumes artifacts from
previous steps.

## File layout (this release)

```
.
├── README.md                                # this file
├── LICENSE                                  # MIT
├── MANIFEST.md                              # release manifest
├── requirements.txt                         # Python deps
├── PLAN.md                                  # 6-phase, 1-line-per-step plan
├── Project_2_OntoContext_Ontology_Compilation.md
├── Purpose_and_Feasibility_Gate.md          # Phase-1 gate
├── bibliography/
│   ├── references.bib                       # 16 entries (5 new for the camera-ready)
│   ├── ontologies.bib                       # OBO IRIs used in the benchmarks
│   ├── library/                             # curated reference text
│   └── search/                              # related-work archive
├── notes/
│   └── RUN_LEDGER.md                        # step-by-step execution log (camera-ready)
├── ontocontext/                             # Cursor rule pack (14 rules, 3 invariants)
├── scripts/                                 # 63 Python scripts (see table above)
├── tables/                                  # 6 camera-ready CSVs + per-model step63_runs/
├── figures/                                 # 14 camera-ready PNG + PDF (story 01–04, step30, step62)
└── evaluation/, reports/, data/, cache/, capsules/, world_model.jsonl
                                              # produced at first run
```

## License

MIT.
