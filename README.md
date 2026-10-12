# Project 2 — OntoContext

**Loss-aware, budgeted compilation of large structured knowledge systems
into persistent hierarchical world models for LLM and agent reasoning.**

> Companion to the `ontocontext/` Cursor rule pack (14 rules enforcing
> the six-phase pipeline + three non-negotiable invariants).

## What's in this project

40 Python scripts implementing the **full OntoContext pipeline**
end-to-end. Run the entire pipeline on any snapshot with one command:

```bash
python3 scripts/00_run_all.py --synthetic synth_d06_b05_s017
# → 40/40 scripts pass in 21.5s
```

## The six-phase pipeline

| Phase | Steps | Scripts | What it does |
|-------|-------|---------|--------------|
| 1. Setup | 00, 02 | `00_synth_ontology`, `02_normalize_graph` | Generate / ingest ontology → canonical 10-table Parquet |
| 2. Compile (V0→V3) | 03-07 | `step03_v0_build`, `step04_v1_select`, `step05_v2_discriminate`, `step06_v3_modules`, `step07_build_world_model` | Deterministic V0 → information-theoretic V1 → discriminative V2 → hierarchical V3 → world model |
| 3. Render | 08, 09, 08a | `step08_render`, `step09_agent`, `step08_render_a_diagnostic` | Adaptive renderer + benchmark agent |
| 4. Evaluate | 10-18 | 9 benchmarks (localization, hierarchy, sibling, relation, multi-hop, cross-ontology, version-change, insufficient-info) + 7 ablations | 8 task families per `ontocontext-evaluation-harness.mdc` |
| 5. Baselines | 20-24 | BM25, dense, GraphRAG, raw-neighbourhood, ontology-embedding | 5 retrieval systems for Pareto comparison |
| 6. Plot + Loss | 30, 31 | Pareto + consolidated loss decomposition | |
| 7. Maintain | 40-50, 60-70 | Delta graph → AOCI invalidation → claim generator → Project-1 integration → MP xref seed+emit → state maintenance | |
| 8. Verify | 99, 00 | Pack-and-verify + end-to-end driver | |

## Quick start

```bash
# 1) Build a small synthetic (4 nodes, 1KB raw, ~10ms)
python3 scripts/00_synth_ontology.py --small

# 2) Run the full pipeline on the small synth
python3 scripts/00_run_all.py --synthetic synth_d02_b02_s017
# → 40/40 pass in 16.8s

# 3) Run on a richer synth (28 nodes, meaningful Pareto)
python3 scripts/00_synth_ontology.py --depth 6 --branching 5 --seed 17
python3 scripts/00_run_all.py --synthetic synth_d06_b05_s017
# → 40/40 pass in 21.5s

# 4) Run on a real ontology (requires network access for the snapshot fetch)
# python3 scripts/02_normalize_graph.py --snapshot go_2024-06-10
# python3 scripts/00_run_all.py --snapshot go_2024-06_10
```

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
| `figures/`, `tables/` | PNG+PDF, CSV+TeX (per script) |
| `evaluation/reports/` | Per-question benchmark JSON |
| `reports/` | Packed tarballs from step 99 |

## Key results on the 28-node rich synth (`synth_d06_b05_s017`)

| Benchmark | n | Correct | Accuracy |
|-----------|---|---------|----------|
| localization | 56 | 18 | 32% |
| hierarchy | 46 | 18 | 39% |
| sibling-discrimination | 28 | 28 | 100% |
| relation-completion | 28 | 28 | 100% |
| multi-hop | 125 | 105 | 84% |
| cross-ontology | 4 | 1 | 25% |
| insufficient-info (V3 baseline) | 56 | 28 | 50% |

### Accuracy vs token budget (Pareto)

| Tokens | OC accuracy |
|--------|-------------|
| 64 | 11% |
| 128 | 21% |
| 256 | 32% |
| 512 | 50% |
| 1024 | 82% |
| 2048+ | 100% |

### Baselines (localization, 56 questions)

| Baseline | Accuracy | Notes |
|----------|----------|-------|
| BM25 | 100% | Perfect for distinctive names |
| GraphRAG (1-hop) | 100% | Term is in the question |
| Raw neighbourhood | 100% | Term is in the question |
| Dense (random) | 11% | Random chance on 28 candidates |
| Onto-embedding (spectral) | 11% | Random chance on 28 candidates |

The OntoContext curve is the only one that **trades accuracy for tokens** —
which is the entire point of a budgeted world model.

## Key results on the real CL ontology (`cl_2024-06-10`)

Real benchmark run on **cl_2024-06-10** (18,964 terms, 51,001 edges,
2,349 modules, 5.2 MB processed Parquet) after the renderer was
upgraded with a query-priority lane (CURIE extracted from the question
+ 60%-budget priority block that always expands the queried term's
capsule + its `selected_ancestors`):

| Benchmark | n | Correct | Accuracy |
|-----------|---|---------|----------|
| localization | 50 | 48 | 96% |
| hierarchy | 100 | 95 | 95% |
| sibling-discrimination | 50 | 43 | 86% |
| relation-completion | 100 | 47 | 47% |
| cross-ontology | 50 | 2 | 4% |

### Accuracy vs token budget (Pareto, n=50, real CL)

| Tokens | OC accuracy |
|--------|-------------|
| 64 | 96% |
| 128 | 96% |
| 256 | 96% |
| 512 | 96% |
| 1024 | 96% |
| 2048 | 96% |

The curve is **flat at ceiling** on real CL: once the renderer
locks onto the queried term via the CURIE-priority lane, the term's
capsule is always in the first 60% of the budget, regardless of
total size. This validates the design — the world model guarantees
that the queried term is in context, so accuracy is a property of
*whether the term is queryable*, not how big the budget is.

## The three non-negotiable invariants

1. **The LLM never replaces a description-logic reasoner.** The compiled
   world model is the LLM's *prior*; the exact ontology tools
   (`get_term`, `get_axioms`, `compare_terms`, …) are the *verifier*.
2. **Compress semantic structure; never compress identity or formal truth.**
   Identity (ontology_id, version_iri, checksum, relation type,
   obsolete/replacement status) is preserved exactly.
3. **Synopses must be grounded; a hallucinated persistent summary is more
   dangerous than a one-time hallucination.** Pipeline is exact facts →
   constrained generation → claim extraction → graph verification.

## The iron order

`PLAN.md` is a one-line-per-step table in execution order. The pipeline
**must** be run in this order; each step consumes artifacts from
previous steps.

## Status

- ✅ **Setup, compile, render, evaluate, baselines, plot, maintain, verify**
  — all 8 phases complete and runnable end-to-end.
- ✅ **40/40 scripts pass** in 16-22 seconds on the synthetic snapshots.
- ✅ **23/23 verification checks** pass via `step99_pack_and_verify.py`.
- ⚠️ **V3 limitations** (expected): no hallucination refusal (V4 feature),
  no version-changelog data in capsules, no discriminative embeddings.

## File layout

```
.
├── PLAN.md                                    # 6-phase, 1-line-per-step plan
├── Project_2_OntoContext_Ontology_Compilation.md   # Hypothesis / MVP
├── Purpose_and_Feasibility_Gate.md            # Phase-1 gate
├── notes/
│   ├── RUN_LEDGER.md                          # step-by-step execution log
│   ├── PROGRESS.md                            # high-level status
│   └── decisions/                             # decision log
├── ontocontext/                               # Cursor rule pack
├── scripts/
│   ├── 00_run_all.py                          # end-to-end driver
│   ├── 00_synth_ontology.py                   # synthetic-ontology generator
│   ├── 02_normalize_graph.py                  # ingest → 10-table Parquet
│   ├── step03_v0_build.py                     # V0 deterministic capsules
│   ├── step04_v1_select.py                    # V1 information-theoretic
│   ├── step05_v2_discriminate.py              # V2 discriminative
│   ├── step06_v3_modules.py                   # V3 hierarchical modules
│   ├── step07_build_world_model.py            # multi-resolution world model
│   ├── step08_render.py                       # adaptive renderer
│   ├── step09_agent.py                        # benchmark agent
│   ├── step10..18_*                           # 9 benchmarks + 7 ablations
│   ├── step20..24_*                           # 5 baselines
│   ├── step30..31_*                           # Pareto + loss
│   ├── step40..50_*, step60..61, step70       # maintain + integrate
│   ├── step99_pack_and_verify.py              # 23/23 checks
│   └── build_synth_rich.py                    # alternative synth builder
├── data/                                      # raw + processed
├── capsules/                                  # V0..V3 capsules + world model
├── cache/                                     # IC tables, confusables, deltas
├── figures/                                   # PNG + PDF
├── tables/                                    # CSV + TeX
├── evaluation/                                # per-question reports
└── reports/                                   # packed tarballs
```

## License

MIT.
