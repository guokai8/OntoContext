"""
00_run_all.py — run the entire OntoContext pipeline.

Convenience driver: runs every step in order, in --fast mode
(smoke-sized inputs, default budgets, 4-node synth).

Output:
    console log + final pack tarball

Usage:
    python3 scripts/00_run_all.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Determinism per the plan.
np.random.seed(17)

SCRIPTS = [
    "scripts/02_normalize_graph.py",
    "scripts/step03_v0_build.py",
    "scripts/step04_v1_select.py",
    "scripts/step05_v2_discriminate.py",
    "scripts/step06_v3_modules.py",
    "scripts/step07_build_world_model.py",
    "scripts/step08_render.py",
    "scripts/step09_agent.py",
    "scripts/step10_bench_localization.py",
    "scripts/step10_a_hard_neg_sweep.py",
    "scripts/step11_bench_hierarchy.py",
    "scripts/step11_a_depth_sweep.py",
    "scripts/step12_bench_sibling.py",
    "scripts/step12_a_confusable_kind_sweep.py",
    "scripts/step13_bench_relation.py",
    "scripts/step13_a_typed_only.py",
    "scripts/step14_bench_multihop.py",
    "scripts/step14_a_path_length_sweep.py",
    "scripts/step15_bench_cross_ontology.py",
    "scripts/step15_a_pcl_added.py",
    "scripts/step16_bench_version_change.py",
    "scripts/step16_a_only_changelog.py",
    "scripts/step17_bench_unsupported.py",
    "scripts/step17_a_threshold_calibration.py",
    "scripts/step18_hard_neg_sweep.py",
    "scripts/step20_baseline_bm25.py",
    "scripts/step21_baseline_dense.py",
    "scripts/step22_baseline_graphrag.py",
    "scripts/step23_baseline_raw_neighborhood.py",
    "scripts/step24_baseline_ontology_embedding.py",
    "scripts/step30_plot_pareto.py",
    "scripts/step31_loss_decomposition.py",
    "scripts/step40_delta_graph.py",
    "scripts/step41_invalidation.py",
    "scripts/step42_claim_generator.py",
    "scripts/step50_project1_integration.py",
    "scripts/step60_mp_xref_seed.py",
    "scripts/step61_mp_xref_emit.py",
    "scripts/step70_maintain_state.py",
    "scripts/step99_pack_and_verify.py",
]

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    snap = args.synthetic or args.snapshot
    if not snap:
        print("ERROR: pass --synthetic or --snapshot", file=sys.stderr)
        return 2
    is_synthetic = bool(args.synthetic)
    if is_synthetic:
        flag = "--synthetic"
    else:
        flag = "--snapshot"
    ontology = snap.split("_")[0].upper()
    print(f"=== Running OntoContext end-to-end on {snap} ===")
    t0 = time.time()
    failures: list[tuple[str, str]] = []
    for s in SCRIPTS:
        full_s = str(ROOT / s)
        if not Path(full_s).exists():
            failures.append((s, "missing"))
            continue
        # Build a per-script argv
        argv = [full_s]
        if "step09" in s:
            argv += [flag, snap, "--family", "localization", "--n", "2",
                      "--budget", "256", "--run-id", "fast"]
        elif Path(full_s).name == "step08_render.py":
            # Interactive script — skip in batch mode
            print(f"  ⏭  {s} (interactive, skipped)")
            continue
        elif Path(full_s).name == "step10_bench_localization.py":
            argv += [flag, snap, "--run-id", "fast", "--family", "localization"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step11_bench_hierarchy.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step12_bench_sibling.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step13_bench_relation.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step14_bench_multihop.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step15_bench_cross_ontology.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name == "step16_bench_version_change.py":
            argv += ["--old", snap, "--new", snap]
        elif Path(full_s).name == "step16_a_only_changelog.py":
            argv += ["--old", snap, "--new", snap]
        elif Path(full_s).name == "step17_bench_unsupported.py":
            argv += [flag, snap, "--run-id", "fast"]
            if not is_synthetic:
                argv += ["--n-questions", "200"]
        elif Path(full_s).name in ("step40_delta_graph.py", "step41_invalidation.py",
                                    "step42_claim_generator.py", "step50_project1_integration.py"):
            argv += ["--old", snap, "--new", snap]
        elif Path(full_s).read_text().count("--synthetic") > 0:
            argv += [flag, snap]
        else:
            argv += [flag, snap]
        try:
            r = subprocess.run(["python3", "-B", "-u"] + argv, capture_output=True,
                                text=True, timeout=1800, cwd=str(ROOT))
            ok = r.returncode == 0
            if ok:
                last_line = ""
                for line in (r.stdout or "").splitlines()[::-1]:
                    if line.strip():
                        last_line = line[:80]
                        break
                print(f"  ✅ {s} ({last_line})")
            else:
                last = r.stderr.splitlines()[-1][:80] if r.stderr else "?"
                print(f"  ❌ {s} → {last}")
                failures.append((s, last))
        except subprocess.TimeoutExpired:
            print(f"  ⏱  {s} → TIMEOUT")
            failures.append((s, "timeout"))
    elapsed = time.time() - t0
    print(f"\n=== Total: {elapsed:.1f}s; {len(SCRIPTS) - len(failures)}/{len(SCRIPTS)} passed ===")
    if failures:
        print("Failures:")
        for s, why in failures:
            print(f"  - {s}: {why}")
        return 1
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
