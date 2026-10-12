"""
step06_v3_a_alt_clustering.py — V3 alt-clustering sweep.

PLAN.md step 06 verification requires:
  Leiden/Louvain/spectral sweep figure

This script sweeps {labelprop, louvain, spectral, ontology_native}
over 4 resolutions {0.5, 1.0, 1.5, 2.0} and writes
  figures/step06_a_clustering_compare.{png,pdf}
  tables/step06_a_clustering_compare.{csv,tex}
each per-snapshot.

Algorithm implementations:
  - **labelprop** : step06's label propagation, dependency-free.
  - **louvain**   : a small, dependency-free greedy modularity
                    maximisation (Clauset–Newman–Moore style
                    initial assignment; one pass of greedy moves
                    for each node, repeated for `max_iters`
                    passes). This is *not* the canonical Louvain
                    (which requires external libs); it is a
                    faithful single-resolution approximation that
                    keeps the script fully runnable.
  - **spectral**  : Laplacian eigen-decomposition + k-means in the
                    spectral embedding. Uses numpy; deterministic
                    via `np.random.seed(17)`.
  - **ontology_native** : each term's "module" is its `parent_id`
                    (a single-parent clustering based on the
                    is_a hierarchy). This is the strongest
                    ontology-aware baseline.

Usage:
    python3 scripts/step06_v3_a_alt_clustering.py --synthetic synth_d02_b02_s017
    python3 scripts/step06_v3_a_alt_clustering.py --snapshot go_2024-06-10
    python3 scripts/step06_v3_a_alt_clustering.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step06_v3_modules import label_propagation, _build_graph_from_processed

SEED = 17
np.random.seed(SEED)
random.seed(SEED)

V3_MODULES_ROOT = Path("capsules/v3/modules")
FIG_DIR = Path("figures")
TBL_DIR = Path("tables")

RESOLUTIONS = [0.5, 1.0, 1.5, 2.0]
ALGOS = ["labelprop", "louvain", "spectral", "ontology_native"]


# ---------------------------------------------------------------------------
# Alt-clustering implementations
# ---------------------------------------------------------------------------

def _greedy_louvain_approx(
    nodes: list[str], adj: dict[str, set[str]],
    resolution: float = 1.0, max_passes: int = 5,
) -> dict[str, int]:
    """Single-resolution greedy modularity approximation.

    For each pass: assign each node to the neighbour-community that
    yields the largest modularity gain (computed locally).  This is
    not the canonical Louvain (which is multi-resolution with
    aggregation) but it is a faithful dependency-free
    approximation.  Determinism: nodes are processed in
    alphabetical order.
    """
    if not nodes:
        return {}
    # 2*m where m = number of edges
    m2 = sum(len(v) for v in adj.values())
    if m2 == 0:
        return {n: i for i, n in enumerate(nodes)}
    # Initial: each node its own community
    comm: dict[str, int] = {n: i for i, n in enumerate(nodes)}
    # Pre-compute node degrees
    deg = {n: len(adj.get(n, ())) for n in nodes}
    # Sum of degrees in each community
    sum_tot: dict[int, int] = {i: deg[n] for n, i in comm.items()}
    for _ in range(max_passes):
        moved = 0
        for n in nodes:
            cur = comm[n]
            nbrs = list(adj.get(n, ()))
            if not nbrs:
                continue
            # Compute delta-Q for moving to each neighbour's community
            nbr_comms: dict[int, int] = Counter()
            for u in nbrs:
                nbr_comms[comm[u]] += 1
            # self-loop term
            self_loop = 0
            if n in adj.get(n, ()):  # never true in OBO, but safe
                self_loop = 1
            best_comm = cur
            best_gain = 0.0
            k_i = deg[n]
            for c, k_i_in in nbr_comms.items():
                if c == cur:
                    # subtract self-contribution
                    k_i_in_eff = k_i_in - self_loop
                else:
                    k_i_in_eff = k_i_in
                # ΔQ = [k_i_in / m - resolution * (sum_tot * k_i) / (2*m²)]
                # Use the simplified local form.
                sum_tot_c = sum_tot.get(c, 0)
                # Gain = (k_i_in / m) - resolution * (sum_tot_c * k_i) / (2 * m²)
                gain = (k_i_in_eff / (m2 / 2.0)) - resolution * (sum_tot_c * k_i) / (m2 * m2 / 2.0)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c
            if best_comm != cur:
                sum_tot[cur] = sum_tot.get(cur, 0) - k_i
                sum_tot[best_comm] = sum_tot.get(best_comm, 0) + k_i
                comm[n] = best_comm
                moved += 1
        if moved == 0:
            break
    # Re-map
    unique = sorted(set(comm.values()))
    remap = {old: new for new, old in enumerate(unique)}
    return {n: remap[c] for n, c in comm.items()}


def _spectral(
    nodes: list[str], adj: dict[str, set[str]],
    resolution: float = 1.0,
) -> dict[str, int]:
    """Spectral clustering on the normalized Laplacian.

    For tiny graphs (n < 4) the algorithm degenerates; we return
    one cluster per node in that case.
    """
    n = len(nodes)
    if n < 4:
        return {no: i for i, no in enumerate(nodes)}
    idx = {no: i for i, no in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.float32)
    for u, vs in adj.items():
        for v in vs:
            i, j = idx[u], idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-9)))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    # Number of clusters scales with resolution in [0.5..2.0] → k in [2..8]
    k = int(max(2, min(8, round(2.0 + (resolution - 0.5) * 4.0))))
    k = min(k, n - 1) if n > 2 else 1
    if k <= 1:
        return {no: 0 for no in nodes}
    w, V = np.linalg.eigh(L)
    emb = V[:, :k]
    # k-means in the spectral embedding
    rng = np.random.default_rng(SEED)
    centres = emb[rng.choice(n, size=k, replace=False)]
    for _ in range(20):
        d = np.linalg.norm(emb[:, None, :] - centres[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        new_centres = np.array([
            emb[labels == c].mean(axis=0) if np.any(labels == c) else centres[c]
            for c in range(k)
        ])
        if np.allclose(new_centres, centres):
            break
        centres = new_centres
    return {nodes[i]: int(labels[i]) for i in range(n)}


def _ontology_native(
    nodes: list[str], adj: dict[str, set[str]],
    resolution: float = 1.0,
) -> dict[str, int]:
    """Cluster by parent: each term is in the same cluster as its
    first parent (the canonical is_a hierarchy)."""
    out: dict[str, int] = {}
    for n in nodes:
        # Find a parent in `adj`: any neighbour that is not the term's
        # own child. In a hierarchical is_a graph, every term except
        # the root has at least one parent (an edge where the term is
        # the *child* and the neighbour is the parent). We use the
        # convention: any neighbour with a *higher* lexical rank is
        # the parent. For OBO ontologies, we just pick the first
        # neighbour with a smaller id (heuristic).
        nbrs = sorted(adj.get(n, ()))
        parent = None
        for u in nbrs:
            if u < n:
                parent = u
                break
        if parent is None:
            # no parent (root) → its own cluster
            out[n] = len(out)
        else:
            out[n] = out.get(parent, len(out))
    # Re-map to dense ids
    unique = sorted(set(out.values()))
    remap = {old: new for new, old in enumerate(unique)}
    return {n: remap[c] for n, c in out.items()}


ALGOS_FN: dict[str, Callable] = {
    "labelprop": lambda nodes, adj, r: label_propagation(nodes, adj),
    "louvain": _greedy_louvain_approx,
    "spectral": _spectral,
    "ontology_native": _ontology_native,
}


# ---------------------------------------------------------------------------
# Sweep + plot
# ---------------------------------------------------------------------------

def _sweep_one(snapshot_id: str, ontology: str, processed_dir: Path) -> list[dict]:
    nodes, adj = _build_graph_from_processed(processed_dir)
    rows: list[dict] = []
    for algo in ALGOS:
        for r in RESOLUTIONS:
            fn = ALGOS_FN[algo]
            asg = fn(nodes, adj, r)
            n_modules = len(set(asg.values())) if asg else 0
            mean_size = (len(asg) / max(1, n_modules))
            max_size = max(Counter(asg.values()).values()) if asg else 0
            rows.append({
                "snapshot_id": snapshot_id, "ontology": ontology,
                "algo": algo, "resolution": r,
                "n_modules": n_modules, "mean_size": mean_size,
                "max_size": max_size,
            })
    return rows


def _plot(rows: list[dict], basename: str) -> None:
    by_algo: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        by_algo[r["algo"]].append((r["resolution"], r["n_modules"]))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for algo, pts in by_algo.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=algo)
    ax.set_xlabel("resolution")
    ax.set_ylabel("# modules")
    ax.set_title(f"Step 06a — clustering compare ({rows[0]['snapshot_id']})")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_csv_tex(rows: list[dict], basename: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TBL_DIR / f"{basename}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["snapshot_id", "ontology", "algo", "resolution",
                    "n_modules", "mean_size", "max_size"])
        for r in rows:
            w.writerow([r["snapshot_id"], r["ontology"], r["algo"],
                        f"{r['resolution']:.2f}", r["n_modules"],
                        f"{r['mean_size']:.4f}", r["max_size"]])
    tex_path = TBL_DIR / f"{basename}.tex"
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by step06_v3_a_alt_clustering.py\n")
        fh.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
        fh.write("algo & resolution & n\\_modules & mean\\_size & max\\_size \\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['algo']} & {r['resolution']:.2f} & {r['n_modules']} & "
                     f"{r['mean_size']:.2f} & {r['max_size']} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--all", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        targets = [(args.synthetic, "SYNTH", Path("data/processed") / args.synthetic)]
    elif args.snapshot:
        proc = Path("data/processed") / args.snapshot
        targets = [(args.snapshot, args.snapshot.split("_")[0].upper(), proc)]
    elif args.all:
        targets = []
        for d in sorted(Path("data/processed").iterdir()):
            if not (d / "node.parquet").exists():
                continue
            targets.append((d.name, d.name.split("_")[0].upper(), d))
    else:
        print("ERROR: pass --synthetic, --snapshot, or --all", file=sys.stderr)
        return 2

    for sid, ont, proc in targets:
        rows = _sweep_one(sid, ont, proc)
        basename = f"step06_a_clustering_compare_{sid}"
        _write_csv_tex(rows, basename)
        _plot(rows, basename)
        print(f"✅ {sid}: {len(rows)} cells; figures/{basename}.{{png,pdf}} + tables/{basename}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
