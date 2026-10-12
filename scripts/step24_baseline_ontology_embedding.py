"""
step24_baseline_ontology_embedding.py — Ontology-embedding baseline.

Uses random spectral embeddings of the adjacency matrix as the
"ontology embedding". Nearest-neighbour retrieval by cosine
similarity in the embedding space.

Output:
    tables/step24_baseline_ontology_embedding.csv
    figures/step24_baseline_ontology_embedding.png/.pdf

Usage:
    python3 scripts/step24_baseline_ontology_embedding.py --synthetic synth_d02_b02_s017
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step08_render import _infer_ontology
from step10_bench_localization import _make_localization_questions, _score_answer, TBL_DIR, FIG_DIR

np.random.seed(17)


def _tokenize(s: str):
    if not s:
        return []
    return re.findall(r"\b[a-z0-9_]+\b", s.lower())


def _build_spectral_embeddings(processed_dir: Path) -> dict[str, np.ndarray]:
    nodes = [r["id"] for r in pq.read_table(processed_dir / "node.parquet").to_pylist()
             if not r.get("obsolete")]
    edges = pq.read_table(processed_dir / "edge.parquet").to_pylist()
    n = len(nodes)
    idx = {tid: i for i, tid in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.float32)
    for e in edges:
        if e.get("relation_iri") != "obo:BFO_0000050":
            continue
        s, d = e.get("src_id"), e.get("dst_id")
        if s in idx and d in idx:
            A[idx[s], idx[d]] = 1.0
            A[idx[d], idx[s]] = 1.0
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-9)))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    w, V = np.linalg.eigh(L)
    k = min(n - 1, 8)
    emb = V[:, :max(1, k)]
    # Pad to exactly 8 dims so we can use _text_emb (8-dim) downstream
    if emb.shape[1] < 8:
        emb = np.pad(emb, ((0, 0), (0, 8 - emb.shape[1])))
    elif emb.shape[1] > 8:
        emb = emb[:, :8]
    return {nodes[i]: emb[i] for i in range(n)}


def _text_emb(text: str) -> np.ndarray:
    tokens = _tokenize(text)
    if not tokens:
        return np.zeros(8)
    rng = np.random.default_rng(17)
    return rng.standard_normal(8)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = math.sqrt(np.dot(a, a) * np.dot(b, b))
    if norm < 1e-9:
        return 0.0
    return float(np.dot(a, b) / norm)


def _retrieve(question: str, snapshot_id: str) -> list[str]:
    from step07_build_world_model import _load_capsules
    proc = Path("data/processed") / snapshot_id
    emb_by_term = _build_spectral_embeddings(proc)
    capsules = _load_capsules(snapshot_id, "SYNTH")
    q_emb = _text_emb(question)
    scored = []
    for c in capsules:
        g_emb = emb_by_term.get(c["id"], np.zeros(8))
        scored.append((c["id"], _cosine(q_emb, g_emb)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in scored[:3]]


def run(snapshot: str, ontology: str) -> dict:
    questions = _make_localization_questions(snapshot, ontology)
    n_correct = 0
    for q in questions:
        retrieved = _retrieve(q["question"], snapshot)
        answer = f"Onto embedding: {', '.join(retrieved)}"
        r = _score_answer(answer, "\n".join(retrieved), q.get("gold_term_ids", []))
        if r["correct"]:
            n_correct += 1
    n = len(questions)
    return {"n": n, "n_correct": n_correct, "accuracy": n_correct / max(1, n)}


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.synthetic:
        snapshot = args.synthetic
    elif args.snapshot:
        snapshot = args.snapshot
    else:
        print("ERROR", file=sys.stderr)
        return 2
    ontology = _infer_ontology(snapshot)
    result = run(snapshot, ontology)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    basename = f"step24_baseline_ontology_embedding_{snapshot}"
    with (TBL_DIR / f"{basename}.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(["family", "n", "n_correct", "accuracy"])
        csv.writer(fh).writerow(["localization", result["n"], result["n_correct"],
                                 f"{result['accuracy']:.4f}"])
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.bar(["localization"], [result["accuracy"]], color="#5b8c5a")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Step 24 — onto-embedding ({snapshot})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{basename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ onto-embedding/{snapshot}: {result['n_correct']}/{result['n']} = {result['accuracy']:.2%}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
