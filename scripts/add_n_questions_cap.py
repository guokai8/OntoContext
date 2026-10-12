"""
add_n_questions_cap.py — add `--n-questions N` argument to all
OntoContext bench scripts, with a default cap of `N` per script.

This is used to cap the number of synthetic questions generated
when running on real ontologies with many thousands of terms,
so that the benchmark runs in reasonable time.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("scripts")

# (script_filename, snippet_to_insert_after_run_id)
TARGETS: list[tuple[str, str]] = [
    ("step10_bench_localization.py", ""),  # already done
    ("step11_bench_hierarchy.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
    ("step12_bench_sibling.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
    ("step13_bench_relation.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
    ("step14_bench_multihop.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
    ("step15_bench_cross_ontology.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
    ("step17_bench_unsupported.py",
     '    p.add_argument("--n-questions", type=int, default=None)\n'),
]


def add_arg(path: Path, snippet: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if "--n-questions" in text:
        return False
    # Insert before `return p.parse_args(argv)`
    new = re.sub(
        r"(    return p\.parse_args\(argv\)\n)",
        snippet + r"\1",
        text, count=1,
    )
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def add_subsample_in_main(path: Path) -> bool:
    """After `questions = _make_..._questions(...)`, add a sub-sample
    block gated on `args.n_questions`."""
    text = path.read_text(encoding="utf-8")
    if "args.n_questions" in text:
        return False
    # Find lines that build questions either via `questions = _make_*_questions(...)`
    # or `questions = gold if gold else _make_*_questions(...)`.
    text = path.read_text(encoding="utf-8")
    if "args.n_questions" in text:
        return False
    pattern = (
        r"((?:    questions = gold if gold else _make_\w+_questions\([^)]+\)\n)"
        r"|(?:    questions = _make_\w+_questions\([^)]+\)\n))"
    )
    repl = (
        r"\1"
        r"    if args.n_questions and len(questions) > args.n_questions:\n"
        r"        import random\n"
        r"        random.seed(17)\n"
        r"        questions = random.sample(questions, args.n_questions)\n"
        r"        print(f'Subsampled to {len(questions)} questions (--n-questions={args.n_questions})')\n"
    )
    new = re.sub(pattern, repl, text, count=1)
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main():
    for fname, snippet in TARGETS:
        path = ROOT / fname
        if not path.exists():
            print(f"  ! {fname} missing")
            continue
        a_ok = add_arg(path, snippet)
        s_ok = add_subsample_in_main(path)
        if a_ok or s_ok:
            print(f"  ✅ {fname}: arg_added={a_ok} subsample_added={s_ok}")
        else:
            print(f"  -- {fname}: already has --n-questions")


if __name__ == "__main__":
    main()
