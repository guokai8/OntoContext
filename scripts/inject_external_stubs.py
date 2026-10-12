"""
inject_external_stubs.py — inject stub [Term] entries for any external
ID referenced in an OBO file but not defined in it.

For example, cl.obo references BFO:0000015 but doesn't define it
(pronto's `symmetrize_lineage` fails with KeyError). This script
appends minimal stub terms so the file parses end-to-end.

Output:
    <infile>.stub.obo  (the augmented copy, ready for step 02)

Usage:
    python3 scripts/inject_external_stubs.py data/raw/snapshots/cl_2024-06-10/cl_2024-06-10.fixed.obo
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


# Find all referenced external ids (is_a, relationship, intersection_of, etc.)
# Use a non-capturing group with proper grouping around the alternation.
REF_RE = re.compile(
    r"^(?:is_a|relationship|intersection_of|union_of|disjoint_from|property_value|synonym_type|equivalent_to|domain|range|inverse_of|transitive_over|holds_over_chain):\s*(\S+)",
    re.MULTILINE,
)


def _iter_referenced_ids(text: str) -> set[str]:
    """All IDs that are referenced (anywhere)."""
    refs: set[str] = set()
    for m in REF_RE.finditer(text, re.MULTILINE):
        rid = m.group(1)
        # `relationship: RO:0002131 CL:0000124` → take RO:0002131
        if " " in rid:
            rid = rid.split(" ")[0]
        refs.add(rid)
    return refs


def _iter_defined_ids(text: str) -> set[str]:
    """All IDs that are defined (either as [Term] or [Typedef])."""
    term_ids = set(re.findall(r"^\[Term\]\nid:\s*(\S+)", text, re.MULTILINE))
    typedef_ids = set(
        re.findall(r"^\[Typedef\]\nid:\s*(\S+)", text, re.MULTILINE)
    )
    return term_ids | typedef_ids


def _is_typedef_id(text: str, term_id: str) -> bool:
    """True if this id is referenced as a *relationship* super/sub."""
    rel_re = re.compile(
        r"^relationship:\s*" + re.escape(term_id) + r"\b", re.MULTILINE
    )
    return bool(rel_re.search(text))


# Pattern matching Typedef block headers: `[Typedef]` line followed by
# `id: <something>` and any lines (including `is_a: BSPO:0000099`).
_TYPEDEF_BLOCK_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _is_in_typedef(text: str, term_id: str) -> bool:
    """True if `term_id` appears as a super/sub of any [Typedef] block.

    We scan for `[Typedef]` headers, find the next `[Term]` or EOF, and
    check whether `term_id` appears in the `is_a:` lines within that
    block.
    """
    # Find all [Typedef] spans
    pos = 0
    while True:
        m = re.search(r"^\[Typedef\]\s*\n", text[pos:], re.MULTILINE)
        if not m:
            return False
        start = pos + m.end()
        end_m = re.search(r"^\[(Term|Typedef)\]\s*\n", text[start:], re.MULTILINE)
        end = start + end_m.start() if end_m else len(text)
        block = text[start:end]
        if re.search(r"^is_a:\s*" + re.escape(term_id) + r"\b", block, re.MULTILINE):
            return True
        if re.search(r"^inverse_of:\s*" + re.escape(term_id) + r"\b", block, re.MULTILINE):
            return True
        if re.search(r"^transitive_over:\s*" + re.escape(term_id) + r"\b", block, re.MULTILINE):
            return True
        pos = end


def make_stub(term_id: str, as_typedef: bool = False) -> str:
    """Minimal stub block for an external id."""
    if ":" not in term_id:
        return ""
    prefix, local = term_id.split(":", 1)
    label = f"{prefix} stub {local}"
    if as_typedef:
        return (
            f"\n[Typedef]\n"
            f"id: {term_id}\n"
            f"name: {label}\n"
            f"def: \"Stub relationship for {term_id} (injected).\" [OntologyStub]\n"
            f"is_transitive: false\n"
        )
    return (
        f"\n[Term]\n"
        f"id: {term_id}\n"
        f"name: {label}\n"
        f"def: \"Stub term for {term_id} (injected by inject_external_stubs).\" [OntologyStub]\n"
        f"is_obsolete: false\n"
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("usage: inject_external_stubs.py <obo-file> [<obo-file> ...]", file=sys.stderr)
        return 2
    for arg in argv:
        infile = Path(arg)
        if not infile.exists():
            print(f"  ! {infile} missing", file=sys.stderr)
            continue
        text = infile.read_text(encoding="utf-8")
        defined = _iter_defined_ids(text)
        referenced = _iter_referenced_ids(text)
        missing = sorted(referenced - defined)
        if not missing:
            print(f"  ✅ {infile.name}: no missing references; nothing to do")
            continue
        # Add stubs just before the first [Term] so the lineage
        # symmetrizer sees them before any reference to them. Typedefs
        # go just before the first [Typedef].
        term_stubs = "".join(make_stub(t) for t in missing if not _is_in_typedef(text, t))
        typedef_stubs = "".join(make_stub(t, as_typedef=True) for t in missing if _is_in_typedef(text, t))
        if term_stubs and "[Term]" in text:
            new_text = text.replace("[Term]", term_stubs + "\n[Term]", 1)
        elif term_stubs:
            new_text = text + term_stubs
        else:
            new_text = text
        if typedef_stubs and "[Typedef]" in new_text:
            new_text = new_text.replace("[Typedef]", typedef_stubs + "\n[Typedef]", 1)
        elif typedef_stubs:
            new_text = new_text + typedef_stubs
        outfile = infile.with_suffix(".stub.obo")
        new_text = re.sub(r"data-version: [^\n]+", "data-version: releases/injected_stubs", new_text, count=1)
        outfile.write_text(new_text, encoding="utf-8")
        print(f"  ✅ {infile.name} → {outfile.name}  ({len(missing)} stubs injected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
