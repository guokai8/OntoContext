"""
preprocess_obo.py — fix known pronto/fastobo parse errors in OBO files.

The cl.obo file contains a backslash-escape in an IRI:
    xref: http://neurolex.org/wiki/Category\:Embryonic_organism
which fastobo (pronto's underlying parser) rejects with:
    "expected EOL, QuotedString, RFC3987_IriPctEncoded, or RFC3987_IriUCSChar"
at line ~201561. We strip the backslash so the IRI parses cleanly.

Output:
    <infile>.fixed.obo  (the cleaned copy, ready for step 02)

Usage:
    python3 scripts/preprocess_obo.py data/raw/snapshots/cl_2024-06-10/cl_2024-06-10.obo
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Pattern: a backslash inside an unquoted IRI after `xref:` etc.
# This is the fastobo-strict violation. We strip the backslash.
BAD_ESCAPE_RE = re.compile(r"(xref:\s+\S+?)\\:(\S)")


def fix_file(infile: Path, outfile: Path) -> tuple[int, int]:
    """Return (lines_in, fixes_made)."""
    text = infile.read_text(encoding="utf-8")
    out = BAD_ESCAPE_RE.sub(r"\1\2", text)
    outfile.write_text(out, encoding="utf-8")
    fixes = text.count("\\:") - out.count("\\:")
    return text.count("\n"), fixes


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print("usage: preprocess_obo.py <obo-file> [<obo-file> ...]", file=sys.stderr)
        return 2
    total_lines, total_fixes = 0, 0
    for arg in argv:
        infile = Path(arg)
        if not infile.exists():
            print(f"  ! {infile} missing", file=sys.stderr)
            continue
        outfile = infile.with_suffix(".fixed.obo")
        lines, fixes = fix_file(infile, outfile)
        total_lines += lines
        total_fixes += fixes
        print(f"  ✅ {infile.name} → {outfile.name}  ({lines} lines, {fixes} fixes)")
    print(f"total: {total_lines} lines, {total_fixes} bad-escape fixes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
