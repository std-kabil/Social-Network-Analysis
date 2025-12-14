#!/usr/bin/env python3
r"""Concatenate report_tex/report.tex and its \\input parts into a single report_full.tex.

Usage:
  python3 report_tex/concat_report.py

Output:
  report_tex/report_full.tex

Notes:
- This script performs a simple expansion of \\input{...} statements.
- It assumes paths are relative to the directory containing report.tex (report_tex/).
"""

from __future__ import annotations

import re
from pathlib import Path

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")


def expand_inputs(text: str, base_dir: Path, visited: set[Path]) -> str:
    def repl(match: re.Match[str]) -> str:
        rel = match.group(1)
        path = (base_dir / rel).resolve()
        if path in visited:
            return f"% Skipped recursive input: {rel}\n"
        if not path.exists():
            return f"% Missing input file: {rel}\n"
        visited.add(path)
        content = path.read_text(encoding="utf-8")
        return f"% ==== BEGIN {rel} ====\n" + expand_inputs(content, path.parent, visited) + f"\n% ==== END {rel} ====\n"

    return INPUT_RE.sub(repl, text)


def main() -> None:
    root = Path(__file__).resolve().parent
    main_tex = root / "report.tex"
    out_tex = root / "report_full.tex"

    text = main_tex.read_text(encoding="utf-8")
    expanded = expand_inputs(text, base_dir=root, visited=set())
    out_tex.write_text(expanded, encoding="utf-8")
    print(f"Wrote: {out_tex}")


if __name__ == "__main__":
    main()
