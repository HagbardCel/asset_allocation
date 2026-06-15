#!/usr/bin/env python3
"""Clear outputs from Jupyter notebooks without strict schema validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def clear_notebook(path: Path) -> bool:
    """Return True if the notebook was modified."""
    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if not changed:
        return False

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: clear-notebook-outputs.py NOTEBOOK [NOTEBOOK ...]", file=sys.stderr)
        return 2

    changed_any = False
    for arg in argv[1:]:
        path = Path(arg)
        if clear_notebook(path):
            changed_any = True
            print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
