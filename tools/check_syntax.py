"""Parse every Python source file without importing optional GPU dependencies."""

from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.rglob("*.py"))
    failures: list[tuple[Path, Exception]] = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:  # report all malformed files in one pass
            failures.append((path, exc))
    if failures:
        for path, exc in failures:
            print(f"FAIL {path.relative_to(root)}: {exc}")
        raise SystemExit(1)
    print(f"AST_PASS files={len(files)}")


if __name__ == "__main__":
    main()
