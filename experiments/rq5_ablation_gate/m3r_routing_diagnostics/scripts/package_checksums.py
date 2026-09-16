#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED = {"PACKAGE_MANIFEST.sha256"}
EXCLUDED_DIRS = {"__pycache__", "logs"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def files(root: Path):
    for path in sorted(root.rglob("*")):
        relative_parts = path.relative_to(root).parts
        if (
            path.is_file()
            and path.name not in EXCLUDED
            and not any(part in EXCLUDED_DIRS for part in relative_parts)
            and not path.name.endswith((".pyc", ".partial"))
        ):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = root / "PACKAGE_MANIFEST.sha256"
    if args.write:
        lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in files(root)]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"OK: wrote {len(lines)} entries")
        return
    if not manifest.exists():
        raise SystemExit("Missing PACKAGE_MANIFEST.sha256")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split("  ", 1)
        expected[relative] = checksum
    actual_paths = {path.relative_to(root).as_posix(): path for path in files(root)}
    if set(expected) != set(actual_paths):
        missing = sorted(set(expected) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected))
        raise SystemExit(f"Checksum file-set mismatch: missing={missing[:10]}, extra={extra[:10]}")
    failures = [relative for relative, path in actual_paths.items() if digest(path) != expected[relative]]
    if failures:
        raise SystemExit(f"Checksum verification failed: {failures[:10]}")
    print(f"OK: verified {len(expected)} files")


if __name__ == "__main__":
    main()
