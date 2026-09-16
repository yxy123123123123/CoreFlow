#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ALLOWED_EXTRA_PREFIXES = ("logs/", "__pycache__/", ".pytest_cache/")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def relative_files(root: Path) -> set[str]:
    values = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "PACKAGE_MANIFEST.sha256" or relative.endswith(".pyc"):
            continue
        if any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in ALLOWED_EXTRA_PREFIXES):
            continue
        values.add(relative)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = root / "PACKAGE_MANIFEST.sha256"
    if args.write:
        lines = [f"{digest(root / relative)}  {relative}" for relative in sorted(relative_files(root))]
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"OK: wrote {len(lines)} files")
        return
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, relative = line.split("  ", 1)
        expected[relative] = sha
    actual = relative_files(root)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise RuntimeError(f"Checksum file-set mismatch: missing={missing}, extra={extra}")
    failures = [
        relative
        for relative, wanted in expected.items()
        if digest(root / relative) != wanted
    ]
    if failures:
        raise RuntimeError(f"Checksum verification failed: {failures}")
    print(f"OK: verified {len(expected)} files")


if __name__ == "__main__":
    main()

