#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    manifest = root / "PACKAGE_MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, rel = line.split("  ", 1)
            expected[rel] = sha
    actual = {
        path.relative_to(root).as_posix(): digest(path)
        for path in root.rglob("*")
        if path.is_file() and path != manifest and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    }
    if set(expected) != set(actual):
        raise SystemExit(f"Checksum file-set mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
    bad = [rel for rel in expected if expected[rel] != actual[rel]]
    if bad:
        raise SystemExit(f"Checksum verification failed: {bad}")
    print(f"OK: verified {len(actual)} files")


if __name__ == "__main__":
    main()
