#!/usr/bin/env python3
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.sha256"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


expected = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    if line.strip():
        value, name = line.split("  ", 1)
        expected[name] = value
actual = {
    path.relative_to(ROOT).as_posix(): digest(path)
    for path in ROOT.rglob("*")
    if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
}
missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
if missing or extra:
    raise SystemExit(f"Checksum file-set mismatch: missing={missing}, extra={extra}")
bad = [name for name, value in expected.items() if actual[name] != value]
if bad:
    raise SystemExit(f"Checksum verification failed: {bad}")
print(f"OK: verified {len(actual)} files")
