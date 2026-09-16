#!/usr/bin/env python3
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "CHECKSUMS.sha256"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

expected = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    if line.strip():
        value, name = line.split("  ", 1)
        expected[name] = value
actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}
missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
if missing or extra:
    raise SystemExit(f"Checksum file-set mismatch: missing={missing}, extra={extra}")
bad = [name for name, value in expected.items() if actual[name] != value]
if bad:
    raise SystemExit(f"Checksum verification failed: {bad}")
print(f"OK: verified {len(actual)} files")
