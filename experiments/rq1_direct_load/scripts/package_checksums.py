#!/usr/bin/env python3
import argparse,hashlib
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8<<20): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root)
    manifest=root/'PACKAGE_MANIFEST.sha256'; expected={}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if line.strip():
            sha,rel=line.split('  ',1);expected[rel]=sha
    actual={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p!=manifest and not p.name.endswith('.pyc') and '__pycache__' not in p.parts}
    if set(expected)!=set(actual): raise SystemExit(f"Checksum file-set mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}")
    bad=[rel for rel in expected if expected[rel]!=actual[rel]]
    if bad: raise SystemExit(f"Checksum verification failed: {bad}")
    print(f"OK: verified {len(actual)} files")
if __name__=='__main__':main()
