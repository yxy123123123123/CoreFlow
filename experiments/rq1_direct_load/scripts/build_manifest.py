#!/usr/bin/env python3
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'PACKAGE_MANIFEST.sha256'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(8<<20):h.update(chunk)
 return h.hexdigest()
paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=TARGET and '__pycache__' not in p.parts and not p.name.endswith('.pyc'))
TARGET.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in paths),encoding='utf-8',newline='\n')
print(f'WROTE {TARGET} files={len(paths)}')
