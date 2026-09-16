#!/usr/bin/env python3
import hashlib, os, tarfile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import paths
p=paths(); final=p["reports"]/"final"/"original_baselines_summary.json"
if not final.is_file(): raise RuntimeError("Run analyze first")
output=Path(os.environ.get("RESULT_PACKAGE","/root/autodl-tmp/coreflow_applsci_original_baselines_v1_results.tar.gz"))
with tarfile.open(output,"w:gz") as tar:
    for name in ["reports","results","data"]:
        target=p["work"]/name
        if target.exists(): tar.add(target,arcname=f"coreflow_applsci_original_baselines_v1_results/{name}")
digest=hashlib.sha256()
with output.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024*1024),b""): digest.update(chunk)
h=digest.hexdigest(); checksum=Path(str(output)+".sha256"); checksum.write_text(f"{h}  {output.name}\n",encoding="ascii",newline="\n"); print(f"PACKED: {output}\nSHA256: {h}")
