#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import tarfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, load_json, paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    cfg, p = config(), paths()
    summary = p["reports"] / "final" / "prs3_pilot_summary.json"
    if load_json(summary).get("status") != "ANALYZED_PILOT_ONLY":
        raise RuntimeError("Run analyze first")
    output = Path("/root/autodl-tmp/coreflow_prs3_pilot_v1_results.tar.gz")
    sidecar = Path(str(output) + ".sha256")
    with tarfile.open(output, "w:gz") as archive:
        for relative in ("reports", "results", "tables", "data", "logs"):
            source = p["work"] / relative
            if source.exists():
                archive.add(source, arcname=relative)
        for relative in ("config/prs3_pilot_protocol.json", "evidence/PRS0_SEALED_BEFORE_NEW_GPU_OUTPUTS.json", "evidence/m3_bank_lock.json", "PACKAGE_MANIFEST.sha256", "README_AUTODL.md"):
            archive.add(ROOT / relative, arcname=f"package/{relative}")
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8", newline="\n")
    print(f"PACKED: {output}")
    print(f"SHA256: {sidecar}")


if __name__ == "__main__":
    main()
