#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from common import ROOT, config, paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg, p = config(), paths()
    final = p["reports"] / "final" / "prs4a_prs4p_summary.json"
    if not final.is_file():
        raise RuntimeError("Run analyze first")
    destination = Path("/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_results.tar.gz")
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    if destination.exists():
        destination.unlink()
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("reports", "results", "logs"):
            source = p["work"] / name
            if source.exists():
                archive.add(source, arcname=f"workspace/{name}")
        for name in ("config", "data", "evidence"):
            archive.add(ROOT / name, arcname=f"package/{name}")
        for name in ("PACKAGE_MANIFEST.sha256", "README_AUTODL.md"):
            archive.add(ROOT / name, arcname=f"package/{name}")
    digest = sha256(destination)
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8", newline="\n")
    print(f"PACKED: {destination}")
    print(json.dumps({"status": "PACKED", "sha256": digest, "protocol_id": cfg["protocol_id"]}))


if __name__ == "__main__":
    main()
