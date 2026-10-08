from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path


def main() -> None:
    work = Path(os.environ.get("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")).expanduser().resolve()
    package = Path(os.environ.get("PACKAGE_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_upload")).expanduser().resolve()
    out = Path(os.environ.get("RESULT_ARCHIVE", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_results.tar.gz")).expanduser().resolve()
    if not (work / "reports" / "reviewer_minimal_phasea" / "PROTOCOL_SEAL.json").exists():
        raise RuntimeError("Protocol is not sealed")
    manifest = {
        "protocol_id": "coreflow-reviewer-minimal-phasea-v1",
        "workspace": str(work),
        "included": ["reports/reviewer_minimal_phasea", "results/reviewer_minimal_phasea", "data", "logs"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        for rel in manifest["included"]:
            src = work / rel
            if src.exists():
                archive.add(src, arcname=f"reviewer_minimal_phasea/{rel}", recursive=True)
        info = tarfile.TarInfo("reviewer_minimal_phasea/RESULT_ARCHIVE_MANIFEST.json")
        raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
        info.size = len(raw)
        import io
        archive.addfile(info, io.BytesIO(raw))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    checksum = out.with_name(out.name + ".sha256")
    checksum.write_text(f"{digest}  {out.name}\n", encoding="utf-8", newline="\n")
    print(f"PACKED: {out}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
