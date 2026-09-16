#!/usr/bin/env python3
from __future__ import annotations

import json

from common import ROOT, config, load_json, now_utc, paths, sha256_file, write_json


def main() -> None:
    cfg, p = config(), paths()
    preflight_path = p["reports"] / "preflight.json"
    if not preflight_path.is_file() or load_json(preflight_path).get("status") != "PASS":
        raise RuntimeError("Run preflight first")
    forbidden = list(p["results"].rglob("*.json")) + list(p["results"].rglob("*.csv"))
    if forbidden:
        raise RuntimeError(f"Audit outputs already exist before seal: {forbidden[:5]}")
    seal_path = p["reports"] / "PRS5_SEAL.json"
    if seal_path.exists():
        print(json.dumps({"status": "ALREADY_SEALED", "seal": str(seal_path)}))
        return
    package_files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
            package_files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload = {"status": "SEALED_BEFORE_PRS5_WEIGHT_AUDIT_OUTPUTS", "protocol_id": cfg["protocol_id"],
               "timestamp_utc": now_utc(), "preflight_sha256": sha256_file(preflight_path),
               "package_files": package_files, "prohibitions": cfg["execution_contract"]}
    write_json(seal_path, payload)
    print(json.dumps({"status": payload["status"], "seal": str(seal_path)}))


if __name__ == "__main__":
    main()
