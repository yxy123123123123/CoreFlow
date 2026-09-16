#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, load_json, now_utc, paths, sha256_file, write_json


def main():
    cfg, p = config(), paths()
    target = p["reports"] / "PROTOCOL_SEAL.json"
    if target.is_file():
        current = load_json(target)
        if current.get("status") != "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS":
            raise RuntimeError("Unexpected existing seal")
        print(json.dumps({"status": "SKIP_ALREADY_SEALED", "seal": str(target)}))
        return
    preflight = p["reports"] / "preflight.json"
    prepare = p["reports"] / "prepare.json"
    if load_json(preflight).get("status") != "PASS" or load_json(prepare).get("status") != "PASS":
        raise RuntimeError("Run successful preflight and prepare first")
    result_files = [item for item in p["results"].rglob("*") if item.is_file()]
    if result_files:
        raise RuntimeError(f"Cannot seal after pilot result files exist: {result_files[:3]}")
    package_manifest = ROOT / "PACKAGE_MANIFEST.sha256"
    payload = {
        "status": "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "pilot_only_not_formal_evidence": True,
        "new_gpu_outputs_before_seal": 0,
        "config_sha256": sha256_file(ROOT / "config" / "prs3_pilot_protocol.json"),
        "package_manifest_sha256": sha256_file(package_manifest),
        "preflight_sha256": sha256_file(preflight),
        "prepare_sha256": sha256_file(prepare),
        "parent_prs0_seal_sha256": sha256_file(ROOT / "evidence" / "PRS0_SEALED_BEFORE_NEW_GPU_OUTPUTS.json"),
        "effective_inputs": load_json(prepare)["effective_inputs"],
    }
    write_json(target, payload)
    print(json.dumps({"status": payload["status"], "seal": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
