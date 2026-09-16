#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, load_json, now_utc, paths, sha256_file, write_json


def main() -> None:
    cfg, p = config(), paths()
    seal = p["reports"] / "PILOT_SEAL.json"
    if seal.exists():
        print(json.dumps({"status": "ALREADY_SEALED", "seal": str(seal)}))
        return
    preflight = p["reports"] / "preflight.json"
    closure = p["reports"] / "scheme_a_legacy_qualification_closure.json"
    if not preflight.is_file() or not closure.is_file():
        raise RuntimeError("Run preflight and archive-a before seal")
    if load_json(preflight).get("status") != "PASS":
        raise RuntimeError("Preflight did not pass")
    if load_json(closure).get("status") != "ARCHIVED":
        raise RuntimeError("Scheme A closure is incomplete")
    pilot_root = p["results"] / "pilot_b"
    existing = list(pilot_root.glob("*")) if pilot_root.exists() else []
    if existing:
        raise RuntimeError(f"Pilot outputs exist before seal: {[str(x) for x in existing]}")
    payload = {
        "status": "SEALED_BEFORE_PILOT_B_GPU_OUTPUTS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "protocol_sha256": sha256_file(ROOT / "config" / "prs4a_prs4p_protocol.json"),
        "pilot_data_sha256": sha256_file(ROOT / "data" / cfg["data"]["pilot_file"]),
        "preflight_sha256": sha256_file(preflight),
        "scheme_a_closure_sha256": sha256_file(closure),
        "package_manifest_sha256": sha256_file(ROOT / "PACKAGE_MANIFEST.sha256"),
        "pilot_outputs_present_at_seal": False,
        "formal164_embedded": False,
        "formal164_access_permitted": False,
        "role": cfg["pilot_B"]["role"],
        "forbidden_claims": cfg["forbidden_claims"],
    }
    write_json(seal, payload)
    print(json.dumps({"status": payload["status"], "seal": str(seal)}))


if __name__ == "__main__":
    main()
