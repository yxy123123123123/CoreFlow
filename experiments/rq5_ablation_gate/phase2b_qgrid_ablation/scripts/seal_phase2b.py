#!/usr/bin/env python3
"""Phase-2B: protocol seal before any formal output."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import sha256_file, load_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "protocol.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dev-data", required=True)
    args = parser.parse_args()
    config_hash = sha256_file(Path(args.config))
    seal = {
        "protocol_id": load_json(Path(args.config))["protocol_id"],
        "status": "SEALED_BEFORE_PHASE2B_OUTPUTS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "dev_data_dir_sha256": sha256_file(Path(args.dev_data) / "code_dev128.jsonl"),
        "dev_math_sha256": sha256_file(Path(args.dev_data) / "math_dev128.jsonl"),
        "package_manifest_sha256": sha256_file(ROOT / "CODE_MANIFEST.sha256") if (ROOT / "CODE_MANIFEST.sha256").exists() else "not_yet_generated",
        "b1_q_values": load_json(Path(args.config))["b1_qgrid"]["q_values"],
        "b3_configs": load_json(Path(args.config))["b3_prefill_ttft"]["configs"],
        "execution_contract_hash": sha256_file(Path(args.config)),
    }
    write_json(Path(args.output), seal)
    print(json.dumps({"status": seal["status"], "seal": args.output}, ensure_ascii=False))

if __name__ == "__main__":
    main()