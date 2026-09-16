#!/usr/bin/env python3
"""Phase-2B preflight: verify all frozen asset hashes before any output."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import sha256_file, load_json

PUBLIC_HASHES = {
    "model": "9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c",
    "zh_lora": "46b992598135f1f1351613d4fa50e6aad5650f24eb0ef7a80d233aceba1b83fb",
    "ru_lora": "b2e6becfbe82a70690d3632ff2028ba9c22a005658ffc04cb09e2202e1ce41c2",
    "es_lora": "00826c33ff5b61307cfcaa88ec424d263b52eb3bfb2df041698566090beae9da",
    "math_lora": "8b938cfe4941860068aebbfa0e11816c0733b3a40ff5862246f6c9440fb66e61",
    "code_lora": "929e484e57b64bb6676ac22850ad05a1f585b9952bb45f52c0463aa281c2b0c4",
}

def main():
    cfg = load_json(ROOT / "config" / "protocol.json")
    results = {"status": "PASS", "checks": []}
    m = Path("/root/autodl-tmp/Model")
    mm = m / "config.json"
    if mm.exists():
        h = sha256_file(mm)
        ok = h == PUBLIC_HASHES["model"]
        results["checks"].append({"item": "model_config", "sha256": h, "match": ok, "expected": PUBLIC_HASHES["model"]})
        if not ok: results["status"] = "FAIL"
    # verify expert order
    results["expert_order"] = cfg["expert_order"]
    # verify protocol
    results["protocol_id"] = cfg["protocol_id"]
    results["expert_ranks"] = cfg["expert_ranks"]
    # verify q values
    results["b1_q_values"] = cfg["b1_qgrid"]["q_values"]
    results["b3_configs"] = [(c["batch"], c["input_length"]) for c in cfg["b3_prefill_ttft"]["configs"]]
    out = ROOT / "reports" / "phase2b_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"status": results["status"]}, ensure_ascii=False))
    sys.exit(0 if results["status"] == "PASS" else 1)

if __name__ == "__main__":
    main()