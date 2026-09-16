#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "prs3_pilot_protocol.json").read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert CFG["protocol_id"] == "coreflow-prs3-pilot-v1"
    assert CFG["protocol_status"] == "PILOT_INSTRUMENTATION_VALIDATION_NOT_FORMAL_EVIDENCE"
    assert list(CFG["methods"]) == [
        "full_legacy_original",
        "coreflow_q185_frozen",
        "isvd_legacy_padded_matched_q185",
    ]
    serialized = json.dumps(CFG, sort_keys=True).lower()
    assert "full_vectorized" not in serialized
    assert "isvd_compact" not in serialized
    workloads = CFG["prs3a"]["workloads"]
    assert [item["name"] for item in workloads] == [
        "w0_b1_l128_o64",
        "w1_b1_l512_o128",
        "w3_b1_l2048_o128",
        "w4_b1_l512_o256",
        "w5_b4_l512_o128",
        "w6_b8_l512_o128",
    ]
    assert sum(len(item["orders"]) * 3 for item in workloads) == 24
    anchor = next(item for item in workloads if item["name"] == CFG["prs3a"]["anchor"])
    for method in CFG["methods"]:
        assert sorted(order.index(method) + 1 for order in anchor["orders"]) == [1, 2, 3]
    assert list(CFG["prs3b"]["pools"]) == ["k3_strict", "k5_formal", "k8_strict"]
    assert CFG["prs3b"]["token_blocks"] == [1, 32, 128, 512]
    assert len(CFG["prs3b"]["modules"]) == 7
    assert CFG["prs3b"]["warmup"] == 3 and CFG["prs3b"]["iterations"] == 10
    assert CFG["prs3b"]["cv_total_launches"] == 3
    assert CFG["execution_contract"]["no_automatic_retry"] is True
    assert CFG["execution_contract"]["single_gpu_serial"] is True
    assert CFG["execution_contract"]["true_effective_input_lengths"] is True
    parent = {
        "prs0_seal": ROOT / "evidence" / "PRS0_SEALED_BEFORE_NEW_GPU_OUTPUTS.json",
        "prs0_assets": ROOT / "evidence" / "prs0_assets.json",
        "m3_bank_lock": ROOT / "evidence" / "m3_bank_lock.json",
    }
    for name, path in parent.items():
        assert sha256(path) == CFG["parent_evidence"][f"{name}_sha256"]
    runtime = (ROOT / "coreflow" / "runtime.py").read_text(encoding="utf-8")
    assert 'torch.einsum("...r,tlr,...t->...l", z, cores, weights.to(z.dtype))' in runtime
    server = (ROOT / "scripts" / "serve_method.py").read_text(encoding="utf-8")
    assert 'kwargs["min_new_tokens"] = args.output_length' in server
    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(json.dumps({"status": "PASS", "tests": ["protocol_scope", "prs3a_matrix", "balanced_anchor_orders", "prs3b_matrix", "parent_evidence", "formal_coreflow_operator", "fixed_output", "single_gpu_no_retry", "python_syntax"]}))


if __name__ == "__main__":
    main()
