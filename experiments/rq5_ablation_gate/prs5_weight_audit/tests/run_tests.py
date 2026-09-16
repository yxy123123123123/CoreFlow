#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "config" / "prs5_protocol.json").read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == "coreflow-prs5-weight-audit-v1"
    assert list(cfg["variants"]) == [
        "A0_bilateral_normed_joint", "A1_bilateral_no_norm", "A2_input_only", "A3_output_only",
        "A4_plain_concat_svd", "A5_gate_frequency_weighted", "A6_random_orthogonal"]
    contract = cfg["execution_contract"]
    assert contract["no_generation"] and contract["no_training"] and contract["no_q_selection"] and contract["no_main_method_change"]
    evidence = json.loads((ROOT / "evidence" / "prs5_local_evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "LOCAL_EVIDENCE_INTEGRATED_AWAITING_WEIGHT_AUDIT"
    assert len(evidence["rows"]) == 7
    assert [row["dev_correct"] for row in evidence["rows"]] == [13, 15, 15, 18, None, None, None]
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and (path.suffix.lower() == ".jsonl" or "generate" in path.name.lower() or "serve" in path.name.lower())]
    assert not forbidden, forbidden
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(json.dumps({"status": "PASS", "tests": ["protocol", "frozen_local_evidence", "no_generation_surface", "python_syntax"]}))


if __name__ == "__main__":
    main()
