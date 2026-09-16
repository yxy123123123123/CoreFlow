#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    cfg = json.loads((ROOT / "config" / "prs4a_prs4p_protocol.json").read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == "coreflow-prs4a-prs4p-pilot-v1"
    assert cfg["data"]["formal164_embedded"] is False
    assert cfg["execution_contract"]["formal164_runtime_access_forbidden"] is True
    assert cfg["pilot_B"]["expected_records"] == 48
    assert cfg["pilot_B"]["method_order"] == [
        "full_legacy_original",
        "coreflow_q185_frozen",
        "isvd_legacy_padded_matched_q185",
    ]
    assert cfg["legacy_A"]["actual_method"] == "full_vectorized"
    assert cfg["legacy_A"]["method_mismatch_note"]

    formal_files = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and "formal164" in path.name.lower()
    ]
    assert formal_files == [], formal_files

    data = ROOT / "data" / cfg["data"]["pilot_file"]
    assert sha256(data) == cfg["data"]["pilot_sha256"]
    rows = [json.loads(line) for line in data.open("r", encoding="utf-8") if line.strip()]
    assert len(rows) == 16
    assert [str(row["question_id"]) for row in rows] == cfg["data"]["pilot_question_ids"]

    runner = (ROOT / "scripts" / "run_pilot_b.py").read_text(encoding="utf-8")
    server = (ROOT / "scripts" / "serve_method.py").read_text(encoding="utf-8")
    for method_id in cfg["pilot_B"]["method_order"]:
        assert method_id in server
    assert "livecodebench_formal164.jsonl" not in runner.lower()
    assert "qualification_file" not in runner
    assert "install_full_vectorized" not in runner
    assert "no_automatic_retry" in json.dumps(cfg)

    for path in list((ROOT / "scripts").glob("*.py")) + list((ROOT / "coreflow").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(json.dumps({"status": "PASS", "tests": [
        "protocol_boundary",
        "three_method_identity",
        "legacy_method_mismatch_preserved",
        "formal164_absence",
        "pilot_data_hash_and_order",
        "pilot_runner_forbidden_scan",
        "python_syntax",
    ]}))


if __name__ == "__main__":
    main()
