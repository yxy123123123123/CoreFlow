#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def load_config() -> dict:
    return json.loads((ROOT / "config" / "formal_protocol.json").read_text(encoding="utf-8"))


def first_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def test_protocol_contract() -> None:
    cfg = load_config()
    assert cfg["protocol_id"] == "coreflow-formal-v1.1"
    assert cfg["assets"]["primary_q"] == 185
    assert cfg["assets"]["conditional_fallback_q"] == 224
    assert cfg["assets"]["q224_policy"].startswith("DO_NOT_RUN")
    assert cfg["hypotheses"]["noninferiority_margin_pp"] == 3.0
    assert cfg["hypotheses"]["minimum_retention_ratio"] == 0.95
    assert cfg["hypotheses"]["code_formal_full_k5_floor_correct"] == 24
    assert cfg["hypotheses"]["paired_cluster_bootstrap_samples"] == 100000
    assert cfg["code_floor"]["confirmatory_min_correct"] == 11
    assert cfg["code_floor"]["supportive_min_correct"] == 7
    assert cfg["execution_contract"]["legacy_math_k5_seed42_pair_required_before_seal"]
    assert cfg["execution_contract"]["formal_code_remains_closed_if_dev_is_stress_only"]


def test_method_matrix_and_order() -> None:
    cfg = load_config()
    assert cfg["phase_orders"]["math_main"] == [
        "math_full_k5_seed41",
        "math_core_q185_k5_seed41",
        "math_core_q185_k5_seed42",
        "math_full_k5_seed42",
        "math_full_k5_seed43",
        "math_core_q185_k5_seed43",
    ]
    assert cfg["phase_orders"]["code_main"][2:4] == [
        "code_core_q185_k5_seed42",
        "code_full_k5_seed42",
    ]
    assert all("q224" not in method for method in cfg["phase_orders"]["math_main"])
    assert all("q224" not in method for method in cfg["phase_orders"]["code_main"])
    assert cfg["phase_orders"]["q224_conditional"] == [
        "math_core_q224_k5_seed41",
        "code_core_q224_k5_seed41",
    ]


def test_embedded_data_contract() -> None:
    cfg = load_config()
    for name, spec in cfg["data"].items():
        path = ROOT / "data" / spec["file"]
        assert digest(path) == spec["sha256"], name
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(bool(line.strip()) for line in handle)
        assert rows == spec["rows"], name
    code = first_row(ROOT / "data" / cfg["data"]["code_formal"]["file"])
    math = first_row(ROOT / "data" / cfg["data"]["math_formal"]["file"])
    legacy = first_row(ROOT / "data" / cfg["data"]["legacy_m2_math"]["file"])
    assert {"task_id", "problem", "io_cases"} <= set(code)
    assert {"task_id", "question", "golden"} <= set(math)
    assert {"task_id", "problem", "answer"} <= set(legacy)


def test_prompt_and_hidden_evaluator_contract() -> None:
    cfg = load_config()
    for task in ("code", "math"):
        path = ROOT / "data" / cfg["prompts"][f"{task}_file"]
        text = path.read_text(encoding="utf-8")
        assert digest(path) == cfg["prompts"][f"{task}_sha256"]
        assert text.count("{problem}") == 1
        assert "{io_cases}" not in text
        assert "{golden}" not in text
        assert "{answer}" not in text
        assert "\ufffd" not in text


def test_asset_payload_lock() -> None:
    lock = json.loads((ROOT / "provenance" / "ASSET_PAYLOAD_LOCK.json").read_text(encoding="utf-8"))
    assert lock["status"] == "PASS"
    assert lock["verification_level"] == "FULL_STREAM_HASH"
    assert lock["archive_sha256"] == load_config()["assets"]["payload_sha256"]
    assert lock["counts"] == {
        "gate_pt": 15,
        "gate_json": 15,
        "core_bank": 10,
        "core_config": 10,
        "build_report": 5,
    }
    expected = json.loads((ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json").read_text(encoding="utf-8"))
    assert len(expected["gates"]) == 15
    assert sum(len(item) for item in expected["banks"].values()) == 10


def test_known_hotfixes_are_present() -> None:
    runtime = (ROOT / "coreflow" / "runtime.py").read_text(encoding="utf-8")
    loraflow = (ROOT / "coreflow" / "loraflow.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_formal_method.py").read_text(encoding="utf-8")
    assert 'item.setdefault("expert_index", index)' in runtime
    assert 'item["checkpoint_sha256"] = item["adapter_sha256"]' in runtime
    assert "layer_pattern = re.compile" in loraflow
    assert "A frozen gate is required for every multi-expert method" in loraflow
    assert 'row.get("question", row.get("problem"))' in runner
    assert 'row.get("golden", row.get("answer"))' in runner


def test_runner_guards_and_benchmark() -> None:
    shell = (ROOT / "scripts" / "run_formal.sh").read_text(encoding="utf-8")
    bench = (ROOT / "scripts" / "run_microbenchmark.py").read_text(encoding="utf-8")
    server = (ROOT / "scripts" / "serve_formal_method.py").read_text(encoding="utf-8")
    assert "partial output was retained and no automatic retry was attempted" in shell
    assert "Run seal before any formal phase" in shell
    assert "code formal split remains closed" in shell
    assert "activate-q224" in shell
    assert "fixed_input_tokens" in bench
    assert "--min-new-tokens" in bench
    assert "min_new_tokens" in server


def test_python_syntax() -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def main() -> None:
    tests = [
        test_protocol_contract,
        test_method_matrix_and_order,
        test_embedded_data_contract,
        test_prompt_and_hidden_evaluator_contract,
        test_asset_payload_lock,
        test_known_hotfixes_are_present,
        test_runner_guards_and_benchmark,
        test_python_syntax,
    ]
    for test in tests:
        test()
    print(json.dumps({"status": "PASS", "tests": [test.__name__ for test in tests]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
