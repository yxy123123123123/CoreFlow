#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from coreflow.classeval_eval import (
    TEST_PASS_MARKER,
    build_class_code,
    evaluate_classeval_method,
    evaluator_source_sha256,
    _method_code_parse_error_line,
    extract_method_code,
    _test_program,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            value.update(chunk)
    return value.hexdigest()


def load_config() -> dict:
    return json.loads((ROOT / "config" / "indep_confirm_classeval_protocol.json").read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_row(path: Path) -> dict:
    return load_jsonl(path)[0]


def test_protocol_contract() -> None:
    cfg = load_config()
    assert cfg["protocol_id"] == "coreflow-indep-confirm-classeval-v1"
    assert cfg["assets"]["primary_q"] == 185
    assert cfg["hypotheses"]["bootstrap_seed"] == 20260806
    assert cfg["hypotheses"]["bootstrap_seed"] != 20260729
    assert cfg["hypotheses"]["noninferiority_margin_pp"] == 3.0
    assert cfg["hypotheses"]["paired_cluster_bootstrap_samples"] == 20000
    assert cfg["hypotheses"]["qualification"]["min_rows"] == 98
    assert cfg["hypotheses"]["qualification"]["min_correct"] == 8
    assert cfg["hypotheses"]["qualification"]["p_null"] == 0.03
    assert cfg["hypotheses"]["qualification"]["p_target"] == 0.15
    assert cfg["generation"]["max_input_tokens"] == 2048
    assert cfg["generation"]["max_new_tokens_code"] == 768
    assert cfg["execution_contract"]["no_automatic_retry"]
    assert cfg["execution_contract"]["seal_before_formal_model_generation"]


def test_method_matrix_and_order() -> None:
    cfg = load_config()
    expected = [
        "code_full_k5_seed41",
        "code_core_q185_k5_seed41",
        "code_isvd_q185_k5_seed41",
        "code_full_k5_seed42",
        "code_core_q185_k5_seed42",
        "code_isvd_q185_k5_seed42",
        "code_full_k5_seed43",
        "code_core_q185_k5_seed43",
        "code_isvd_q185_k5_seed43",
    ]
    assert cfg["phase_orders"]["formal_main"] == expected
    assert cfg["phase_orders"]["qualification"] == ["code_full_k5_seed41"]
    assert all(name.startswith("code_") for name in expected)
    assert cfg["expert_pools"]["k5_code"] == ["zh", "ru", "es", "math", "code"]


def test_embedded_data_contract() -> None:
    cfg = load_config()
    split_names = ("qualification", "formal", "reserve")
    for name in split_names:
        spec = cfg["data"][name]
        path = ROOT / "data" / spec["file"]
        assert digest(path) == spec["sha256"], name
        rows = load_jsonl(path)
        assert len(rows) == spec["rows"], name
        assert all(row["evaluator"] == "classeval_method" for row in rows)
        assert all("solution_code" not in row and "hidden_test" not in row for row in rows)
    qual = load_jsonl(ROOT / "data" / cfg["data"]["qualification"]["file"])
    formal = load_jsonl(ROOT / "data" / cfg["data"]["formal"]["file"])
    reserve = load_jsonl(ROOT / "data" / cfg["data"]["reserve"]["file"])
    qual_ids = {row["task_id"] for row in qual}
    formal_ids = {row["task_id"] for row in formal}
    reserve_ids = {row["task_id"] for row in reserve}
    assert not (qual_ids & formal_ids)
    assert not (qual_ids & reserve_ids)
    assert not (formal_ids & reserve_ids)
    assert len(qual_ids) == len(qual) == 98
    assert len(formal_ids) == len(formal) == 202
    assert len(reserve_ids) == len(reserve) == 39
    row = first_row(ROOT / "data" / cfg["data"]["formal"]["file"])
    for field in ("task_id", "class_id", "class_name", "entry_point", "method_order", "problem", "test_code", "test_class"):
        assert field in row
    assert hashlib.sha256(row["problem"].encode("utf-8")).hexdigest() == row["prompt_sha256"]


def test_prompt_and_evaluator_contract() -> None:
    cfg = load_config()
    prompt = ROOT / "data" / cfg["prompts"]["code_file"]
    text = prompt.read_text(encoding="utf-8")
    assert digest(prompt) == cfg["prompts"]["code_sha256"]
    assert text.count("{problem}") == 1
    assert "{entry_point}" not in text
    assert "{test_code}" not in text and "{solution_code}" not in text
    assert digest(ROOT / "coreflow" / "classeval_eval.py") == cfg["evaluator"]["file_sha256"]
    assert evaluator_source_sha256() == cfg["evaluator"]["file_sha256"]


def test_asset_locks() -> None:
    lock = json.loads((ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json").read_text(encoding="utf-8"))
    assert len(lock["gates"]) == 15
    assert sum(len(item) for item in lock["banks"].values()) == 10
    isvd = json.loads((ROOT / "provenance" / "EXPECTED_ISVD_LOCK.json").read_text(encoding="utf-8"))
    assert set(isvd["banks"]) == {"k5_math", "k5_code"}
    assert isvd["banks"]["k5_code"]["isvd_bank_sha256"] == "a8b5a3bf6ba3265fd8c35ec26f100d541be6f27e5b2fb4aaf8cc37f673804433"


def test_split_manifest_contract() -> None:
    cfg = load_config()
    manifest_path = ROOT / "data" / cfg["data"]["manifest"]["file"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert digest(manifest_path) == cfg["data"]["manifest"]["sha256"]
    assert manifest["method_counts"] == {"qualification": 98, "formal": 202, "reserve": 39}
    assert manifest["checks"]["no_excluded_class_in_splits"]
    assert len(manifest["excluded_class_ids_with_reasons"]) == 17
    assert manifest["checks"]["no_split_overlap"]
    assert manifest["checks"]["no_solution_code_in_rows"]
    assert manifest["tokenizer_note"]["max_prompt_tokens"] <= 2048


def test_classeval_evaluator_units() -> None:
    code = extract_method_code("```python\ndef foo(x):\n    return x + 1\n```", "foo")
    assert "def foo(" in code and "return x + 1" in code
    code2 = extract_method_code("class demo:\n    def bar(self, y):\n        return y * 2\n\ndef other():\n    pass", "bar")
    assert code2.startswith("    def bar(")
    assert extract_method_code("def baz(x):\n    return x", "missing") == ""
    raw4 = "def doc(x):\n    \"\"\"unclosed\n    return x\n"
    code4 = extract_method_code(raw4, "doc")
    assert code4.rstrip().endswith('"""') or code4.rstrip().endswith("pass")

    classes = json.loads((ROOT / "raw_sources" / "ClassEval_data.json").read_text(encoding="utf-8"))
    target = min(classes, key=lambda cls: (len(cls["methods_info"]), cls["task_id"]))
    order = [str(method["method_name"]) for method in target["methods_info"]]
    raw_outputs = {str(method["method_name"]): method["solution_code"] for method in target["methods_info"]}
    first = target["methods_info"][0]
    canonical_code = extract_method_code(raw_outputs[str(first["method_name"])], str(first["method_name"]))
    assert _method_code_parse_error_line(canonical_code) is None
    broken_code = "    def broken(self, x):\n        return x +\n"
    assert _method_code_parse_error_line(broken_code) is not None
    row = {
        "task_id": f'{target["task_id"]}::{first["method_name"]}',
        "class_id": target["task_id"],
        "entry_point": str(first["method_name"]),
        "method_order": order,
        "import_statement": list(target["import_statement"]),
        "class_constructor": target["class_constructor"],
        "test_code": first["test_code"],
        "test_class": first["test_class"],
    }
    class_code = build_class_code(row, raw_outputs)
    ast.parse(class_code)
    program = _test_program(class_code, row["test_code"], row["test_class"])
    namespace = {}
    stream = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(stream):
        exec(compile(program, "<classeval-smoke>", "exec"), namespace)
    assert TEST_PASS_MARKER in stream.getvalue()


def test_runner_guards() -> None:
    shell = (ROOT / "scripts" / "run_indep_confirm.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_formal_method.py").read_text(encoding="utf-8")
    control = (ROOT / "scripts" / "v2_control.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "coreflow" / "classeval_eval.py").read_text(encoding="utf-8")
    assert '"-I",' in evaluator and '"-B",' in evaluator
    assert '"-S",' not in evaluator
    assert "stat," in evaluator
    assert "from unittest.mock import MagicMock, patch, call" in evaluator
    assert "_method_code_parse_error_line" in evaluator
    assert "partial output was retained and no automatic retry was attempted" in shell
    assert "Run seal first" in shell
    assert "seal_before_formal_model_generation" in control or "SEALED_BEFORE_FORMAL_MODEL_OUTPUTS" in control
    assert "evaluate_classeval_method" in runner
    assert "classeval_raw" in runner
    assert "check-isvd" in shell
    assert "isvd_bank_projection" in control
    assert "\"--isvd-asset-lock\"" in control


def test_script_sys_import_contract() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "sys.path" in text:
            assert "import sys" in text, f"{path.name} uses sys.path without importing sys"

def test_no_primary_overlap_record() -> None:
    cfg = load_config()
    audit = cfg["no_overlap_audit"]
    assert audit["status"] == "PASS"
    assert audit["mbppplus_dev128_task_ids"] > 0
    assert audit["mbppplus_formal250_task_ids"] > 0
    assert audit["overlap_with_dev128"] == 0
    assert audit["overlap_with_formal250"] == 0


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
        test_prompt_and_evaluator_contract,
        test_asset_locks,
        test_split_manifest_contract,
        test_classeval_evaluator_units,
        test_runner_guards,
        test_script_sys_import_contract,
        test_no_primary_overlap_record,
        test_python_syntax,
    ]
    for test in tests:
        test()
    print(json.dumps({"status": "PASS", "tests": [test.__name__ for test in tests]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
