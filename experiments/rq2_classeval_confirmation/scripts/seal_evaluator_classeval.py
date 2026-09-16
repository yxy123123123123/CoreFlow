#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.classeval_eval import (
    build_class_code,
    evaluate_classeval_method,
    evaluator_source_sha256,
    extract_method_code,
)
from coreflow.gate_io import write_json
from coreflow.io import load_json, load_jsonl, sha256_file
import ast


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extractor_cases() -> list[dict]:
    cases = []
    raw = "Below is an answer.\n### Response:\n```python\ndef foo(x):\n    return x + 1\n```"
    code = extract_method_code(raw, "foo")
    cases.append({
        "name": "fenced_target_method",
        "status": "PASS" if "def foo(" in code and "return x + 1" in code else "FAIL",
        "code": code,
    })
    raw2 = "class demo:\n    def bar(self, y):\n        return y * 2\n\ndef other():\n    pass"
    code2 = extract_method_code(raw2, "bar")
    cases.append({
        "name": "indented_class_method",
        "status": "PASS" if code2.startswith("    def bar(") else "FAIL",
        "code": code2,
    })
    code3 = extract_method_code("def baz(x):\n    return x", "missing")
    cases.append({
        "name": "missing_target",
        "status": "PASS" if code3 == "" else "FAIL",
        "code": code3,
    })
    raw4 = "def doc(x):\n    \"\"\"unclosed\n    return x\n"
    code4 = extract_method_code(raw4, "doc")
    cases.append({
        "name": "unclosed_docstring_repaired",
        "status": "PASS" if code4.rstrip().endswith('"""') or code4.rstrip().endswith("pass") else "FAIL",
        "code": code4,
    })
    return cases


def _runnable_class(cfg: dict) -> tuple[str, list[dict], dict[str, str]]:
    rows = load_jsonl(ROOT / "data" / cfg["data"]["formal"]["file"])
    class_id = str(rows[0]["class_id"])
    class_rows = [row for row in rows if str(row["class_id"]) == class_id]
    classes = load_json(ROOT / "raw_sources" / "ClassEval_data.json")
    target = next(cls for cls in classes if cls["task_id"] == class_id)
    raw_outputs = {str(method["method_name"]): method["solution_code"] for method in target["methods_info"]}
    return class_id, class_rows, raw_outputs


def canonical_class_smoke(cfg: dict) -> dict:
    class_id, rows, raw_outputs = _runnable_class(cfg)
    assembly = "PASS"
    if os.name != "posix":
        return {
            "status": "SKIPPED_NON_POSIX",
            "class_id": class_id,
            "methods": len(rows),
            "assembly": assembly,
            "note": "resource-limited subprocess smoke only runs on AutoDL/Linux",
        }
    evaluator = cfg["evaluator"]
    results = []
    for row in rows:
        class_code = build_class_code(row, raw_outputs)
        try:
            ast.parse(class_code)
        except SyntaxError as exc:
            assembly = f"FAIL:{exc.lineno}"
        result = evaluate_classeval_method(
            raw_outputs,
            row,
            evaluator,
            timeout_seconds=int(evaluator["code_timeout_seconds"]),
            cpu_seconds=int(evaluator["cpu_limit_seconds"]),
            memory_mb=int(evaluator["address_space_limit_mb"]),
            file_mb=int(evaluator["file_size_limit_mb"]),
            process_limit=int(evaluator["process_limit"]),
        )
        results.append(result)
    passed = assembly == "PASS" and all(result["passed"] for result in results)
    return {
        "status": "PASS" if passed else "FAIL",
        "class_id": class_id,
        "methods": len(rows),
        "assembly": assembly,
        "evaluation": results,
    }


def broken_method_smoke(cfg: dict) -> dict:
    if os.name != "posix":
        return {"status": "SKIPPED_NON_POSIX"}
    class_id, rows, raw_outputs = _runnable_class(cfg)
    row = rows[0]
    entry_point = str(row["entry_point"])
    bad_outputs = dict(raw_outputs)
    bad_outputs[entry_point] = f"def {entry_point}(self, *args, **kwargs):\n    return None"
    evaluator = cfg["evaluator"]
    result = evaluate_classeval_method(
        bad_outputs,
        row,
        evaluator,
        timeout_seconds=int(evaluator["code_timeout_seconds"]),
        cpu_seconds=int(evaluator["cpu_limit_seconds"]),
        memory_mb=int(evaluator["address_space_limit_mb"]),
        file_mb=int(evaluator["file_size_limit_mb"]),
        process_limit=int(evaluator["process_limit"]),
    )
    return {
        "status": "PASS" if not result["passed"] else "FAIL",
        "evaluation": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    cases = extractor_cases()
    canonical = canonical_class_smoke(cfg)
    broken = broken_method_smoke(cfg)
    execution_ok = (
        canonical["status"] == "PASS"
        and broken["status"] == "PASS"
    ) or canonical["status"] == "SKIPPED_NON_POSIX"
    all_pass = all(item["status"] == "PASS" for item in cases) and execution_ok
    payload = {
        "status": "PASS" if all_pass else "FAIL",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now(),
        "config_sha256": sha256_file(args.config),
        "evaluator_file_sha256": evaluator_source_sha256(),
        "m2_eval_file_sha256": sha256_file(ROOT / "coreflow" / "m2_eval.py"),
        "compositional_contract": {
            "official_style_prompt": True,
            "method_extracted_from_response": True,
            "class_assembled_before_unittest": True,
            "canonical_solutions_used_only_for_evaluator_seal": True,
        },
        "extractor_cases": cases,
        "canonical_class_smoke": canonical,
        "broken_method_smoke": broken,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "extractor_cases": len(cases), "canonical": canonical["status"], "broken": broken["status"]}, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
