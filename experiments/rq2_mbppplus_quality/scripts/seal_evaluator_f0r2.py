#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import write_json
from coreflow.io import sha256_file
from coreflow.m2_eval import evaluate_mbppplus_program, postprocess_entrypoint_program


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def case(name: str, raw: str, entry_point: str, expected_mode: str, expect_has_entry: bool) -> dict:
    code, mode = postprocess_entrypoint_program(raw, entry_point)
    ok = mode == expected_mode and (f"def {entry_point}(" in code) == expect_has_entry
    return {
        "name": name,
        "status": "PASS" if ok else "FAIL",
        "mode": mode,
        "expected_mode": expected_mode,
        "processed_code": code,
        "contains_entrypoint_text": f"def {entry_point}(" in code,
    }


def numpy_smoke(spec: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="coreflow_v2_numpy_smoke_") as tmp:
        script = Path(tmp) / "smoke.py"
        script.write_text("import numpy as np\nprint(np.__version__)\n", encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            (
                "import resource,runpy,sys\n"
                f"resource.setrlimit(resource.RLIMIT_CPU, ({int(spec['cpu_limit_seconds'])},{int(spec['cpu_limit_seconds'])}))\n"
                f"resource.setrlimit(resource.RLIMIT_AS, ({int(spec['address_space_limit_mb'])}*1024*1024,{int(spec['address_space_limit_mb'])}*1024*1024))\n"
                f"resource.setrlimit(resource.RLIMIT_FSIZE, ({int(spec['file_size_limit_mb'])}*1024*1024,{int(spec['file_size_limit_mb'])}*1024*1024))\n"
                f"resource.setrlimit(resource.RLIMIT_NPROC, ({int(spec['process_limit'])},{int(spec['process_limit'])}))\n"
                "runpy.run_path(sys.argv[1],run_name='__main__')\n"
            ),
            str(script),
        ]
        result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, errors="replace", timeout=10)
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr_tail": result.stderr[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tests = [
        case(
            "example_call_then_function",
            "```python\nfoo(1)\n```\nNow the solution:\ndef foo(x):\n    return x\n",
            "foo",
            "entrypoint_text_ast_prefix",
            True,
        ),
        case(
            "multiple_fenced_blocks",
            "```python\nbar(1)\n```\n```python\ndef bar(x):\n    return x + 1\n```",
            "bar",
            "entrypoint_fenced_block",
            True,
        ),
        case(
            "no_fence_plain_function",
            "Here is code:\ndef baz(x):\n    return x * 2\n\nExplanation: done",
            "baz",
            "entrypoint_text_ast_prefix",
            True,
        ),
        case(
            "missing_target_function",
            "```python\ndef other(x):\n    return x\n```",
            "target",
            "non_entrypoint_fenced_block",
            False,
        ),
        case(
            "syntax_error_after_valid_function",
            "```python\ndef zap(x):\n    return x\nthis is prose\n```",
            "zap",
            "entrypoint_fenced_block_ast_prefix",
            True,
        ),
    ]
    hidden = "import numpy as np\nassert np.array_equal(np.asarray(inc(1)), np.asarray(2))\n"
    eval_pass = evaluate_mbppplus_program("```python\ndef inc(x):\n    return x + 1\n```", hidden, cfg["evaluator"], "inc")
    eval_fail = evaluate_mbppplus_program("```python\ndef other(x):\n    return x + 1\n```", hidden, cfg["evaluator"], "inc")
    smoke = numpy_smoke(cfg["evaluator"])
    all_pass = (
        all(item["status"] == "PASS" for item in tests)
        and eval_pass["status"] == "PASS"
        and eval_fail["status"] == "ENTRYPOINT_MISSING"
        and smoke["status"] == "PASS"
    )
    payload = {
        "status": "PASS" if all_pass else "FAIL",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now(),
        "config_sha256": sha256_file(args.config),
        "evaluator_file_sha256": sha256_file(ROOT / "coreflow" / "m2_eval.py"),
        "selection_contract": {
            "uses_entry_point": True,
            "multi_fenced_block_rule": "first AST-valid block containing def entry_point(...), then AST prefix fallbacks",
            "hidden_tests_used_for_selection": False,
            "numpy_available_in_isolated_executor": smoke["status"] == "PASS",
        },
        "postprocessor_tests": tests,
        "pass_case": eval_pass,
        "missing_entrypoint_case": eval_fail,
        "numpy_smoke": smoke,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "tests": len(tests), "numpy": smoke["status"]}, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
