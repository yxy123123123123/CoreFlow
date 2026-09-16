from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path


RESPONSE_MARKERS = ("### Response:", "@@ Response:", "[/INST]")
FENCE_PATTERNS = (
    re.compile(r"```(?:python|py)?(.*?)```", re.I | re.S),
    re.compile(r"\[PYTHON\](.*?)\[/PYTHON\]", re.I | re.S),
)
TEST_PASS_MARKER = "__COREFLOW_CLASSEVAL_ALL_TESTS_PASSED__"


def _leading_spaces(text: str) -> int:
    return len(text) - len(text.lstrip())


def _strip_response_marker(text: str) -> str:
    for marker in RESPONSE_MARKERS:
        if marker in text:
            return text.split(marker, 1)[1]
    return text


def _strip_fences(text: str) -> str:
    for pattern in FENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return text


def extract_method_code(raw_output: str, method_name: str) -> str:
    """Frozen ClassEval method extractor.

    Mirrors the official ClassEval compositional postprocessor: find the
    target def method_name( line, keep its indented body, normalize to
    4-space class indentation, and add @staticmethod only when the signature
    contains neither self nor cls.
    """
    text = (raw_output or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()
    if not text:
        return ""
    text = _strip_response_marker(text)
    text = _strip_fences(text)
    lines = text.split("\n")
    prefix = f"def {method_name}("
    start = next((index for index, line in enumerate(lines) if prefix in line), None)
    if start is None:
        return ""

    body = ["" if line.strip() in ("```", "```python", "```py") else line for line in lines[start:]]
    first_indent = _leading_spaces(body[0])
    end = len(body)
    for index in range(1, len(body)):
        line = body[index]
        if line.strip() and _leading_spaces(line) <= first_indent:
            end = index
            break
    body = body[:end]
    if not body:
        return ""

    normalized = []
    for line in body:
        if not line.strip():
            normalized.append("")
            continue
        shift = max(0, 4 - first_indent)
        normalized.append(" " * shift + line)

    first_line = normalized[0]
    if "self" not in first_line and "cls" not in first_line:
        normalized.insert(0, "    @staticmethod")

    docstring_mark = 0
    for line in normalized:
        if '"""' in line:
            docstring_mark += line.count('"""')
    if docstring_mark % 2 == 1:
        normalized.append(" " * 8 + '"""')
        normalized.append(" " * 8 + "pass")

    if len(normalized) == 1:
        normalized.append("        pass")
    code = "\n".join(normalized).rstrip() + "\n"
    return code


def get_method_signature(method_description: str, method_name: str) -> str:
    prefix = f"def {method_name}("
    for segment in method_description.split("):"):
        if prefix in segment:
            return "    " + segment + "):"
    return ""


def build_class_code(row: dict, raw_outputs: dict[str, str]) -> str:
    imports = "\n".join(row.get("import_statement") or [])
    code = imports + "\n" + row["class_constructor"]
    for name in row["method_order"]:
        code += "\n\n" + extract_method_code(raw_outputs.get(name, ""), name)
    return code


def _limit_script(cpu_seconds: int, memory_mb: int, file_mb: int, process_limit: int) -> str:
    return (
        "import resource,runpy,sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds},{cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({memory_mb}*1024*1024,{memory_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({file_mb}*1024*1024,{file_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_NPROC, ({process_limit},{process_limit}))\n"
        "runpy.run_path(sys.argv[1],run_name='__main__')\n"
    )


def _has_entry_def(code: str, entry_point: str) -> bool:
    return bool(re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M))


def _method_code_parse_error_line(code: str) -> int | None:
    for candidate in (code, textwrap.dedent(code)):
        try:
            ast.parse(candidate)
            return None
        except SyntaxError as exc:
            last_line = exc.lineno
    return last_line


TEST_PREAMBLE = (
    "import os, sys, re, json, random, io, math, unittest, time, copy, stat, "
    "collections, itertools, functools, string, tempfile, shutil, pathlib, statistics, "
    "csv, hashlib, sqlite3, calendar, decimal, fractions, typing\n"
    "from unittest.mock import MagicMock, patch, call\n"
    "try:\n"
    "    datetime\n"
    "except NameError:\n"
    "    from datetime import datetime\n"
    "try:\n"
    "    timedelta\n"
    "except NameError:\n"
    "    from datetime import timedelta\n"
    "try:\n"
    "    date\n"
    "except NameError:\n"
    "    from datetime import date\n"
    "try:\n"
    "    gettempdir\n"
    "except NameError:\n"
    "    from tempfile import gettempdir\n"
    "import numpy as np\n"
    "if not hasattr(np, 'mat'):\n"
    "    np.mat = np.asmatrix\n"
)


def _test_program(class_code: str, test_code: str, test_class: str) -> str:
    runner = (
        "\n\n"
        "import io\n"
        f"_suite = unittest.defaultTestLoader.loadTestsFromTestCase({test_class})\n"
        "_stream = io.StringIO()\n"
        f"_result = unittest.TextTestRunner(stream=_stream, verbosity=0).run(_suite)\n"
        "if not _result.wasSuccessful():\n"
        "    print(_stream.getvalue(), file=sys.stderr)\n"
        f"print({TEST_PASS_MARKER!r} if _result.wasSuccessful() else '')\n"
    )
    return class_code + "\n\n" + TEST_PREAMBLE + "\n" + test_code + runner


def evaluate_classeval_method(
    raw_outputs: dict[str, str],
    row: dict,
    spec: dict,
    *,
    timeout_seconds: int,
    cpu_seconds: int,
    memory_mb: int,
    file_mb: int,
    process_limit: int,
) -> dict:
    entry_point = str(row["entry_point"])
    target_code = extract_method_code(raw_outputs.get(entry_point, ""), entry_point)
    base = {
        "parse_mode": "classeval_compositional",
        "processed_code": target_code,
        "processed_chars": len(target_code),
        "processed_lines": len(target_code.splitlines()) if target_code else 0,
        "evaluator": "classeval_method",
        "entry_point": entry_point,
        "has_entrypoint_def": _has_entry_def(target_code, entry_point),
        "class_methods": list(row["method_order"]),
    }
    if not target_code:
        return {**base, "compiled": False, "execution_attempted": False, "executed": False, "passed": False, "status": "PARSE_FAILURE"}
    parse_error_line = _method_code_parse_error_line(target_code)
    if parse_error_line is not None:
        return {
            **base,
            "compiled": False,
            "execution_attempted": False,
            "executed": False,
            "passed": False,
            "status": f"COMPILE_FAILURE:{parse_error_line}",
        }
    class_code = build_class_code(row, raw_outputs)
    try:
        ast.parse(class_code)
    except SyntaxError as exc:
        return {
            **base,
            "compiled": False,
            "execution_attempted": False,
            "executed": False,
            "passed": False,
            "status": f"CLASS_COMPILE_FAILURE:{exc.lineno}",
        }
    if not _has_entry_def(class_code, entry_point):
        return {**base, "compiled": True, "execution_attempted": False, "executed": False, "passed": False, "status": "ENTRYPOINT_MISSING"}

    program = _test_program(class_code, str(row["test_code"]), str(row["test_class"]))
    with tempfile.TemporaryDirectory(prefix="coreflow_classeval_exec_") as temp:
        temp_root = Path(temp)
        candidate = temp_root / "candidate.py"
        candidate.write_text(program, encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _limit_script(cpu_seconds, memory_mb, file_mb, process_limit),
            str(candidate),
        ]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        }
        try:
            result = subprocess.run(
                command,
                cwd=temp_root,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                **base,
                "compiled": True,
                "execution_attempted": True,
                "executed": False,
                "passed": False,
                "status": "TIMEOUT",
            }
    passed = result.returncode == 0 and TEST_PASS_MARKER in result.stdout
    if passed:
        status = "PASS"
    elif "AssertionError" in result.stderr:
        status = "ASSERTION_FAILURE"
    else:
        status = "RUNTIME_FAILURE"
    return {
        **base,
        "compiled": True,
        "execution_attempted": True,
        "executed": True,
        "passed": passed,
        "status": status,
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-3000:],
    }


def evaluator_source_sha256() -> str:
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
