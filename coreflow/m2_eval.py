from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)
CODE_START = re.compile(r"^\s*(?:import|from\s+\S+\s+import|def|class|if\s+__name__\s*==|@)", re.I)
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")
NON_CODE_LABEL = re.compile(r"^\s*(?:Input|Output|Test|Tests|Example|Examples|Explanation|Function|Solution|Constraints|Note|Hint|Code)\s*:\s*$", re.I)


def postprocess_program(raw: str) -> tuple[str, str]:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", "empty"
    fenced = FENCE.search(text)
    if fenced:
        return fenced.group(1).strip(), "fenced_block"
    lines = text.splitlines()
    start = next((idx for idx, line in enumerate(lines) if CODE_START.match(line)), None)
    if start is None:
        return "", "no_code_start"
    return "\n".join(lines[start:]).strip(), "first_code_line"


def postprocess_mbppplus_program(raw: str) -> tuple[str, str]:
    code, mode = postprocess_program(raw)
    if not code:
        return code, mode
    lines = code.splitlines()
    kept = []
    for line in lines:
        if kept and NON_CODE_LABEL.match(line):
            break
        kept.append(line)
    candidate = "\n".join(kept).strip()
    if candidate != code:
        return candidate, mode + "_truncated_at_label"
    # If the model emits a valid function followed by stray prose, keep the
    # longest prefix that parses. This is deliberately a postprocessor only:
    # it does not rename functions or repair semantics.
    best = None
    for end in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:end]).strip()
        try:
            ast.parse(prefix)
        except SyntaxError:
            continue
        best = prefix
        break
    if best is not None and best != code:
        return best, mode + "_longest_ast_prefix"
    return code, mode


def _longest_ast_prefix(lines: list[str]) -> str | None:
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate
    return None


def _truncate_at_label(code: str) -> str:
    lines = code.splitlines()
    kept: list[str] = []
    for line in lines:
        if kept and NON_CODE_LABEL.match(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def contains_entry_def(code: str, entry_point: str) -> bool:
    if not code or not entry_point:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )


def postprocess_entrypoint_program(raw: str, entry_point: str) -> tuple[str, str]:
    """F0R2-frozen MBPP+ code selector.

    Selection is based only on syntax and the requested entry point. It never
    runs hidden tests to choose among candidates.
    """
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", "empty"

    fenced_blocks = [match.group(1).strip() for match in FENCE.finditer(text)]
    for block in fenced_blocks:
        candidate = _truncate_at_label(block)
        if contains_entry_def(candidate, entry_point):
            return candidate, "entrypoint_fenced_block"
    for block in fenced_blocks:
        if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", block, re.M):
            parsed = _longest_ast_prefix(_truncate_at_label(block).splitlines())
            if parsed and contains_entry_def(parsed, entry_point):
                return parsed, "entrypoint_fenced_block_ast_prefix"

    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", line):
            start = index
            break
    if start is not None:
        parsed = _longest_ast_prefix(_truncate_at_label("\n".join(lines[start:])).splitlines())
        if parsed and contains_entry_def(parsed, entry_point):
            return parsed, "entrypoint_text_ast_prefix"

    for block in fenced_blocks:
        parsed = _longest_ast_prefix(_truncate_at_label(block).splitlines())
        if parsed:
            return parsed, "non_entrypoint_fenced_block"
    start = next((idx for idx, line in enumerate(lines) if CODE_START.match(line)), None)
    if start is not None:
        parsed = _longest_ast_prefix(_truncate_at_label("\n".join(lines[start:])).splitlines())
        if parsed:
            return parsed, "non_entrypoint_text_ast_prefix"
    return "", "no_entrypoint_code"


def _limit_script(cpu_seconds: int, memory_mb: int, file_mb: int, process_limit: int) -> str:
    return (
        "import resource,runpy,sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds},{cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({memory_mb}*1024*1024,{memory_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({file_mb}*1024*1024,{file_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_NPROC, ({process_limit},{process_limit}))\n"
        "runpy.run_path(sys.argv[1],run_name='__main__')\n"
    )


def normalize_stdout(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines()).strip()


def evaluate_apps_program(raw_output: str, io_cases: list[dict], spec: dict) -> dict:
    code, parse_mode = postprocess_program(raw_output)
    base = {"parse_mode": parse_mode, "processed_code": code, "processed_chars": len(code), "case_count": len(io_cases)}
    if not code:
        return {**base, "compiled": False, "passed": False, "status": "PARSE_FAILURE", "passed_cases": 0}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {**base, "compiled": False, "passed": False, "status": f"COMPILE_FAILURE:{exc.lineno}", "passed_cases": 0}
    with tempfile.TemporaryDirectory(prefix="coreflow_m2_apps_") as temp:
        root = Path(temp)
        candidate = root / "candidate.py"
        candidate.write_text(code, encoding="utf-8", newline="\n")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
        passed_cases = 0
        failures = []
        for index, case in enumerate(io_cases):
            command = [
                sys.executable, "-I", "-S", "-B", "-c",
                _limit_script(int(spec["cpu_limit_seconds"]), int(spec["address_space_limit_mb"]), int(spec["file_size_limit_mb"]), int(spec["process_limit"])),
                str(candidate),
            ]
            try:
                result = subprocess.run(
                    command,
                    input=str(case["input"]),
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=int(spec["code_timeout_seconds"]),
                )
            except subprocess.TimeoutExpired:
                failures.append({"case_index": index, "status": "TIMEOUT"})
                continue
            ok = result.returncode == 0 and normalize_stdout(result.stdout) == normalize_stdout(str(case["output"]))
            if ok:
                passed_cases += 1
            else:
                failures.append({"case_index": index, "status": "WRONG_ANSWER", "returncode": result.returncode, "stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-1000:]})
    return {**base, "compiled": True, "passed": passed_cases == len(io_cases), "status": "PASS" if passed_cases == len(io_cases) else "FAIL", "passed_cases": passed_cases, "failures": failures[:3]}


def evaluate_mbppplus_program(raw_output: str, hidden_test: str, spec: dict, entry_point: str = "") -> dict:
    if entry_point:
        code, parse_mode = postprocess_entrypoint_program(raw_output, entry_point)
        has_entrypoint = contains_entry_def(code, entry_point) if code else False
        evaluator_name = "mbppplus_asserts_f0r2"
    else:
        code, parse_mode = postprocess_mbppplus_program(raw_output)
        has_entrypoint = None
        evaluator_name = "mbppplus_asserts"
    base = {
        "parse_mode": parse_mode,
        "processed_code": code,
        "processed_chars": len(code),
        "case_count": None,
        "evaluator": evaluator_name,
        "entry_point": entry_point,
        "has_entrypoint_def": has_entrypoint,
    }
    if not code:
        return {**base, "compiled": False, "passed": False, "status": "PARSE_FAILURE"}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {**base, "compiled": False, "passed": False, "status": f"COMPILE_FAILURE:{exc.lineno}"}
    if entry_point and not contains_entry_def(code, entry_point):
        return {**base, "compiled": True, "passed": False, "status": "ENTRYPOINT_MISSING"}
    with tempfile.TemporaryDirectory(prefix="coreflow_v2_mbppplus_") as temp:
        root = Path(temp)
        candidate = root / "candidate.py"
        candidate.write_text(code + "\n\n" + hidden_test + "\n", encoding="utf-8", newline="\n")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        }
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _limit_script(
                int(spec["cpu_limit_seconds"]),
                int(spec["address_space_limit_mb"]),
                int(spec["file_size_limit_mb"]),
                int(spec["process_limit"]),
            ),
            str(candidate),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=int(spec["code_timeout_seconds"]),
            )
        except subprocess.TimeoutExpired:
            return {**base, "compiled": True, "passed": False, "status": "TIMEOUT"}
    passed = result.returncode == 0
    return {
        **base,
        "compiled": True,
        "passed": bool(passed),
        "status": "PASS" if passed else "FAIL",
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def _number_value(text: str):
    matches = NUMBER.findall((text or "").replace(",", ""))
    if not matches:
        return None
    token = matches[-1]
    try:
        return Fraction(token)
    except Exception:
        try:
            return Fraction(str(float(token))).limit_denominator(1000000)
        except Exception:
            return None


def evaluate_math_answer(raw_output: str, answer: str) -> dict:
    predicted = _number_value(raw_output)
    expected = _number_value(answer)
    passed = predicted is not None and expected is not None and predicted == expected
    return {
        "processed_answer": None if predicted is None else str(predicted),
        "expected_answer": None if expected is None else str(expected),
        "passed": bool(passed),
        "status": "PASS" if passed else "WRONG_ANSWER",
    }
