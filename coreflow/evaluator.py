from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


CODE_START = re.compile(r"^\s*(?:async\s+def|def|class|from\s+\S+\s+import|import\s+\S+|@)", re.I)
FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)


def postprocess_code(raw: str) -> tuple[str, str]:
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return "", "empty"
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
        source = "fenced_block"
    else:
        text = text.replace('"""', "").strip()
        lines = text.splitlines()
        start = next((index for index, line in enumerate(lines) if CODE_START.match(line)), None)
        if start is None:
            return "", "no_code_start"
        text = "\n".join(lines[start:]).strip()
        source = "first_code_line"
    for marker in ("\n[/INST]", "\n[INST]", "\n### Explanation", "\nExplanation:"):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    return text, source


def ast_safety_scan(code: str, unsafe_imports: set[str], unsafe_calls: set[str]) -> tuple[bool, list[str], ast.AST | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"syntax:{exc.msg}@{exc.lineno}"], None
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in unsafe_imports:
                    findings.append(f"unsafe_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in unsafe_imports:
                findings.append(f"unsafe_import:{node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in unsafe_calls:
                findings.append(f"unsafe_call:{node.func.id}")
    return not findings, sorted(set(findings)), tree


def repeated_line_degeneration(code: str) -> bool:
    lines = [re.sub(r"\s+", " ", line).strip() for line in code.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    counts = {}
    for line in lines:
        counts[line] = counts.get(line, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines) >= 0.5


def _limit_script(cpu_seconds: int, memory_mb: int, file_mb: int, process_limit: int) -> str:
    return (
        "import resource,runpy,sys\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds},{cpu_seconds}))\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({memory_mb}*1024*1024,{memory_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_FSIZE, ({file_mb}*1024*1024,{file_mb}*1024*1024))\n"
        f"resource.setrlimit(resource.RLIMIT_NPROC, ({process_limit},{process_limit}))\n"
        "runpy.run_path(sys.argv[1],run_name='__main__')\n"
    )


def evaluate_candidate(
    raw_output: str,
    test_list: list[str],
    test_setup_code: str,
    *,
    timeout_seconds: int,
    cpu_seconds: int,
    memory_mb: int,
    file_mb: int,
    process_limit: int,
    unsafe_imports: set[str],
    unsafe_calls: set[str],
) -> dict:
    code, parse_mode = postprocess_code(raw_output)
    base = {
        "empty_output": not bool((raw_output or "").strip()),
        "parse_mode": parse_mode,
        "processed_code": code,
        "processed_chars": len(code),
        "processed_lines": len(code.splitlines()) if code else 0,
        "repetition_degeneration": repeated_line_degeneration(code),
    }
    if not code:
        return {**base, "compiled": False, "execution_attempted": False, "executed": False, "passed": False, "status": "PARSE_FAILURE", "findings": []}
    safe, findings, tree = ast_safety_scan(code, unsafe_imports, unsafe_calls)
    if tree is None:
        return {**base, "compiled": False, "execution_attempted": False, "executed": False, "passed": False, "status": "COMPILE_FAILURE", "findings": findings}
    if not safe:
        return {**base, "compiled": True, "execution_attempted": False, "executed": False, "passed": False, "status": "UNSAFE_REJECTED", "findings": findings}
    marker = "__COREFLOW_M1_ALL_TESTS_PASSED__"
    program = "\n".join(
        [
            code,
            str(test_setup_code or ""),
            *[str(test) for test in test_list],
            f"print({marker!r})",
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="coreflow_m1_exec_") as temp:
        temp_root = Path(temp)
        candidate_path = temp_root / "candidate.py"
        candidate_path.write_text(program, encoding="utf-8", newline="\n")
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _limit_script(cpu_seconds, memory_mb, file_mb, process_limit),
            str(candidate_path),
        ]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
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
                "findings": findings,
            }
    passed = result.returncode == 0 and marker in result.stdout
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
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-4000:],
        "findings": findings,
    }


def evaluator_source_sha256() -> str:
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
