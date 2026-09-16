from __future__ import annotations

import ast
import base64
import json
import multiprocessing as mp
import pickle
import re
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor_livecodebench"


def prompt_for(row: dict) -> str:
    prompt = "### Question:\n" + row["question_content"].rstrip() + "\n\n"
    if row.get("starter_code"):
        prompt += "### Format: Use the following starter code to write the solution. Return one complete Python program in a fenced code block.\n"
        prompt += "```python\n" + row["starter_code"].rstrip() + "\n```\n\n"
    else:
        prompt += "### Format: Read from stdin and write to stdout. Return one complete Python program in a fenced code block. Do not hard-code sample tests.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt + "### Answer:\n"


def extract_code(text: str) -> tuple[str, str]:
    blocks = re.findall(r"```(?:python|Python)?\s*\n?(.*?)```", text, flags=re.DOTALL)
    candidate, mode = (blocks[-1].strip(), "last_complete_fenced_block") if blocks else (text.strip(), "raw_text")
    try:
        ast.parse(candidate)
        return candidate, mode
    except SyntaxError:
        lines = candidate.splitlines()
        for end in range(len(lines) - 1, 0, -1):
            prefix = "\n".join(lines[:end]).rstrip()
            try:
                tree = ast.parse(prefix)
                if tree.body:
                    return prefix, mode + "_ast_prefix"
            except SyntaxError:
                continue
    return candidate, mode + "_syntax_error"


def _decode_private(value: str) -> list[dict]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))


def evaluation_sample(row: dict) -> dict:
    tests = json.loads(row["public_test_cases"]) + _decode_private(row["private_test_cases"])
    metadata = json.loads(row["metadata"])
    return {"input_output": json.dumps({
        "inputs": [item["input"] for item in tests],
        "outputs": [item["output"] for item in tests],
        "fn_name": metadata.get("func_name"),
    })}, len(tests)


def _worker(sample: dict, code: str, timeout: int, queue) -> None:
    try:
        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (max(16, timeout * 2), max(20, timeout * 2 + 4)))
            resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
        sys.path.insert(0, str(VENDOR))
        from testing_util import run_test
        result, metadata = run_test(sample, test=code, debug=False, timeout=timeout)
        fixed = [bool(item.item() if hasattr(item, "item") else item) for item in result]
        queue.put({"results": fixed, "metadata": metadata})
    except BaseException as exc:
        queue.put({"error": repr(exc)})


def evaluate(row: dict, raw_output: str, per_test_timeout: int, global_timeout: int) -> dict:
    code, mode = extract_code(raw_output)
    sample, tests = evaluation_sample(row)
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(sample, code, per_test_timeout, queue))
    process.start(); process.join(global_timeout)
    if process.is_alive():
        process.kill(); process.join(10)
        return {"passed": False, "status": "TIMEOUT", "tests": tests, "postprocess_mode": mode}
    if queue.empty():
        return {"passed": False, "status": "EVALUATOR_ERROR", "tests": tests,
                "postprocess_mode": mode, "exit_code": process.exitcode}
    payload = queue.get()
    if "error" in payload:
        return {"passed": False, "status": "RUNTIME_OR_EVALUATOR_ERROR", "tests": tests,
                "postprocess_mode": mode, "error": payload["error"]}
    results = payload["results"]
    passed = len(results) == tests and all(results)
    return {"passed": passed, "status": "PASS" if passed else "WRONG_ANSWER_OR_RUNTIME",
            "tests": tests, "passed_tests": sum(results), "postprocess_mode": mode,
            "metadata": payload.get("metadata", {})}
