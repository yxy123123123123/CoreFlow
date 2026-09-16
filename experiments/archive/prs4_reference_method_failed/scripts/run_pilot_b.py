#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.lcb_eval import evaluate, extract_code, prompt_for
from scripts.common import (
    config,
    core_bank_path,
    gate_path,
    isvd_bank_path,
    load_jsonl,
    paths,
    request_json,
    sha256_file,
    wait_health,
    write_json,
)


def method_command(method: dict, cfg: dict, p: dict[str, Path]) -> list[str]:
    command = [
        str(p["runtime_python"]),
        str(ROOT / "scripts" / "serve_method.py"),
        "--model",
        str(p["model"]),
        "--asset-root",
        str(p["asset_root"]),
        "--official-root",
        str(p["official_root"]),
        "--adapter-order",
        ",".join(cfg["expert_order"]),
        "--gate",
        str(gate_path(cfg, p)),
        "--method",
        method["id"],
        "--port",
        str(method["port"]),
        "--max-input-tokens",
        str(cfg["pilot_B"]["max_input_tokens"]),
        "--output-length",
        str(cfg["pilot_B"]["max_new_tokens"]),
    ]
    if method["kind"] == "core":
        command.extend(["--bank-dir", str(core_bank_path(cfg, p))])
    elif method["kind"] == "isvd":
        command.extend(["--bank-dir", str(isvd_bank_path(cfg, p))])
    return command


def run_method(method: dict, rows: list[dict], cfg: dict, p: dict[str, Path]) -> dict:
    method_id = method["id"]
    final = p["results"] / "pilot_b" / method_id
    partial = final.with_name(final.name + ".partial")
    if (final / "COMPLETE.json").is_file():
        print(f"[SKIP] {final}", flush=True)
        return json.loads((final / "COMPLETE.json").read_text(encoding="utf-8"))
    if final.exists() or partial.exists():
        raise RuntimeError(f"Retained partial/final output exists for {method_id}: {final} / {partial}")
    partial.mkdir(parents=True)
    command = method_command(method, cfg, p)
    write_json(
        partial / "resolved_run.json",
        {
            "protocol_id": cfg["protocol_id"],
            "scheme": "B",
            "role": cfg["pilot_B"]["role"],
            "method": method_id,
            "paper_name": method["paper_name"],
            "seed": cfg["pilot_B"]["gate_seed"],
            "question_ids": [str(row["question_id"]) for row in rows],
            "command": command,
            "formal164_accessed": False,
        },
    )
    server_log = (partial / "server.log").open("w", encoding="utf-8", newline="\n")
    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    evaluated = []
    started = time.time()
    try:
        health = wait_health(f"http://127.0.0.1:{method['port']}/health", server)
        write_json(partial / "health.json", health)
        with (partial / "generations.jsonl").open("w", encoding="utf-8", newline="\n") as generation_file, (
            partial / "executor_results.jsonl"
        ).open("w", encoding="utf-8", newline="\n") as executor_file:
            for index, row in enumerate(rows):
                prompt = prompt_for(row)
                response = request_json(
                    f"http://127.0.0.1:{method['port']}/infer",
                    {"prompt": prompt, "seed": cfg["pilot_B"]["gate_seed"] * 100000 + index},
                    timeout=1800,
                )
                raw = response["answer"]
                code, postprocess_mode = extract_code(raw)
                timing = response.get("timing", {})
                generation = {
                    "question_id": str(row["question_id"]),
                    "order_index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "raw_output": raw,
                    "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "extracted_code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "postprocess_mode_preview": postprocess_mode,
                    "timing": timing,
                    "reached_max_new_tokens": int(timing.get("new_tokens", -1)) == cfg["pilot_B"]["max_new_tokens"],
                }
                generation_file.write(json.dumps(generation, ensure_ascii=False, sort_keys=True) + "\n")
                generation_file.flush()
                evaluation = evaluate(
                    row,
                    raw,
                    cfg["pilot_B"]["per_test_timeout_seconds"],
                    cfg["pilot_B"]["per_problem_global_timeout_seconds"],
                )
                item = {
                    "question_id": str(row["question_id"]),
                    "order_index": index,
                    "raw_output_sha256": generation["raw_output_sha256"],
                    "evaluation": evaluation,
                }
                executor_file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                executor_file.flush()
                evaluated.append(item)
                if index == 0 or (index + 1) % 4 == 0 or index + 1 == len(rows):
                    correct = sum(bool(x["evaluation"].get("passed")) for x in evaluated)
                    print(
                        f"[PRS4P] method={method_id} rows={index + 1}/{len(rows)} "
                        f"correct={correct} elapsed_min={(time.time() - started) / 60:.1f}",
                        flush=True,
                    )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()

    correct = sum(bool(x["evaluation"].get("passed")) for x in evaluated)
    status_counts = Counter(x["evaluation"].get("status", "MISSING") for x in evaluated)
    summary = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "scheme": "B",
        "role": cfg["pilot_B"]["role"],
        "method": method_id,
        "paper_name": method["paper_name"],
        "seed": cfg["pilot_B"]["gate_seed"],
        "rows": len(evaluated),
        "correct": correct,
        "strict_pass_at_1": correct / len(evaluated),
        "status_counts": dict(status_counts),
        "elapsed_seconds": time.time() - started,
        "formal164_accessed": False,
    }
    write_json(partial / "summary.json", summary)
    complete = {
        **summary,
        "generations_sha256": sha256_file(partial / "generations.jsonl"),
        "executor_results_sha256": sha256_file(partial / "executor_results.jsonl"),
    }
    write_json(partial / "COMPLETE.json", complete)
    shutil.move(str(partial), str(final))
    print(json.dumps({"status": "PASS", "method": method_id, "correct": correct, "rows": len(rows)}), flush=True)
    return complete


def main() -> None:
    cfg, p = config(), paths()
    if not (p["reports"] / "PILOT_SEAL.json").is_file():
        raise RuntimeError("Run seal before pilot-b")
    rows = load_jsonl(p["data"] / cfg["data"]["pilot_file"])
    if len(rows) != cfg["data"]["pilot_rows"]:
        raise ValueError(f"Expected {cfg['data']['pilot_rows']} rows, found {len(rows)}")
    method_lookup = {item["id"]: item for item in cfg["methods"]}
    completed = []
    for method_id in cfg["pilot_B"]["method_order"]:
        completed.append(run_method(method_lookup[method_id], rows, cfg, p))
    write_json(
        p["reports"] / "pilot_b_complete.json",
        {
            "status": "PASS",
            "protocol_id": cfg["protocol_id"],
            "methods": [item["method"] for item in completed],
            "records": sum(item["rows"] for item in completed),
            "expected_records": cfg["pilot_B"]["expected_records"],
            "formal164_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
