#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.classeval_eval import evaluate_classeval_method
from coreflow.data_audit import load_jsonl
from coreflow.gate_io import load_json, write_json
from coreflow.io import sha256_file
from coreflow.m2_eval import evaluate_apps_program, evaluate_math_answer, evaluate_mbppplus_program


def request_json(url: str, payload: dict | None = None, timeout: float = 900.0) -> dict:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_health(url: str, process: subprocess.Popen, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            return request_json(url, timeout=3)
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy: {last_error}")


def method_spec(name: str, cfg: dict) -> dict:
    task = name.split("_", 1)[0]
    if task not in {"code", "math"}:
        raise ValueError(f"Method must start with code_ or math_: {name}")
    if "_task_only_" in name:
        kind = "task_only"
        k = 1
        q = None
        pool = None
        order = [task]
    else:
        if "_static_" in name:
            kind = "static"
        elif "_isvd_" in name:
            kind = "isvd"
        elif "_core_" in name:
            kind = "core"
        elif "_full_" in name:
            kind = "full"
        else:
            raise ValueError(f"Cannot infer method kind: {name}")
        tokens = name.split("_")
        k = int(next(token[1:] for token in tokens if token.startswith("k") and token[1:].isdigit()))
        q = 185 if "q185" in tokens else 224 if "q224" in tokens else None
        pool = f"k{k}_{task}"
        order = list(cfg["expert_pools"][pool])
    seed = int(next(token[4:] for token in name.split("_") if token.startswith("seed")))
    return {
        "name": name,
        "task": task,
        "kind": kind,
        "k": k,
        "q": q,
        "pool": pool,
        "adapter_order": order,
        "seed": seed,
    }


def row_prompt(row: dict, task: str, template: str) -> str:
    if row.get("evaluator") == "classeval_method":
        problem = row["problem"]
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"Invalid ClassEval prompt field for {row.get('task_id')}")
        return template.format(problem=problem)
    if task == "code":
        problem = row["problem"]
    else:
        problem = row.get("question", row.get("problem"))
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"Invalid {task} prompt field for {row.get('task_id')}")
    values = {"problem": problem, "entry_point": row.get("entry_point", "")}
    return template.format(**values)


def row_expected(row: dict, task: str):
    if row.get("evaluator") == "classeval_method":
        return None
    if task == "code":
        if row.get("evaluator") == "mbppplus_asserts":
            return row["hidden_test"]
        return row["io_cases"]
    answer = row.get("golden", row.get("answer"))
    if answer is None:
        raise KeyError(f"No math gold field for {row.get('task_id')}")
    return str(answer)


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        raise ValueError("Cannot summarize an empty run")
    status = Counter(row["evaluation"]["status"] for row in rows)
    passed = sum(bool(row["evaluation"]["passed"]) for row in rows)
    timings = [row["timing"] for row in rows]
    return {
        "rows": count,
        "correct": passed,
        "pass_at_1": passed / count,
        "status_counts": dict(sorted(status.items())),
        "input_tokens_mean": sum(item["input_tokens"] for item in timings) / count,
        "new_tokens_mean": sum(item["new_tokens"] for item in timings) / count,
        "latency_seconds_mean": sum(item["total_latency_seconds"] for item in timings) / count,
        "tokens_per_second_total_mean": sum(item["tokens_per_second_total"] for item in timings) / count,
        "decode_tokens_per_second_mean": (
            sum(item["decode_tokens_per_second"] for item in timings if item["decode_tokens_per_second"] is not None)
            / max(1, sum(item["decode_tokens_per_second"] is not None for item in timings))
        ),
        "peak_memory_allocated_bytes": max(item["peak_memory_allocated_bytes"] for item in timings),
        "peak_memory_reserved_bytes": max(item["peak_memory_reserved_bytes"] for item in timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "formal_protocol.json"))
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--isvd-asset-root")
    parser.add_argument("--isvd-lock")
    parser.add_argument("--frozen-data", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=5701)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_json(args.config)
    lock = load_json(args.asset_lock)
    spec = method_spec(args.method, cfg)
    rows = load_jsonl(args.frozen_data)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    template_file = cfg["prompts"][f"{spec['task']}_file"]
    template = (ROOT / "data" / template_file).read_text(encoding="utf-8")

    formal_work = Path(args.formal_work_root)
    source_work = Path(args.source_work_root)
    assets = formal_work / "m2_assets"
    gate = None
    bank_dir = None
    if spec["kind"] != "task_only":
        gate = assets / "gates" / f"{spec['task']}_{spec['pool']}_seed{spec['seed']}.pt"
        if sha256_file(gate) != lock["gates"][gate.name]:
            raise ValueError(f"Gate hash mismatch: {gate.name}")
    if spec["kind"] in {"core", "static"}:
        bank_dir = assets / "core_banks" / spec["pool"] / f"q{spec['q']}_bf16"
        expected = lock["banks"][spec["pool"]][str(spec["q"])]["core_bank_sha256"]
        if sha256_file(bank_dir / "core_bank.safetensors") != expected:
            raise ValueError(f"Core bank hash mismatch: {spec['pool']}/q{spec['q']}")
    elif spec["kind"] == "isvd":
        if not args.isvd_asset_root or not args.isvd_lock:
            raise ValueError("--isvd-asset-root and --isvd-lock are required for isvd methods")
        bank_dir = Path(args.isvd_asset_root) / spec["pool"] / "matched_q185_bf16"
        isvd_lock = load_json(args.isvd_lock)
        expected = isvd_lock["banks"][spec["pool"]]["isvd_bank_sha256"]
        if sha256_file(bank_dir / "isvd_bank.safetensors") != expected:
            raise ValueError(f"ISVD bank hash mismatch: {spec['pool']}")

    final = Path(args.output).resolve()
    partial = final.with_name(final.name + ".partial")
    if (final / "COMPLETE.json").exists():
        print(f"[SKIP] {final}", flush=True)
        return
    if final.exists() or partial.exists():
        raise RuntimeError(f"Retained partial/final output exists for {final}")
    partial.mkdir(parents=True)

    max_new = cfg["generation"][f"max_new_tokens_{spec['task']}"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_formal_method.py"),
        "--model", args.model,
        "--asset-root", str(source_work / "official_assets"),
        "--official-root", str(source_work / "vendor" / "LoRAFlow"),
        "--adapter-order", ",".join(spec["adapter_order"]),
        "--method-kind", spec["kind"],
        "--port", str(args.port),
        "--max-input-tokens", str(cfg["generation"]["max_input_tokens"]),
        "--max-new-tokens", str(max_new),
        "--do-sample", "true" if cfg["generation"]["do_sample"] else "false",
        "--temperature", str(cfg["generation"]["temperature"]),
        "--top-p", str(cfg["generation"]["top_p"]),
    ]
    if gate is not None:
        command.extend(["--gate", str(gate)])
    if bank_dir is not None:
        command.extend(["--bank-dir", str(bank_dir)])
    write_json(partial / "resolved_run.json", {
        "method": spec,
        "command": command,
        "frozen_data": str(Path(args.frozen_data).resolve()),
        "frozen_data_sha256": sha256_file(args.frozen_data),
        "row_count": len(rows),
        "classeval_rows": sum(1 for row in rows if row.get("evaluator") == "classeval_method"),
        "config_sha256": sha256_file(args.config),
        "asset_lock_sha256": sha256_file(args.asset_lock),
    })

    server_log = (partial / "server.log").open("w", encoding="utf-8", newline="\n")
    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    evaluated: list[dict] = []
    started = time.time()
    health = None
    try:
        health = wait_health(f"http://127.0.0.1:{args.port}/health", server)
        for index in range(int(cfg["generation"]["warmup_requests"])):
            request_json(
                f"http://127.0.0.1:{args.port}/infer",
                {"prompt": "warmup", "seed": 900000 + index},
            )
        classeval_raw: dict[str, str] = {}
        classeval_timing: dict[str, dict] = {}
        order_index: dict[str, int] = {}
        with (
            (partial / "generations.jsonl").open("w", encoding="utf-8", newline="\n") as gen_file,
            (partial / "executor_results.jsonl").open("w", encoding="utf-8", newline="\n") as eval_file,
        ):
            for index, row in enumerate(rows):
                prompt = row_prompt(row, spec["task"], template)
                response = request_json(
                    f"http://127.0.0.1:{args.port}/infer",
                    {"prompt": prompt, "seed": spec["seed"] * 100000 + index},
                )
                generation = {
                    "task_id": row["task_id"],
                    "order_index": index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "raw_output": response["answer"],
                    "timing": response["timing"],
                }
                gen_file.write(json.dumps(generation, ensure_ascii=False, sort_keys=True) + "\n")
                gen_file.flush()
                if row.get("evaluator") == "classeval_method":
                    classeval_raw[row["task_id"]] = generation["raw_output"]
                    classeval_timing[row["task_id"]] = generation["timing"]
                    order_index[row["task_id"]] = index
                    completed_gen = index + 1
                    if completed_gen == 1 or completed_gen % 10 == 0 or completed_gen == len(rows):
                        elapsed = time.time() - started
                        print(
                            f"[PROGRESS] method={args.method} generated={completed_gen}/{len(rows)} "
                            f"elapsed_min={elapsed / 60:.1f}",
                            flush=True,
                        )
                    continue
                expected = row_expected(row, spec["task"])
                if spec["task"] == "code" and row.get("evaluator") == "mbppplus_asserts":
                    evaluation = evaluate_mbppplus_program(
                        generation["raw_output"],
                        expected,
                        cfg["evaluator"],
                        row.get("entry_point", ""),
                    )
                elif spec["task"] == "code":
                    evaluation = evaluate_apps_program(generation["raw_output"], expected, cfg["evaluator"])
                else:
                    evaluation = evaluate_math_answer(generation["raw_output"], expected)
                result = {
                    "task_id": row["task_id"],
                    "order_index": index,
                    "timing": generation["timing"],
                    "raw_output_sha256": hashlib.sha256(generation["raw_output"].encode("utf-8")).hexdigest(),
                    "evaluation": evaluation,
                }
                eval_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                eval_file.flush()
                evaluated.append(result)
                completed = len(evaluated)
                if completed == 1 or completed % 10 == 0 or completed == len(rows):
                    correct = sum(bool(item["evaluation"]["passed"]) for item in evaluated)
                    elapsed = time.time() - started
                    print(
                        f"[PROGRESS] method={args.method} rows={completed}/{len(rows)} "
                        f"correct={correct} elapsed_min={elapsed / 60:.1f}",
                        flush=True,
                    )

            if classeval_raw:
                classes: dict[str, list[dict]] = {}
                for row in rows:
                    if row.get("evaluator") == "classeval_method":
                        classes.setdefault(row["class_id"], []).append(row)
                for class_id in sorted(classes):
                    class_rows = classes[class_id]
                    raw_outputs = {
                        row["entry_point"]: classeval_raw[row["task_id"]] for row in class_rows
                    }
                    for row in class_rows:
                        evaluation = evaluate_classeval_method(
                            raw_outputs,
                            row,
                            cfg["evaluator"],
                            timeout_seconds=int(cfg["evaluator"]["code_timeout_seconds"]),
                            cpu_seconds=int(cfg["evaluator"]["cpu_limit_seconds"]),
                            memory_mb=int(cfg["evaluator"]["address_space_limit_mb"]),
                            file_mb=int(cfg["evaluator"]["file_size_limit_mb"]),
                            process_limit=int(cfg["evaluator"]["process_limit"]),
                        )
                        task_id = row["task_id"]
                        result = {
                            "task_id": task_id,
                            "order_index": order_index[task_id],
                            "timing": classeval_timing[task_id],
                            "raw_output_sha256": hashlib.sha256(
                                classeval_raw[task_id].encode("utf-8")
                            ).hexdigest(),
                            "evaluation": evaluation,
                        }
                        eval_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                        eval_file.flush()
                        evaluated.append(result)
                        completed = len(evaluated)
                        if completed == 1 or completed % 10 == 0 or completed == len(rows):
                            correct = sum(bool(item["evaluation"]["passed"]) for item in evaluated)
                            elapsed = time.time() - started
                            print(
                                f"[PROGRESS] method={args.method} rows={completed}/{len(rows)} "
                                f"correct={correct} elapsed_min={elapsed / 60:.1f}",
                                flush=True,
                            )
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
        server_log.close()

    summary = summarize(evaluated)
    write_json(partial / "summary.json", summary)
    complete = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "method": args.method,
        "task": spec["task"],
        "kind": spec["kind"],
        "pool": spec["pool"],
        "adapter_order": spec["adapter_order"],
        "q": spec["q"],
        "seed": spec["seed"],
        "row_count": len(rows),
        "frozen_data_sha256": sha256_file(args.frozen_data),
        "asset_lock_sha256": sha256_file(args.asset_lock),
        "generation_sha256": sha256_file(partial / "generations.jsonl"),
        "executor_results_sha256": sha256_file(partial / "executor_results.jsonl"),
        "summary_sha256": sha256_file(partial / "summary.json"),
        "elapsed_seconds": time.time() - started,
        "server_health": health,
    }
    write_json(partial / "COMPLETE.json", complete)
    shutil.move(str(partial), str(final))
    print(json.dumps({
        "status": "PASS",
        "method": args.method,
        "correct": summary["correct"],
        "rows": len(rows),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
