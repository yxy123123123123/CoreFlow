#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import bank_path, config, gate_path, paths, request_json, sha256_file, wait_health, write_json


def snapshot(device):
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=index,name,uuid,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.memory,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True, check=False)
    return {"values_csv": query.stdout.strip(), "pmon": processes.stdout.splitlines(), "captured_unix": time.time()}


def choose_batch(rows, index, size):
    start = (index * size) % len(rows)
    return [rows[(start + offset) % len(rows)] for offset in range(size)]


def scalar_stats(values):
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(rows):
    keys = [
        "decode_tokens_per_second",
        "total_tokens_per_second",
        "engine_wall_total_seconds",
        "engine_cuda_total_seconds",
        "prefill_wall_seconds",
        "decode_wall_seconds",
        "server_ttft_seconds",
        "tpot_seconds",
        "client_total_seconds",
    ]
    output = {}
    for key in keys:
        values = [float(row["timing"][key]) for row in rows if row["timing"].get(key) is not None]
        output[key] = scalar_stats(values)
    for key in (
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "startup_resident_allocated_bytes",
        "startup_resident_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
    ):
        output[key] = max(int(row["timing"][key]) for row in rows)
    return output


def main():
    parser = argparse.ArgumentParser()
    for name in ("method", "workload", "block-id", "output"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("order-position", "batch-size", "input-length", "output-length", "warmup", "measurements", "port"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    cfg, p = config(), paths()
    seal = p["reports"] / "PROTOCOL_SEAL.json"
    if not seal.is_file() or json.loads(seal.read_text(encoding="utf-8")).get("status") != "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS":
        raise RuntimeError("Run seal first")
    final = Path(args.output)
    partial = Path(str(final) + ".partial")
    oom = Path(str(final) + ".oom")
    if (final / "COMPLETE.json").is_file():
        print(f"[SKIP COMPLETE] {final}")
        return
    if (oom / "OOM.json").is_file():
        print(f"[SKIP RECORDED OOM] {oom}")
        raise SystemExit(20)
    if final.exists() or partial.exists() or oom.exists():
        raise RuntimeError(f"Retained partial/final output exists: {final}")
    partial.mkdir(parents=True)
    input_path = p["data"] / f"effective_inputs_l{args.input_length}.json"
    inputs = json.loads(input_path.read_text(encoding="utf-8"))
    if any(len(item["input_ids"]) != args.input_length or sum(item["attention_mask"]) != args.input_length for item in inputs):
        raise ValueError("True effective input contract failed")
    command = [
        str(p["runtime_python"]),
        str(ROOT / "scripts" / "serve_method.py"),
        "--model", str(p["model"]),
        "--asset-root", str(p["asset_root"]),
        "--official-root", str(p["official_root"]),
        "--adapter-order", ",".join(cfg["expert_order"]),
        "--gate", str(gate_path(cfg, p)),
        "--method", args.method,
        "--port", str(args.port),
        "--max-input-tokens", str(args.input_length),
        "--output-length", str(args.output_length),
        "--fixed-output",
    ]
    if args.method == "coreflow_q185_frozen":
        command += ["--bank-dir", str(bank_path(cfg, p, "k5_formal", "core"))]
    if args.method == "isvd_legacy_padded_matched_q185":
        command += ["--bank-dir", str(bank_path(cfg, p, "k5_formal", "isvd"))]
    write_json(
        partial / "resolved_run.json",
        {
            "protocol_id": cfg["protocol_id"],
            "command": command,
            "block_id": args.block_id,
            "order_position": args.order_position,
            "input_sha256": sha256_file(input_path),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    server_log = (partial / "server.log").open("w", encoding="utf-8", newline="\n")
    before = snapshot(p["cuda_device"])
    startup = time.perf_counter()
    process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
    measured = []
    try:
        health = wait_health(f"http://127.0.0.1:{args.port}/health", process)
        client_startup = time.perf_counter() - startup
        for index in range(args.warmup):
            items = choose_batch(inputs, index, args.batch_size)
            request_json(
                f"http://127.0.0.1:{args.port}/infer",
                {"input_ids": [item["input_ids"] for item in items], "attention_mask": [item["attention_mask"] for item in items], "seed": 930000 + index},
            )
        for index in range(args.measurements):
            items = choose_batch(inputs, index, args.batch_size)
            started = time.perf_counter()
            response = request_json(
                f"http://127.0.0.1:{args.port}/infer",
                {"input_ids": [item["input_ids"] for item in items], "attention_mask": [item["attention_mask"] for item in items], "seed": 940000 + index},
            )
            elapsed = time.perf_counter() - started
            if response["input_tokens_per_sequence"] != args.input_length or response["new_tokens_per_sequence"] != args.output_length:
                raise ValueError("Fixed-length response contract failed")
            timing = dict(response["timing"])
            timing["client_total_seconds"] = elapsed
            timing["tpot_seconds"] = (
                float(timing["decode_wall_seconds"]) / (args.output_length - 1)
                if timing.get("decode_wall_seconds") is not None and args.output_length > 1
                else None
            )
            row = {
                "protocol_id": cfg["protocol_id"],
                "workload": args.workload,
                "block_id": args.block_id,
                "order_position": args.order_position,
                "method": args.method,
                "request_index": index,
                "batch_size": args.batch_size,
                "input_length": args.input_length,
                "output_length": args.output_length,
                "task_ids": [item["task_id"] for item in items],
                "timing": timing,
                "generated_token_ids_sha256": response["generated_token_ids_sha256"],
            }
            measured.append(row)
            with (partial / "measurements.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[PRS3A] workload={args.workload} block={args.block_id} method={args.method} "
                f"measured={index + 1}/{args.measurements} decode_tok_s={timing['decode_tokens_per_second']:.3f}",
                flush=True,
            )
    except Exception as exc:
        server_log.flush()
        log_text = (partial / "server.log").read_text(encoding="utf-8", errors="replace")
        if "out of memory" in (repr(exc) + "\n" + log_text).lower():
            write_json(
                partial / "OOM.json",
                {
                    "status": "OOM",
                    "protocol_id": cfg["protocol_id"],
                    "method": args.method,
                    "workload": args.workload,
                    "block_id": args.block_id,
                    "batch_size": args.batch_size,
                    "input_length": args.input_length,
                    "output_length": args.output_length,
                    "error": repr(exc),
                    "gpu_before": before,
                },
            )
            shutil.move(str(partial), str(oom))
            print(json.dumps({"status": "OOM", "method": args.method, "workload": args.workload, "block": args.block_id}))
            raise SystemExit(20)
        write_json(partial / "PARTIAL.json", {"status": "PARTIAL", "error": repr(exc)})
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        server_log.close()
    summary = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "method": args.method,
        "workload": args.workload,
        "block_id": args.block_id,
        "order_position": args.order_position,
        "requests": len(measured),
        "warmup": args.warmup,
        "measurements": args.measurements,
        "batch_size": args.batch_size,
        "input_length": args.input_length,
        "output_length": args.output_length,
        "all_attention_mask_tokens_active": True,
        "client_startup_to_health_seconds": client_startup,
        "server_health": health,
        "gpu_before": before,
        "gpu_after": snapshot(p["cuda_device"]),
        "statistics": summarize(measured),
    }
    write_json(partial / "summary.json", summary)
    write_json(
        partial / "COMPLETE.json",
        {
            "status": "PASS",
            "measurements_sha256": sha256_file(partial / "measurements.jsonl"),
            "summary_sha256": sha256_file(partial / "summary.json"),
        },
    )
    shutil.move(str(partial), str(final))
    print(json.dumps({"status": "PASS", "method": args.method, "workload": args.workload, "block": args.block_id}))


if __name__ == "__main__":
    main()
