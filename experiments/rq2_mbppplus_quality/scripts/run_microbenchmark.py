#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import sha256_file
from coreflow.statistics import quantile
from scripts.run_formal_method import method_spec, request_json, wait_health


def summarize(rows: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = [float(row["timing"][key]) for row in rows if row["timing"].get(key) is not None]
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "median": quantile(values, 0.5),
            "p90": quantile(values, 0.9),
            "min": min(values),
            "max": max(values),
        }
    return {
        "requests": len(rows),
        "latency_seconds": stats("total_latency_seconds"),
        "ttft_seconds": stats("ttft_seconds"),
        "tokens_per_second_total": stats("tokens_per_second_total"),
        "decode_tokens_per_second": stats("decode_tokens_per_second"),
        "peak_memory_allocated_bytes": max(row["timing"]["peak_memory_allocated_bytes"] for row in rows),
        "peak_memory_reserved_bytes": max(row["timing"]["peak_memory_reserved_bytes"] for row in rows),
    }


def run_method(args, cfg: dict, lock: dict, name: str, output_root: Path, port: int) -> dict:
    spec = method_spec(name, cfg)
    final = output_root / name
    partial = output_root / f"{name}.partial"
    if (final / "COMPLETE.json").exists():
        print(f"[SKIP] {final}", flush=True)
        return load_json(final / "COMPLETE.json")
    if final.exists() or partial.exists():
        raise RuntimeError(f"Retained microbenchmark output exists for {final}")
    partial.mkdir(parents=True)

    formal_work = Path(args.formal_work_root)
    source_work = Path(args.source_work_root)
    assets = formal_work / "m2_assets"
    gate = assets / "gates" / f"{spec['task']}_{spec['pool']}_seed{spec['seed']}.pt"
    if sha256_file(gate) != lock["gates"][gate.name]:
        raise ValueError(f"Gate hash mismatch: {gate.name}")
    bank_dir = None
    if spec["kind"] == "core":
        bank_dir = assets / "core_banks" / spec["pool"] / f"q{spec['q']}_bf16"
        expected = lock["banks"][spec["pool"]][str(spec["q"])]["core_bank_sha256"]
        if sha256_file(bank_dir / "core_bank.safetensors") != expected:
            raise ValueError(f"Core bank hash mismatch: {name}")

    fixed_new = int(cfg["microbenchmark"]["fixed_new_tokens"])
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_formal_method.py"),
        "--model", args.model,
        "--asset-root", str(source_work / "official_assets"),
        "--official-root", str(source_work / "vendor" / "LoRAFlow"),
        "--adapter-order", ",".join(spec["adapter_order"]),
        "--gate", str(gate),
        "--method-kind", spec["kind"],
        "--port", str(port),
        "--max-input-tokens", str(max(cfg["microbenchmark"]["input_token_lengths"])),
        "--max-new-tokens", str(fixed_new),
        "--min-new-tokens", str(fixed_new),
        "--do-sample", "false",
        "--temperature", "1.0",
        "--top-p", "1.0",
    ]
    if bank_dir is not None:
        command.extend(["--bank-dir", str(bank_dir)])
    write_json(partial / "resolved_run.json", {"method": spec, "command": command})
    log = (partial / "server.log").open("w", encoding="utf-8", newline="\n")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    started = time.time()
    all_rows = []
    try:
        health = wait_health(f"http://127.0.0.1:{port}/health", process)
        with (partial / "measurements.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for input_length in cfg["microbenchmark"]["input_token_lengths"]:
                for index in range(int(cfg["microbenchmark"]["warmup_repetitions"])):
                    request_json(
                        f"http://127.0.0.1:{port}/infer",
                        {"fixed_input_tokens": input_length, "seed": 800000 + input_length * 100 + index},
                    )
                for index in range(int(cfg["microbenchmark"]["measured_repetitions"])):
                    response = request_json(
                        f"http://127.0.0.1:{port}/infer",
                        {"fixed_input_tokens": input_length, "seed": 810000 + input_length * 100 + index},
                    )
                    timing = response["timing"]
                    if timing["input_tokens"] != input_length or timing["new_tokens"] != fixed_new:
                        raise ValueError(
                            f"Fixed-length contract failed for {name}: "
                            f"input={timing['input_tokens']}, new={timing['new_tokens']}"
                        )
                    row = {"method": name, "input_tokens": input_length, "repetition": index, "timing": timing}
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    all_rows.append(row)
                    if (index + 1) % 10 == 0:
                        print(
                            f"[BENCH] method={name} input={input_length} "
                            f"measured={index + 1}/{cfg['microbenchmark']['measured_repetitions']}",
                            flush=True,
                        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        log.close()

    by_length = {
        str(length): summarize([row for row in all_rows if row["input_tokens"] == length])
        for length in cfg["microbenchmark"]["input_token_lengths"]
    }
    write_json(partial / "summary.json", {"method": name, "by_input_length": by_length})
    complete = {
        "status": "PASS",
        "method": name,
        "requests": len(all_rows),
        "elapsed_seconds": time.time() - started,
        "measurement_sha256": sha256_file(partial / "measurements.jsonl"),
        "summary_sha256": sha256_file(partial / "summary.json"),
        "server_health": health,
    }
    write_json(partial / "COMPLETE.json", complete)
    shutil.move(str(partial), str(final))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "formal_protocol.json"))
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=5710)
    args = parser.parse_args()
    cfg = load_json(args.config)
    lock = load_json(args.asset_lock)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed = {}
    for offset, name in enumerate(cfg["microbenchmark"]["methods"]):
        completed[name] = run_method(args, cfg, lock, name, root, args.port + offset)
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "fixed_new_tokens": cfg["microbenchmark"]["fixed_new_tokens"],
        "input_token_lengths": cfg["microbenchmark"]["input_token_lengths"],
        "warmup_repetitions": cfg["microbenchmark"]["warmup_repetitions"],
        "measured_repetitions": cfg["microbenchmark"]["measured_repetitions"],
        "methods": list(completed),
        "complete_markers": completed,
        "summaries": {
            name: load_json(root / name / "summary.json")
            for name in completed
        },
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "methods": len(completed), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
