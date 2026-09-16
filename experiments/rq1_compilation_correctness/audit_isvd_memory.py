#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import sha256_file
from scripts.run_formal_method import method_spec


def request_json(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as response:
        return json.load(response)


def wait_health(url: str, process: subprocess.Popen, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            return request_json(url)
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise TimeoutError(f"health timeout: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--base-port", type=int, default=6300)
    args = parser.parse_args()

    cfg = load_json(args.config)
    lock = load_json(args.asset_lock)
    formal_work = Path(args.formal_work_root)
    source = Path(args.source_work_root)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows = []
    for index, method in enumerate(methods):
        spec = method_spec(method, cfg)
        assets = formal_work / "m2_assets"
        gate = assets / "gates" / f"{spec['task']}_{spec['pool']}_seed{spec['seed']}.pt"
        if sha256_file(gate) != lock["gates"][gate.name]:
            raise ValueError(f"Gate hash mismatch: {gate.name}")
        bank_dir = None
        if spec["kind"] in {"core", "static"}:
            bank_dir = assets / "core_banks" / spec["pool"] / f"q{spec['q']}_bf16"
        elif spec["kind"] == "isvd":
            bank_dir = formal_work / "isvd_banks" / spec["pool"] / "matched_q185_bf16"
        port = args.base_port + index
        command = [
            sys.executable,
            str(ROOT / "scripts" / "serve_formal_method.py"),
            "--model", args.model,
            "--asset-root", str(source / "official_assets"),
            "--official-root", str(source / "vendor" / "LoRAFlow"),
            "--adapter-order", ",".join(spec["adapter_order"]),
            "--method-kind", spec["kind"],
            "--gate", str(gate),
            "--port", str(port),
            "--max-input-tokens", str(cfg["generation"]["max_input_tokens"]),
            "--max-new-tokens", "1",
            "--do-sample", "false",
            "--temperature", "1.0",
            "--top-p", "1.0",
        ]
        if bank_dir is not None:
            command.extend(["--bank-dir", str(bank_dir)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
        started = time.time()
        log_path = formal_work / "logs" / "memory_audit" / f"{method}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
            try:
                health = wait_health(f"http://127.0.0.1:{port}/health", proc)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        rows.append({
            "method": method,
            "kind": spec["kind"],
            "task": spec["task"],
            "pool": spec["pool"],
            "adapter_order": spec["adapter_order"],
            "startup_memory_allocated_bytes": health["startup_memory_allocated_bytes"],
            "startup_memory_reserved_bytes": health["startup_memory_reserved_bytes"],
            "server_health": health,
            "elapsed_seconds": time.time() - started,
            "log": str(log_path),
        })
        print(json.dumps({"method": method, "allocated_gb": rows[-1]["startup_memory_allocated_bytes"] / 1e9, "reserved_gb": rows[-1]["startup_memory_reserved_bytes"] / 1e9}, sort_keys=True), flush=True)

    by_pool: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_pool.setdefault(row["pool"], {})[row["kind"]] = row
    warnings = []
    for pool, items in by_pool.items():
        if "isvd" in items and "full" in items:
            if items["isvd"]["startup_memory_allocated_bytes"] > items["full"]["startup_memory_allocated_bytes"] * 1.05:
                warnings.append(f"{pool}: ISVD allocated memory exceeds Full by >5%")
            if items["isvd"]["startup_memory_reserved_bytes"] > items["full"]["startup_memory_reserved_bytes"] * 1.10:
                warnings.append(f"{pool}: ISVD reserved memory exceeds Full by >10%")
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(args.config),
        "device": args.device,
        "audit_type": "independent_process_startup_memory_smoke",
        "rows": rows,
        "warnings": warnings,
        "interpretation": "Warnings do not block formal generation; they flag fairness issues for final analysis.",
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "methods": len(rows), "warnings": warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
