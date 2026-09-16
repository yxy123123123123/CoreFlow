#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunningJob:
    method: str
    device: str
    process: subprocess.Popen
    log_handle: object
    reader: Thread


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_methods(config: dict, phase: str, activation: Path | None) -> list[str]:
    methods = list(config["phase_orders"][phase])
    if phase == "q224_conditional":
        if activation is None:
            raise ValueError("q224_conditional requires --activation")
        tasks = set(load_json(activation)["tasks"])
        methods = [name for name in methods if name.split("_", 1)[0] in tasks]
    return methods


def dataset_for(method: str, args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    task = method.split("_", 1)[0]
    if task == "code":
        candidate = Path(args.data_root) / "mbppplus_dev128.jsonl"
        if candidate.exists():
            return candidate
        return Path(args.data_root) / "code_formal_300.jsonl"
    if task == "math":
        candidate = Path(args.data_root) / "cmath_dev128.jsonl"
        if candidate.exists():
            return candidate
        return Path(args.data_root) / "math_formal_300.jsonl"
    raise ValueError(f"Cannot infer dataset for method: {method}")


def output_group_for(method: str, args: argparse.Namespace) -> str:
    if args.output_group:
        return args.output_group
    if args.phase == "q224_conditional":
        return "q224"
    return str(args.phase)


def floor_decision(path: str | None) -> str | None:
    if not path:
        return None
    return str(load_json(Path(path))["decision"])


def reader_thread(method: str, device: str, process: subprocess.Popen, log_handle) -> Thread:
    def run() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip("\n")
            log_handle.write(text + "\n")
            log_handle.flush()
            print(f"[gpu{device} {method}] {text}", flush=True)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread


def launch(method: str, device: str, port: int, args: argparse.Namespace) -> RunningJob:
    dataset = dataset_for(method, args)
    group = output_group_for(method, args)
    output = Path(args.result_root) / group / method
    log_dir = Path(args.formal_work_root) / "logs" / "formal_parallel"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{group}_{method}.gpu{device}.log"
    command = [
        args.python,
        str(ROOT / "scripts" / "run_formal_method.py"),
        "--formal-work-root",
        args.formal_work_root,
        "--source-work-root",
        args.source_work_root,
        "--model",
        args.model,
        "--config",
        args.config,
        "--asset-lock",
        args.asset_lock,
        "--frozen-data",
        str(dataset),
        "--method",
        method,
        "--output",
        str(output),
        "--port",
        str(port),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.isvd_asset_root:
        command.extend(["--isvd-asset-root", args.isvd_asset_root])
    if args.isvd_lock:
        command.extend(["--isvd-lock", args.isvd_lock])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    log_handle = log_path.open("w", encoding="utf-8", newline="\n")
    log_handle.write(json.dumps({"method": method, "device": device, "port": port, "command": command}, sort_keys=True) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    reader = reader_thread(method, device, process, log_handle)
    print(f"[START] method={method} gpu={device} port={port}", flush=True)
    return RunningJob(method=method, device=device, process=process, log_handle=log_handle, reader=reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--methods")
    parser.add_argument("--activation")
    parser.add_argument("--output-group")
    parser.add_argument("--dataset")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--isvd-asset-root")
    parser.add_argument("--isvd-lock")
    parser.add_argument("--base-port", type=int, default=5900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--floor-decision")
    parser.add_argument("--skip-code-if-stress-only", action="store_true")
    args = parser.parse_args()

    if not args.phase and not args.methods:
        raise ValueError("Provide --phase or --methods")
    config = load_json(Path(args.config))
    if args.methods:
        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    else:
        methods = phase_methods(config, str(args.phase), Path(args.activation) if args.activation else None)
    if args.skip_code_if_stress_only and floor_decision(args.floor_decision) == "CODE_STRESS_ONLY":
        skipped = [method for method in methods if method.startswith("code_")]
        methods = [method for method in methods if not method.startswith("code_")]
        for method in skipped:
            print(f"[STOP-RULE] skip formal-code method {method}; code split remains closed.", flush=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices cannot be empty")
    if not methods:
        print(json.dumps({"status": "PASS", "launched": 0}, sort_keys=True), flush=True)
        return

    queue = list(methods)
    running: list[RunningJob] = []
    completed: list[str] = []
    failures: list[dict] = []
    next_port = args.base_port
    started = time.time()
    while queue or running:
        free_devices = [device for device in devices if all(job.device != device for job in running)]
        while queue and free_devices and not failures:
            device = free_devices.pop(0)
            method = queue.pop(0)
            running.append(launch(method, device, next_port, args))
            next_port += 1
        time.sleep(2)
        still_running: list[RunningJob] = []
        for job in running:
            code = job.process.poll()
            if code is None:
                still_running.append(job)
                continue
            job.reader.join(timeout=10)
            job.log_handle.close()
            if code == 0:
                completed.append(job.method)
                print(f"[DONE] method={job.method} gpu={job.device}", flush=True)
            else:
                failures.append({"method": job.method, "device": job.device, "exit_code": code})
                print(f"[FAIL] method={job.method} gpu={job.device} exit={code}", flush=True)
        running = still_running
        if failures and queue:
            print("[STOP] a method failed; no new methods will be launched. Running jobs are left to finish.", flush=True)
            queue.clear()
    report = {
        "status": "PASS" if not failures else "FAIL",
        "completed": completed,
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
