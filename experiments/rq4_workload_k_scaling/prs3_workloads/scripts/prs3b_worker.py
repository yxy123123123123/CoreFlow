#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.io import load_adapter
from coreflow.runtime import coreflow_delta, isvd_delta
from scripts.common import bank_path, config, expert_path, load_json, paths, sha256_file, write_json


METHODS = ("full_legacy_original", "coreflow_q185_frozen", "isvd_legacy_padded_matched_q185")


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def gpu_snapshot(device):
    query = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=index,name,uuid,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"values_csv": query.stdout.strip(), "captured_unix": time.time()}


def load_full_modules(cfg, p, pool):
    loaded = {}
    for name in cfg["prs3b"]["pools"][pool]:
        _, loaded[name] = load_adapter(expert_path(cfg, p, name))
    return loaded


def full_runtime(source_modules, experts, module_name, device):
    modules = [source_modules[name][module_name] for name in experts]
    a_values = [item.a.to(device=device, dtype=torch.bfloat16).contiguous() for item in modules]
    b_values = [item.b.to(device=device, dtype=torch.bfloat16).contiguous() for item in modules]
    scales = [float(item.scaling) for item in modules]
    in_features = int(a_values[0].shape[1])
    out_features = int(b_values[0].shape[0])

    def execute(x, weights):
        output = torch.zeros((x.shape[0], out_features), dtype=x.dtype, device=x.device)
        for index, (a, b, scale) in enumerate(zip(a_values, b_values, scales)):
            output.add_(((x @ a.T) @ b.T) * (weights[:, index : index + 1] * scale))
        return output

    logical = sum(a.numel() + b.numel() for a, b in zip(a_values, b_values))
    return execute, in_features, {
        "logical_tensor_elements": logical,
        "stored_tensor_elements": logical,
        "executed_tensor_elements": logical,
        "resident_selected_module_bytes": sum(t.numel() * t.element_size() for t in a_values + b_values),
        "logical_expert_branches_per_invocation": len(experts),
        "logical_matmuls_per_invocation": 2 * len(experts),
    }


def core_runtime(cfg, p, pool, module_name, device):
    from safetensors import safe_open

    root = bank_path(cfg, p, pool, "core")
    bank_cfg = load_json(root / "core_config.json")
    item = next(value for value in bank_cfg["modules"] if value["name"] == module_name)
    prefix = item["tensor_prefix"]
    with safe_open(str(root / "core_bank.safetensors"), framework="pt", device="cpu") as handle:
        u = handle.get_tensor(f"{prefix}.U").to(device=device, dtype=torch.bfloat16).contiguous()
        v = handle.get_tensor(f"{prefix}.V").to(device=device, dtype=torch.bfloat16).contiguous()
        cores = handle.get_tensor(f"{prefix}.C").to(device=device, dtype=torch.bfloat16).contiguous()

    def execute(x, weights):
        return coreflow_delta(x, weights, u, v, cores)

    elements = u.numel() + v.numel() + cores.numel()
    return execute, int(v.shape[0]), {
        "logical_tensor_elements": elements,
        "stored_tensor_elements": elements,
        "executed_tensor_elements": elements,
        "resident_selected_module_bytes": sum(t.numel() * t.element_size() for t in (u, v, cores)),
        "logical_expert_branches_per_invocation": 1,
        "logical_matmuls_per_invocation": 3,
    }


def isvd_runtime(cfg, p, pool, module_name, device):
    from safetensors import safe_open

    root = bank_path(cfg, p, pool, "isvd")
    bank_cfg = load_json(root / "isvd_config.json")
    item = next(value for value in bank_cfg["modules"] if value["name"] == module_name)
    prefix = item["tensor_prefix"]
    with safe_open(str(root / "isvd_bank.safetensors"), framework="pt", device="cpu") as handle:
        b = handle.get_tensor(f"{prefix}.B").to(device=device, dtype=torch.bfloat16).contiguous()
        v = handle.get_tensor(f"{prefix}.V").to(device=device, dtype=torch.bfloat16).contiguous()
    ranks = torch.tensor(item["expert_ranks"], device=device, dtype=torch.long)

    def execute(x, weights):
        return isvd_delta(x, weights, b, v, ranks)

    logical = sum((int(item["in_features"]) + int(item["out_features"])) * int(rank) for rank in item["expert_ranks"])
    stored = b.numel() + v.numel()
    return execute, int(v.shape[1]), {
        "logical_tensor_elements": logical,
        "stored_tensor_elements": stored,
        "executed_tensor_elements": stored,
        "resident_selected_module_bytes": sum(t.numel() * t.element_size() for t in (b, v, ranks)),
        "logical_expert_branches_per_invocation": 1,
        "logical_matmuls_per_invocation": 2,
        "effective_ranks": [int(value) for value in item["expert_ranks"]],
        "padded_max_rank": int(item["max_rank"]),
    }


def timed(execute, x, weights, warmup, iterations):
    for _ in range(warmup):
        execute(x, weights)
    torch.cuda.synchronize()
    wall, cuda = [], []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        execute(x, weights)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start_wall) * 1e6)
        cuda.append(float(start.elapsed_time(end)) * 1000.0)
    return wall, cuda


def profile_cell(execute, x, weights, label, iterations):
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False) as prof:
        with record_function(label):
            for _ in range(iterations):
                execute(x, weights)
            torch.cuda.synchronize()
    averages = list(prof.key_averages())
    events = list(prof.events())
    cuda_events = [event for event in events if "CUDA" in str(getattr(event, "device_type", "")).upper()]
    scope = next((item for item in averages if item.key == label), None)
    return {
        "profile_iterations": iterations,
        "operator_calls": int(sum(item.count for item in averages if not item.key.startswith("ProfilerStep"))),
        "cuda_kernel_events": len(cuda_events),
        "adapter_scope_cuda_time_us": float(getattr(scope, "device_time_total", 0.0) if scope is not None else 0.0),
        "self_cuda_time_total_us": float(sum(getattr(item, "self_device_time_total", 0.0) for item in averages)),
        "kernel_name_sample": [str(getattr(event, "name", "")) for event in cuda_events[:20]],
        "chrome_trace_exported": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--launch", type=int, required=True)
    parser.add_argument("--sentinel-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg, p = config(), paths()
    seal = load_json(p["reports"] / "PROTOCOL_SEAL.json")
    if seal.get("status") != "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS":
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(int(cfg["prs3b"]["seed"]) + args.launch)
    torch.cuda.manual_seed_all(int(cfg["prs3b"]["seed"]) + args.launch)
    experts = cfg["prs3b"]["pools"][args.pool]
    source_modules = load_full_modules(cfg, p, args.pool) if args.method == "full_legacy_original" else None
    token_blocks = [int(cfg["prs3b"]["cv_sentinel_token_block"])] if args.sentinel_only else cfg["prs3b"]["token_blocks"]
    rows = []
    before = gpu_snapshot(p["cuda_device"])
    started = time.time()
    try:
        for module_index, module_name in enumerate(cfg["prs3b"]["modules"]):
            if args.method == "full_legacy_original":
                execute, in_features, metadata = full_runtime(source_modules, experts, module_name, device)
            elif args.method == "coreflow_q185_frozen":
                execute, in_features, metadata = core_runtime(cfg, p, args.pool, module_name, device)
            else:
                execute, in_features, metadata = isvd_runtime(cfg, p, args.pool, module_name, device)
            for token_block in token_blocks:
                generator = torch.Generator(device="cuda").manual_seed(
                    int(cfg["prs3b"]["seed"]) + args.launch * 100000 + module_index * 1000 + int(token_block)
                )
                x = torch.randn(int(token_block), in_features, generator=generator, device=device, dtype=torch.bfloat16)
                weights = torch.full((int(token_block), len(experts)), 1.0 / len(experts), device=device, dtype=torch.bfloat16)
                startup_allocated = int(torch.cuda.memory_allocated())
                startup_reserved = int(torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats()
                wall_values, cuda_values = timed(execute, x, weights, int(cfg["prs3b"]["warmup"]), int(cfg["prs3b"]["iterations"]))
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
                profiler = None
                if not args.sentinel_only and int(token_block) in cfg["prs3b"]["profiler_token_blocks"]:
                    profiler = profile_cell(
                        execute,
                        x,
                        weights,
                        f"prs3b_{args.method}_adapter_scope",
                        int(cfg["prs3b"]["profiler_iterations"]),
                    )
                row = {
                    "protocol_id": cfg["protocol_id"],
                    "scope": cfg["prs3b"]["scope_label"],
                    "pool": args.pool,
                    "k": len(experts),
                    "method": args.method,
                    "launch": args.launch,
                    "sentinel_only": args.sentinel_only,
                    "module": module_name,
                    "token_block": int(token_block),
                    "gate_kind": cfg["prs3b"]["gate_kind"],
                    "dtype": cfg["prs3b"]["dtype"],
                    "warmup": cfg["prs3b"]["warmup"],
                    "iterations": cfg["prs3b"]["iterations"],
                    "wall_microseconds": stats(wall_values),
                    "cuda_event_microseconds": stats(cuda_values),
                    "effective_tokens_per_second": 1e6 * int(token_block) / statistics.median(cuda_values),
                    "startup_resident_allocated_bytes": startup_allocated,
                    "startup_resident_reserved_bytes": startup_reserved,
                    "absolute_peak_allocated_bytes": peak_allocated,
                    "absolute_peak_reserved_bytes": peak_reserved,
                    "incremental_peak_allocated_bytes": max(0, peak_allocated - startup_allocated),
                    "incremental_peak_reserved_bytes": max(0, peak_reserved - startup_reserved),
                    "adapter_scope_invocations_measured": int(cfg["prs3b"]["iterations"]),
                    "profiler": profiler,
                    **metadata,
                }
                rows.append(row)
                print(
                    f"[PRS3B] pool={args.pool} method={args.method} launch={args.launch} "
                    f"module={module_name} T={token_block} median_cuda_us={row['cuda_event_microseconds']['median']:.3f}",
                    flush=True,
                )
            del execute
            torch.cuda.empty_cache()
        measurement = partial / "measurements.jsonl"
        with measurement.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "status": "PASS",
            "protocol_id": cfg["protocol_id"],
            "pool": args.pool,
            "k": len(experts),
            "method": args.method,
            "launch": args.launch,
            "sentinel_only": args.sentinel_only,
            "rows": len(rows),
            "gpu_before": before,
            "gpu_after": gpu_snapshot(p["cuda_device"]),
        }
        write_json(partial / "summary.json", summary)
        write_json(
            partial / "COMPLETE.json",
            {
                "status": "PASS",
                "elapsed_seconds": time.time() - started,
                "measurements_sha256": sha256_file(measurement),
                "summary_sha256": sha256_file(partial / "summary.json"),
            },
        )
        shutil.move(str(partial), str(final))
    except Exception as exc:
        if "out of memory" in repr(exc).lower():
            write_json(partial / "OOM.json", {"status": "OOM", "pool": args.pool, "method": args.method, "launch": args.launch, "error": repr(exc)})
            shutil.move(str(partial), str(oom))
            print(json.dumps({"status": "OOM", "pool": args.pool, "method": args.method, "launch": args.launch}))
            raise SystemExit(20)
        write_json(partial / "PARTIAL.json", {"status": "PARTIAL", "pool": args.pool, "method": args.method, "launch": args.launch, "error": repr(exc)})
        raise
    print(json.dumps({"status": "PASS", "pool": args.pool, "method": args.method, "launch": args.launch, "rows": len(rows)}))


if __name__ == "__main__":
    main()
