from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CFG, group_root, write_json
from quality_core_worker import install, load_model


def power_watts() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "0"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(len(ordered) - 1, int(pos) + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--method", choices=["full", "core"], required=True)
    ap.add_argument("--q", type=int, default=185)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("SERVICE_CONCURRENCY", "1")))
    ap.add_argument("--input-tokens", type=int, default=int(os.environ.get("SERVICE_INPUT_TOKENS", "512")))
    ap.add_argument("--output-tokens", type=int, default=int(os.environ.get("SERVICE_OUTPUT_TOKENS", "128")))
    ap.add_argument("--measured-requests", type=int, default=int(os.environ.get("SERVICE_MEASURED_REQUESTS", "50")))
    args = ap.parse_args()
    if args.concurrency < 1 or args.measured_requests < 1:
        raise ValueError("concurrency and measured-requests must be positive")
    out = group_root(args.group) / "system" / ("independent_full" if args.method == "full" else f"core_q{args.q}")
    out.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    tok, model = load_model(args.group)
    hooks = install(model, args.group, args.method, args.q)
    startup_seconds = time.perf_counter() - load_started
    text = ("This is a fixed serving benchmark prompt. Generate a concise deterministic continuation. " * 512).strip()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.input_tokens, add_special_tokens=True)
    enc = {key: value.to("cuda:0") for key, value in enc.items()}
    if enc["input_ids"].shape[-1] != args.input_tokens:
        raise RuntimeError(f"Input length is not exact: {enc['input_ids'].shape[-1]} != {args.input_tokens}")
    warmup = int(CFG["efficiency"].get("warmup_requests", 5))
    values: list[float] = []
    latencies: list[float] = []
    powers: list[float] = []
    try:
        for _ in range(warmup):
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=args.output_tokens, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        overall_started = time.perf_counter()
        batches = (args.measured_requests + args.concurrency - 1) // args.concurrency
        completed = 0
        for batch_index in range(batches):
            batch_started = time.perf_counter()
            for _ in range(min(args.concurrency, args.measured_requests - completed)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    model.generate(**enc, max_new_tokens=args.output_tokens, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                values.append(args.output_tokens / elapsed)
                reading = power_watts()
                if reading is not None:
                    powers.append(reading)
                completed += 1
            print(f"[SERVICE] method={args.method} concurrency={args.concurrency} request={completed}/{args.measured_requests} tok_s={values[-1]:.3f}", flush=True)
            _ = time.perf_counter() - batch_started
        overall_elapsed = time.perf_counter() - overall_started
        average_power = sum(powers) / len(powers) if powers else None
        metrics = {
            "status": "PASS",
            "method": "independent_full" if args.method == "full" else f"core_q{args.q}",
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency_label": args.concurrency,
            "measurement_scope": "single-worker queued-load; not continuous batching",
            "startup_health_ready_seconds": startup_seconds,
            "latencies_seconds": latencies,
            "decode_tokens_per_second": values,
            "mean_decode_tokens_per_second": sum(values) / len(values),
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "p99_latency_seconds": percentile(latencies, 0.99),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "average_power_watts": average_power,
            "joules_per_request": (average_power * overall_elapsed / len(latencies)) if average_power is not None else None,
            "joules_per_token": (average_power * overall_elapsed / (len(latencies) * args.output_tokens)) if average_power is not None else None,
        }
        write_json(out / "metrics.json", metrics)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
