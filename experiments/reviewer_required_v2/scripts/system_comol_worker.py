from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import CFG, MODEL, group_root, write_json
from train_independent import find_adapter


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
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(len(ordered) - 1, int(pos) + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("SERVICE_CONCURRENCY", "1")))
    ap.add_argument("--input-tokens", type=int, default=int(os.environ.get("SERVICE_INPUT_TOKENS", "512")))
    ap.add_argument("--output-tokens", type=int, default=int(os.environ.get("SERVICE_OUTPUT_TOKENS", "128")))
    ap.add_argument("--measured-requests", type=int, default=int(os.environ.get("SERVICE_MEASURED_REQUESTS", "50")))
    args = ap.parse_args()
    root = group_root(args.group) / "comol_outputs"
    model_path = find_adapter(root)
    if model_path is None:
        raise RuntimeError("CoMoL checkpoint not found")
    comol_root = Path(os.environ.get("COMOL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload"))
    sys.path.insert(0, str(comol_root))
    from src import PeftModelForCausalLM

    load_started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=False, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    model = PeftModelForCausalLM.from_pretrained(model=base, model_id=str(model_path))
    model.to("cuda:0").eval()
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
        for _ in range(batches):
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
            print(f"[SERVICE] method=comol_native concurrency={args.concurrency} request={completed}/{args.measured_requests} tok_s={values[-1]:.3f}", flush=True)
        overall_elapsed = time.perf_counter() - overall_started
        average_power = sum(powers) / len(powers) if powers else None
        metrics = {
            "status": "PASS",
            "method": "comol_native",
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
        write_json(group_root(args.group) / "system" / "comol_native" / "metrics.json", metrics)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
