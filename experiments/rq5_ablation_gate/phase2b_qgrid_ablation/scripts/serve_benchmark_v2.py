#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.loraflow import load_loraflow_model
from coreflow.runtime import (
    install_coreflow,
    install_full_vectorized,
    install_isvd,
    install_isvd_compact,
)


class FirstTokenTimer:
    def __init__(self, wall_started: float, cuda_started: torch.cuda.Event):
        self.wall_started = wall_started
        self.cuda_started = cuda_started
        self.first_wall_seconds = None
        self.first_cuda_seconds = None
        self.prefill_peak_allocated_bytes = None
        self.prefill_peak_reserved_bytes = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_wall_seconds is None:
            first = torch.cuda.Event(enable_timing=True)
            first.record()
            first.synchronize()
            self.first_cuda_seconds = self.cuda_started.elapsed_time(first) / 1000.0
            self.first_wall_seconds = time.perf_counter() - self.wall_started
            self.prefill_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            self.prefill_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        return False


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--adapter-order", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument(
        "--method-kind",
        choices=("full_legacy", "full_vectorized", "core", "isvd_legacy", "isvd_compact"),
        required=True,
    )
    parser.add_argument("--bank-dir")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args()

    order = [part for part in args.adapter_order.split(",") if part]
    if len(order) < 2:
        raise ValueError("Benchmark requires a multi-expert pool")
    random.seed(20260730)
    np.random.seed(20260730)
    torch.manual_seed(20260730)
    torch.cuda.manual_seed_all(20260730)

    load_started = time.perf_counter()
    tokenizer, model, gate_report = load_loraflow_model(
        args.model,
        args.asset_root,
        args.official_root,
        task=None,
        adapter_order=order,
        gate_path=args.gate,
        dtype=torch.bfloat16,
        device="cuda",
    )
    model_loaded_seconds = time.perf_counter() - load_started

    install_started = time.perf_counter()
    install_report = None
    if args.method_kind == "full_vectorized":
        install_report = install_full_vectorized(model, expected_expert_order=order, release_sources=True)
    elif args.method_kind == "core":
        if not args.bank_dir:
            raise ValueError("--bank-dir is required for CoreFlow")
        install_report = install_coreflow(
            model,
            args.bank_dir,
            uniform=False,
            release_sources=True,
            expected_expert_order=order,
        )
    elif args.method_kind == "isvd_legacy":
        if not args.bank_dir:
            raise ValueError("--bank-dir is required for Independent-SVD")
        install_report = install_isvd(
            model,
            args.bank_dir,
            release_sources=True,
            expected_expert_order=order,
        )
    elif args.method_kind == "isvd_compact":
        if not args.bank_dir:
            raise ValueError("--bank-dir is required for Independent-SVD")
        install_report = install_isvd_compact(
            model,
            args.bank_dir,
            release_sources=True,
            expected_expert_order=order,
        )
    install_seconds = time.perf_counter() - install_started

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    startup_allocated = int(torch.cuda.memory_allocated())
    startup_reserved = int(torch.cuda.memory_reserved())

    from flask import Flask, jsonify, request
    from transformers import StoppingCriteriaList

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "method_kind": args.method_kind,
                "adapter_order": order,
                "gate": gate_report,
                "install": install_report,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                "startup_memory_allocated_bytes": startup_allocated,
                "startup_memory_reserved_bytes": startup_reserved,
                "model_load_seconds": model_loaded_seconds,
                "runtime_install_seconds": install_seconds,
            }
        )

    @app.route("/infer", methods=["POST"])
    def infer():
        request_received = time.perf_counter()
        payload = request.get_json(force=True)
        input_ids_cpu = torch.tensor(payload["input_ids"], dtype=torch.long)
        attention_cpu = torch.tensor(payload["attention_mask"], dtype=torch.long)
        if input_ids_cpu.ndim != 2 or attention_cpu.shape != input_ids_cpu.shape:
            raise ValueError("input_ids and attention_mask must be equal-shape rank-2 arrays")
        task_seed = int(payload.get("seed", 20260730))
        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        torch.cuda.manual_seed_all(task_seed)
        inputs = {
            "input_ids": input_ids_cpu.to("cuda"),
            "attention_mask": attention_cpu.to("cuda"),
        }
        input_tokens = int(inputs["input_ids"].shape[-1])
        request_to_engine_seconds = time.perf_counter() - request_received

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        wall_started = time.perf_counter()
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_ended = torch.cuda.Event(enable_timing=True)
        cuda_started.record()
        timer = FirstTokenTimer(wall_started, cuda_started)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                min_new_tokens=args.output_length,
                max_new_tokens=args.output_length,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([timer]),
            )
        cuda_ended.record()
        cuda_ended.synchronize()
        wall_total = time.perf_counter() - wall_started
        cuda_total = cuda_started.elapsed_time(cuda_ended) / 1000.0
        new_tokens = int(outputs.shape[-1] - input_tokens)
        prefill_wall = timer.first_wall_seconds
        prefill_cuda = timer.first_cuda_seconds
        decode_wall = None if prefill_wall is None else max(0.0, wall_total - prefill_wall)
        decode_cuda = None if prefill_cuda is None else max(0.0, cuda_total - prefill_cuda)
        decode_peak_allocated = int(torch.cuda.max_memory_allocated())
        decode_peak_reserved = int(torch.cuda.max_memory_reserved())
        prefill_peak_allocated = int(timer.prefill_peak_allocated_bytes or 0)
        prefill_peak_reserved = int(timer.prefill_peak_reserved_bytes or 0)
        generated = outputs[:, input_tokens:].detach().cpu().tolist()
        return jsonify(
            {
                "method_kind": args.method_kind,
                "batch_size": int(outputs.shape[0]),
                "input_tokens_per_sequence": input_tokens,
                "new_tokens_per_sequence": new_tokens,
                "generated_token_ids_sha256": sha256_json(generated),
                "generated_token_ids": generated if payload.get("return_tokens", False) else None,
                "timing": {
                    "engine_wall_total_seconds": wall_total,
                    "engine_cuda_total_seconds": cuda_total,
                    "prefill_wall_seconds": prefill_wall,
                    "prefill_cuda_seconds": prefill_cuda,
                    "server_ttft_seconds": (
                        request_to_engine_seconds + prefill_wall
                        if prefill_wall is not None
                        else None
                    ),
                    "request_to_engine_seconds": request_to_engine_seconds,
                    "decode_wall_seconds": decode_wall,
                    "decode_cuda_seconds": decode_cuda,
                    "decode_tokens_per_second": (
                        outputs.shape[0] * (new_tokens - 1) / decode_wall
                        if new_tokens > 1 and decode_wall and decode_wall > 0
                        else None
                    ),
                    "engine_total_tokens_per_second": (
                        outputs.shape[0] * new_tokens / wall_total if wall_total > 0 else None
                    ),
                    "prefill_peak_memory_allocated_bytes": prefill_peak_allocated,
                    "prefill_peak_memory_reserved_bytes": prefill_peak_reserved,
                    "decode_peak_memory_allocated_bytes": decode_peak_allocated,
                    "decode_peak_memory_reserved_bytes": decode_peak_reserved,
                    "peak_memory_allocated_bytes": max(
                        prefill_peak_allocated,
                        decode_peak_allocated,
                    ),
                    "peak_memory_reserved_bytes": max(
                        prefill_peak_reserved,
                        decode_peak_reserved,
                    ),
                },
            }
        )

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
