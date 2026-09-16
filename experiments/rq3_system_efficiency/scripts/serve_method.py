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
from coreflow.runtime import install_coreflow, install_isvd


class FirstTokenTimer:
    def __init__(self, wall_started, cuda_started):
        self.wall_started = wall_started
        self.cuda_started = cuda_started
        self.wall = None
        self.cuda = None
        self.peak_allocated = None
        self.peak_reserved = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.wall is None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(); event.synchronize()
            self.cuda = self.cuda_started.elapsed_time(event) / 1000.0
            self.wall = time.perf_counter() - self.wall_started
            self.peak_allocated = int(torch.cuda.max_memory_allocated())
            self.peak_reserved = int(torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
        return False


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--asset-root", required=True)
    ap.add_argument("--official-root", required=True)
    ap.add_argument("--adapter-order", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--method", choices=("full_legacy_original", "coreflow_q185_frozen", "isvd_legacy_padded_matched_q185"), required=True)
    ap.add_argument("--bank-dir", default="")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--output-length", type=int, required=True)
    ap.add_argument("--fixed-output", action="store_true")
    args = ap.parse_args()

    seed = 20260802
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    order = [x for x in args.adapter_order.split(",") if x]
    load_started = time.perf_counter()
    tokenizer, model, gate_report = load_loraflow_model(
        args.model, args.asset_root, args.official_root, task=None,
        adapter_order=order, gate_path=args.gate, dtype=torch.bfloat16, device="cuda",
    )
    model_loaded_seconds = time.perf_counter() - load_started
    install_started = time.perf_counter(); install_report = None
    if args.method == "coreflow_q185_frozen":
        install_report = install_coreflow(model, args.bank_dir, uniform=False, release_sources=True, expected_expert_order=order)
    elif args.method == "isvd_legacy_padded_matched_q185":
        install_report = install_isvd(model, args.bank_dir, release_sources=True, expected_expert_order=order)
    install_seconds = time.perf_counter() - install_started
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    startup_allocated = int(torch.cuda.memory_allocated())
    startup_reserved = int(torch.cuda.memory_reserved())

    from flask import Flask, jsonify, request
    from transformers import StoppingCriteriaList
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok", "method": args.method, "adapter_order": order,
            "gate": gate_report, "install": install_report,
            "cuda_device": torch.cuda.get_device_name(0),
            "startup_memory_allocated_bytes": startup_allocated,
            "startup_memory_reserved_bytes": startup_reserved,
            "model_load_seconds": model_loaded_seconds,
            "runtime_install_seconds": install_seconds,
            "health_ready_seconds": time.perf_counter() - load_started,
        })

    @app.post("/infer")
    def infer():
        received = time.perf_counter(); payload = request.get_json(force=True)
        task_seed = int(payload.get("seed", seed))
        random.seed(task_seed); np.random.seed(task_seed); torch.manual_seed(task_seed); torch.cuda.manual_seed_all(task_seed)
        if "prompt" in payload:
            enc = tokenizer(str(payload["prompt"]), max_length=args.max_input_tokens, truncation=True, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in enc.items()}
        else:
            ids = torch.tensor(payload["input_ids"], dtype=torch.long)
            mask = torch.tensor(payload["attention_mask"], dtype=torch.long)
            if ids.ndim != 2 or ids.shape != mask.shape or not bool(torch.all(mask == 1)):
                raise ValueError("System inputs must be rank-2, equal-shaped, and fully active")
            inputs = {"input_ids": ids.to("cuda"), "attention_mask": mask.to("cuda")}
        input_len = int(inputs["input_ids"].shape[-1]); request_to_engine = time.perf_counter() - received
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        wall_started = time.perf_counter(); cuda_start = torch.cuda.Event(enable_timing=True); cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record(); timer = FirstTokenTimer(wall_started, cuda_start)
        kwargs = dict(**inputs, do_sample=False, temperature=1.0, top_p=1.0,
                      max_new_tokens=args.output_length, use_cache=True,
                      stopping_criteria=StoppingCriteriaList([timer]))
        if args.fixed_output:
            kwargs["min_new_tokens"] = args.output_length
        with torch.inference_mode():
            outputs = model.generate(**kwargs)
        cuda_end.record(); cuda_end.synchronize()
        wall_total = time.perf_counter() - wall_started; cuda_total = cuda_start.elapsed_time(cuda_end) / 1000.0
        new_tokens = int(outputs.shape[-1] - input_len)
        decode_wall = None if timer.wall is None else max(0.0, wall_total - timer.wall)
        decode_cuda = None if timer.cuda is None else max(0.0, cuda_total - timer.cuda)
        generated = outputs[:, input_len:].detach().cpu().tolist()
        peak_allocated = max(int(timer.peak_allocated or 0), int(torch.cuda.max_memory_allocated()))
        peak_reserved = max(int(timer.peak_reserved or 0), int(torch.cuda.max_memory_reserved()))
        timing = {
            "input_tokens": input_len, "new_tokens": new_tokens,
            "engine_wall_total_seconds": wall_total, "engine_cuda_total_seconds": cuda_total,
            "prefill_wall_seconds": timer.wall, "prefill_cuda_seconds": timer.cuda,
            "decode_wall_seconds": decode_wall, "decode_cuda_seconds": decode_cuda,
            "request_to_engine_seconds": request_to_engine,
            "server_ttft_seconds": None if timer.wall is None else request_to_engine + timer.wall,
            "decode_tokens_per_second": None if not decode_wall or new_tokens < 2 else outputs.shape[0] * (new_tokens - 1) / decode_wall,
            "total_tokens_per_second": outputs.shape[0] * new_tokens / wall_total,
            "peak_memory_allocated_bytes": peak_allocated, "peak_memory_reserved_bytes": peak_reserved,
            "startup_resident_allocated_bytes": startup_allocated, "startup_resident_reserved_bytes": startup_reserved,
            "incremental_peak_allocated_bytes": max(0, peak_allocated - startup_allocated),
            "incremental_peak_reserved_bytes": max(0, peak_reserved - startup_reserved),
        }
        return jsonify({
            "method": args.method, "batch_size": int(outputs.shape[0]),
            "input_tokens_per_sequence": input_len, "new_tokens_per_sequence": new_tokens,
            "generated_token_ids_sha256": json_hash(generated),
            "answer": tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True) if "prompt" in payload else None,
            "timing": timing,
        })

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
