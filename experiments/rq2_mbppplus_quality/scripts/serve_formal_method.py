#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


class TimingCriteria:
    def __init__(self, started: float):
        self.started = started
        self.first_token_seconds = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_seconds is None:
            torch.cuda.synchronize()
            self.first_token_seconds = time.perf_counter() - self.started
        return False


def fixed_inputs(tokenizer, length: int) -> dict[str, torch.Tensor]:
    if length < 2:
        raise ValueError("fixed_input_tokens must be >= 2")
    seed_ids = tokenizer(
        "CoreFlow fixed length benchmark input. ",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    if not len(seed_ids):
        raise ValueError("Tokenizer produced no benchmark tokens")
    repeats = (length - 1 + len(seed_ids) - 1) // len(seed_ids)
    body = seed_ids.repeat(repeats)[: length - 1]
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(seed_ids[0])
    ids = torch.cat([torch.tensor([bos_id], dtype=torch.long), body]).unsqueeze(0)
    return {
        "input_ids": ids.to("cuda"),
        "attention_mask": torch.ones_like(ids).to("cuda"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--adapter-order", required=True)
    parser.add_argument("--gate")
    parser.add_argument("--method-kind", choices=("full", "core", "static", "task_only", "isvd"), required=True)
    parser.add_argument("--bank-dir")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--do-sample", choices=("true", "false"), default="false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    order = [part for part in args.adapter_order.split(",") if part]
    if args.method_kind == "task_only" and len(order) != 1:
        raise ValueError("task_only requires exactly one adapter")
    if args.method_kind != "task_only" and len(order) < 2:
        raise ValueError("Full/Core/Static methods require at least two adapters")
    if args.method_kind != "task_only" and not args.gate:
        raise ValueError("A gate is required for multi-expert methods")

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
    install_report = None
    if args.method_kind in {"core", "static"}:
        if not args.bank_dir:
            raise ValueError("--bank-dir is required for CoreFlow methods")
        install_report = install_coreflow(
            model,
            args.bank_dir,
            uniform=args.method_kind == "static",
            release_sources=True,
            expected_expert_order=order,
        )
    elif args.method_kind == "isvd":
        if not args.bank_dir:
            raise ValueError("--bank-dir is required for Independent-SVD methods")
        install_report = install_isvd(
            model,
            args.bank_dir,
            release_sources=True,
            expected_expert_order=order,
        )
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    from flask import Flask, jsonify, request
    from transformers import StoppingCriteriaList

    app = Flask(__name__)
    startup_allocated = int(torch.cuda.memory_allocated())
    startup_reserved = int(torch.cuda.memory_reserved())

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "adapter_order": order,
            "method_kind": args.method_kind,
            "gate": gate_report,
            "install": install_report,
            "cuda_device": torch.cuda.get_device_name(0),
            "startup_memory_allocated_bytes": startup_allocated,
            "startup_memory_reserved_bytes": startup_reserved,
        })

    @app.route("/infer", methods=["POST"])
    def infer():
        payload = request.get_json(force=True)
        task_seed = int(payload["seed"])
        fixed_length = payload.get("fixed_input_tokens")
        if fixed_length is None:
            prompt = str(payload["prompt"])
            inputs = tokenizer(
                prompt,
                max_length=args.max_input_tokens,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
        else:
            inputs = fixed_inputs(tokenizer, int(fixed_length))
        input_tokens = int(inputs["input_ids"].shape[-1])

        random.seed(task_seed)
        np.random.seed(task_seed)
        torch.manual_seed(task_seed)
        torch.cuda.manual_seed_all(task_seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        timing = TimingCriteria(started)
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample == "true",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "stopping_criteria": StoppingCriteriaList([timing]),
            "use_cache": True,
        }
        if args.min_new_tokens:
            generation_kwargs["min_new_tokens"] = args.min_new_tokens
        with torch.inference_mode():
            outputs = model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = int(outputs.shape[-1] - input_tokens)
        ttft = timing.first_token_seconds
        decode_seconds = None if ttft is None else max(0.0, elapsed - ttft)
        answer = tokenizer.decode(outputs[0, input_tokens:], skip_special_tokens=True)
        return jsonify({
            "answer": answer,
            "timing": {
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "total_latency_seconds": elapsed,
                "ttft_seconds": ttft,
                "decode_latency_seconds": decode_seconds,
                "tokens_per_second_total": new_tokens / elapsed if elapsed > 0 else None,
                "decode_tokens_per_second": (
                    (new_tokens - 1) / decode_seconds
                    if new_tokens > 1 and decode_seconds and decode_seconds > 0
                    else None
                ),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
        })

    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
