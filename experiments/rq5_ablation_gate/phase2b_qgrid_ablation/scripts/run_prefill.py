#!/usr/bin/env python3
"""Phase-2B B3a: isolated prefill measurement."""
from __future__ import annotations
import json, sys, argparse, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import load_json, write_json
from coreflow.loraflow import load_loraflow_model
from coreflow.runtime import install_coreflow

def gen_inputs(tokenizer, length: int, batch: int, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    # use random token IDs in valid Llama-2 range
    ids = torch.randint(1, 32000, (batch, length), dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

def measure_prefill(model, inputs: dict, warmup: int, trials: int):
    device = next(model.parameters()).device
    inp = {k: v.to(device) for k, v in inputs.items()}
    for _ in range(warmup):
        with torch.inference_mode(): model(**inp, use_cache=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode(): model(**inp, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        mem = int(torch.cuda.max_memory_allocated())
    return {"latency_ms": [t * 1000 for t in times], "latency_median_ms": sorted(times)[len(times)//2] * 1000,
            "peak_allocated_bytes": mem, "input_tokens": inputs["input_ids"].numel()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-assets", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--method", choices=("full_vectorized","core_q185"), required=True)
    parser.add_argument("--bank", default=None)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-length", type=int, required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_json(ROOT / "config" / "protocol.json")
    order = cfg["expert_order"]
    tokenizer, model, _ = load_loraflow_model(args.model, args.source_assets, args.vendor_root,
        task=None, adapter_order=order, gate_path=args.gate, dtype=torch.bfloat16, device="cuda")
    if args.method == "core_q185":
        install_coreflow(model, args.bank, uniform=False, release_sources=True, expected_expert_order=order)
    model.eval(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    b3 = cfg["b3_prefill_ttft"]
    config_spec = next(c for c in b3["configs"] if c["batch"] == args.batch and c["input_length"] == args.input_length)
    inputs = gen_inputs(tokenizer, config_spec["input_length"], config_spec["batch"])
    res = measure_prefill(model, inputs, config_spec["warmup"], config_spec["trials"])
    res["method"] = args.method; res["batch"] = args.batch; res["input_length"] = args.input_length
    res["status"] = "PASS"
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, res)
    print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()