#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.data_audit import load_jsonl
from coreflow.gate_io import load_json, write_json
from coreflow.io import sha256_file
from coreflow.loraflow import load_loraflow_model


class Summary:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None

    def add_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        work = tensor.detach().float().cpu()
        self.count += int(work.numel())
        self.sum += float(work.sum().item())
        vmin = float(work.min().item())
        vmax = float(work.max().item())
        self.min = vmin if self.min is None else min(self.min, vmin)
        self.max = vmax if self.max is None else max(self.max, vmax)

    def add_value(self, value: float, n: int = 1) -> None:
        self.count += int(n)
        self.sum += float(value) * int(n)
        self.min = float(value) if self.min is None else min(self.min, float(value))
        self.max = float(value) if self.max is None else max(self.max, float(value))

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": None if self.count == 0 else self.sum / self.count,
            "min": self.min,
            "max": self.max,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_entropy(probs: torch.Tensor) -> torch.Tensor:
    k = probs.shape[-1]
    entropy = -(probs.clamp_min(1e-30) * probs.clamp_min(1e-30).log()).sum(dim=-1)
    return entropy / math.log(k)


def bin_label(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f">={edges[-1]:.2f}"


def prompt_for(row: dict, template: str) -> str:
    return template.format(problem=row["problem"], entry_point=row.get("entry_point", ""))


def make_accumulators(layers: int, bins: list[float]) -> dict:
    return {
        "global": {
            "entropy": Summary(),
            "top1": Summary(),
            "top2": Summary(),
            "effective_sum": Summary(),
            "effective_min": Summary(),
            "effective_negative_fraction": Summary(),
        },
        "per_layer": {
            str(layer): {
                "entropy": Summary(),
                "top1": Summary(),
                "top2": Summary(),
                "effective_sum": Summary(),
                "effective_min": Summary(),
                "effective_negative_fraction": Summary(),
                "entropy_bins": {bin_label((bins[i] + bins[i + 1]) / 2, bins): 0 for i in range(len(bins) - 1)},
            }
            for layer in range(layers)
        },
        "entropy_bins": {bin_label((bins[i] + bins[i + 1]) / 2, bins): 0 for i in range(len(bins) - 1)},
    }


def serialise_acc(acc: dict) -> dict:
    return {
        "global": {key: value.as_dict() for key, value in acc["global"].items()},
        "entropy_bins": dict(acc["entropy_bins"]),
        "per_layer": {
            layer: {
                key: value.as_dict() if isinstance(value, Summary) else dict(value)
                for key, value in item.items()
            }
            for layer, item in acc["per_layer"].items()
        },
    }


def audit_seed(args, cfg: dict, seed: int, rows: list[dict], template: str) -> dict:
    formal_work = Path(args.formal_work_root)
    source_work = Path(args.source_work_root)
    gate_name = f"code_k5_code_seed{seed}.pt"
    gate_candidates = [
        Path(args.m2_asset_root) / "gates" / gate_name if args.m2_asset_root else None,
        formal_work / "m2_assets" / "gates" / gate_name,
    ]
    gate = None
    for candidate in gate_candidates:
        if candidate is None:
            continue
        if candidate.exists():
            gate = candidate.resolve()
            break
    if gate is None:
        searched = [str(x) for x in gate_candidates if x is not None and x.name != "asset_lock.json"]
        raise FileNotFoundError(f"Missing frozen K5 code gate for seed {seed}; searched={searched}")
    tokenizer, model, load_report = load_loraflow_model(
        args.model,
        source_work / "official_assets",
        source_work / "vendor" / "LoRAFlow",
        adapter_order=cfg["expert_order"],
        gate_path=gate,
        dtype=torch.bfloat16,
        device="cuda",
    )
    layers = list(model.base_model.model.model.layers if hasattr(model.base_model.model, "model") else model.base_model.model.layers)
    bins = [float(x) for x in cfg["entropy_bins"]]
    acc = make_accumulators(len(layers), bins)
    hook_handles = []

    def make_hook(layer_index: int, layer):
        def hook(_module, _inputs, output):
            scores = output.detach().float()
            probs = torch.softmax(scores / float(getattr(layer, "temperature", 1.0)), dim=-1)
            bias = layer.weight_bias.detach().float().to(probs.device).view(1, 1, -1)
            effective = probs + bias
            entropy = normalized_entropy(probs)
            top = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
            top1 = top[..., 0]
            top2 = top.sum(dim=-1)
            eff_sum = effective.sum(dim=-1)
            eff_min = effective.min(dim=-1).values
            neg_fraction = (effective < 0).float().mean(dim=-1)
            for key, tensor in [
                ("entropy", entropy),
                ("top1", top1),
                ("top2", top2),
                ("effective_sum", eff_sum),
                ("effective_min", eff_min),
                ("effective_negative_fraction", neg_fraction),
            ]:
                acc["global"][key].add_tensor(tensor)
                acc["per_layer"][str(layer_index)][key].add_tensor(tensor)
            flat_entropy = entropy.reshape(-1).detach().cpu().tolist()
            for value in flat_entropy:
                label = bin_label(float(value), bins)
                acc["entropy_bins"][label] += 1
                acc["per_layer"][str(layer_index)]["entropy_bins"][label] += 1
        return hook

    for idx, layer in enumerate(layers):
        hook_handles.append(layer.lora_fusion_gate.register_forward_hook(make_hook(idx, layer)))

    started = time.time()
    processed = 0
    prompt_hashes = []
    try:
        with torch.inference_mode():
            for index, row in enumerate(rows):
                prompt = prompt_for(row, template)
                prompt_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
                inputs = tokenizer(
                    prompt,
                    max_length=int(cfg["max_input_tokens"]),
                    truncation=True,
                    return_tensors="pt",
                ).to("cuda")
                _ = model(**inputs, use_cache=False)
                processed += 1
                if processed == 1 or processed % 25 == 0 or processed == len(rows):
                    print(f"[SEED {seed}] prompts={processed}/{len(rows)} elapsed_min={(time.time()-started)/60:.1f}", flush=True)
    finally:
        for handle in hook_handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()

    summary = serialise_acc(acc)
    total = max(1, sum(summary["entropy_bins"].values()))
    fractions = {key: value / total for key, value in summary["entropy_bins"].items()}
    lt_010 = sum(value for key, value in fractions.items() if key.startswith("[0.00,0.10"))
    lt_025 = sum(value for key, value in fractions.items() if key.startswith("[0.00,0.10") or key.startswith("[0.10,0.25"))
    ge_050 = sum(value for key, value in fractions.items() if key.startswith("[0.50") or key.startswith("[0.75") or key.startswith("[0.90"))
    return {
        "seed": seed,
        "status": "ANALYZED",
        "load_report": load_report,
        "gate_path": str(gate),
        "gate_sha256": sha256_file(gate),
        "rows": processed,
        "elapsed_seconds": time.time() - started,
        "prompt_hash_sha256": hashlib.sha256("".join(prompt_hashes).encode()).hexdigest(),
        "summary": summary,
        "entropy_bin_fractions": fractions,
        "sparsity_fractions": {
            "entropy_lt_0.10": lt_010,
            "entropy_lt_0.25": lt_025,
            "entropy_ge_0.50": ge_050,
        },
    }


def decide(cfg: dict, seeds: list[dict]) -> dict:
    total_counts = defaultdict(int)
    for item in seeds:
        for label, count in item["summary"]["entropy_bins"].items():
            total_counts[label] += int(count)
    total = max(1, sum(total_counts.values()))
    fractions = {key: value / total for key, value in sorted(total_counts.items())}
    lt_010 = sum(value for key, value in fractions.items() if key.startswith("[0.00,0.10"))
    lt_025 = sum(value for key, value in fractions.items() if key.startswith("[0.00,0.10") or key.startswith("[0.10,0.25"))
    ge_050 = sum(value for key, value in fractions.items() if key.startswith("[0.50") or key.startswith("[0.75") or key.startswith("[0.90"))
    th = cfg["decision_thresholds"]
    if lt_010 >= float(th["strong_sparse_entropy_lt_0_10_fraction"]):
        decision = "REAL_GATE_STRONGLY_SPARSE_K8_FOLLOWUP_PLAUSIBLE"
    elif lt_025 >= float(th["qualified_sparse_entropy_lt_0_25_fraction"]):
        decision = "REAL_GATE_QUALIFIED_SPARSE_K8_FOLLOWUP_POSSIBLE"
    elif ge_050 >= float(th["dense_entropy_ge_0_50_fraction"]):
        decision = "REAL_GATE_DENSE_PAUSE_K8_ENDPOINT"
    else:
        decision = "REAL_GATE_MIXED_NEEDS_LAYER_ANALYSIS"
    return {
        "decision": decision,
        "pooled_entropy_bin_fractions": fractions,
        "pooled_sparsity_fractions": {
            "entropy_lt_0.10": lt_010,
            "entropy_lt_0.25": lt_025,
            "entropy_ge_0.50": ge_050,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--formal-upload-root", required=True)
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--m2-asset-root", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    formal_upload = Path(args.formal_upload_root)
    formal_work = Path(args.formal_work_root)
    data_path = formal_work / "data" / "formal_v2_primary" / cfg["dataset"]
    if not data_path.exists():
        data_path = formal_upload / "data" / cfg["dataset"]
    rows = load_jsonl(data_path)
    if cfg.get("limit_rows") is not None:
        rows = rows[: int(cfg["limit_rows"])]
    template = (formal_upload / "data" / cfg["prompt_file"]).read_text(encoding="utf-8")
    results = []
    for seed in cfg["gate_seeds"]:
        results.append(audit_seed(args, cfg, int(seed), rows, template))
    decision = decide(cfg, results)
    payload = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "config_sha256": sha256_file(args.config),
        "dataset": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "prompt_file": str(formal_upload / "data" / cfg["prompt_file"]),
        "prompt_file_sha256": sha256_file(formal_upload / "data" / cfg["prompt_file"]),
        "expert_order": cfg["expert_order"],
        "seeds": results,
        **decision,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "ANALYZED", "decision": payload["decision"], "pooled": payload["pooled_sparsity_fractions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
