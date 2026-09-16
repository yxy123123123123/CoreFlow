#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.decompose import build_banks
from coreflow.gate_io import load_json, write_json
from coreflow.io import AdapterModule, adapter_checkpoint_path, load_adapter, module_signature, sha256_file
from coreflow.isvd import build_isvd_bank


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def expert_paths(cfg: dict, source_work_root: Path) -> dict[str, Path]:
    paths = {}
    for name, dirname in cfg["existing_experts"].items():
        paths[name] = source_work_root / "official_assets" / "LoRAs" / dirname
    for name, relative in cfg["new_experts"].items():
        paths[name] = ROOT / relative
    return paths


def load_all_adapters(cfg: dict, source_work_root: Path) -> tuple[dict[str, dict], dict[str, dict[str, AdapterModule]], dict[str, dict]]:
    paths = expert_paths(cfg, source_work_root)
    configs = {}
    modules = {}
    meta = {}
    for name, path in paths.items():
        config, loaded = load_adapter(path)
        checkpoint = adapter_checkpoint_path(path)
        configs[name] = config
        modules[name] = loaded
        meta[name] = {
            "name": name,
            "path": str(path),
            "adapter_config_sha256": sha256_file(path / "adapter_config.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "rank": int(config["r"]),
            "lora_alpha": config.get("lora_alpha"),
            "lora_dropout": config.get("lora_dropout"),
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "base_model_name_or_path": config.get("base_model_name_or_path"),
            "target_modules": sorted(config.get("target_modules") or []),
            "modules_to_save": config.get("modules_to_save"),
            "module_count": len(loaded),
        }
    return configs, modules, meta


def check_assets(args, cfg: dict) -> None:
    source = Path(args.source_work_root).resolve()
    manifest = load_json(ROOT / "lora_assets" / "m3_k8_strict" / "M3_K8_STRICT_LORA_MANIFEST.json")
    configs, modules, meta = load_all_adapters(cfg, source)
    errors = []
    for name, item in meta.items():
        base = str(item["base_model_name_or_path"] or "")
        if "Llama-2-7b" not in base and "llama-2-7b" not in base.lower():
            errors.append(f"{name}: suspicious base model {base}")
        if item["peft_type"] != "LORA":
            errors.append(f"{name}: peft_type is not LORA")
        if item["modules_to_save"]:
            errors.append(f"{name}: modules_to_save is non-empty")
    target_counts = Counter(tuple(item["target_modules"]) for item in meta.values())
    module_sets = {name: set(mods) for name, mods in modules.items()}
    reference = module_sets["zh"]
    diffs = {name: sorted(current.symmetric_difference(reference))[:20] for name, current in module_sets.items() if current != reference}
    if diffs:
        errors.append(f"adapter module-name sets differ: {diffs}")
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "source_work_root": str(source),
        "download_manifest_sha256": sha256_file(ROOT / "lora_assets" / "m3_k8_strict" / "M3_K8_STRICT_LORA_MANIFEST.json"),
        "download_manifest": manifest,
        "experts": meta,
        "module_signature_sha256_by_expert": {
            name: __import__("hashlib").sha256(json.dumps(module_signature(modules[name]), sort_keys=True).encode()).hexdigest()
            for name in modules
        },
        "target_module_pattern_counts": {";".join(key): value for key, value in target_counts.items()},
        "errors": errors,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "experts": len(meta), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def build(args, cfg: dict) -> None:
    work = Path(args.work_root).resolve()
    source = Path(args.source_work_root).resolve()
    _, modules, meta = load_all_adapters(cfg, source)
    q = int(cfg["q"])
    bank_root = work / "banks"
    lock = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "q": q,
        "pools": {},
        "experts": meta,
    }
    for pool_name, order in cfg["pools"].items():
        print(f"[BUILD] {pool_name} order={order}", flush=True)
        groups = [modules[name] for name in order]
        source_metadata = [meta[name] for name in order]
        core_out = build_banks(
            groups,
            order,
            [q],
            bank_root / "coreflow" / pool_name,
            beta=0.5,
            save_dtype=torch.bfloat16,
            balanced=True,
            source_metadata=source_metadata,
            bank_label_suffix="_bf16",
            schema_format="coreflow-m3-scaling-mini-v1",
        )[str(q)]
        budget = int(core_out["efficiency"]["coreflow_resident_adapter_parameters"])
        isvd_dir = bank_root / "isvd" / pool_name / "matched_q185_bf16"
        isvd_config = build_isvd_bank(groups, order, isvd_dir, budget, source_metadata, save_dtype=torch.bfloat16)
        lock["pools"][pool_name] = {
            "expert_order": order,
            "source_full_parameters": int(core_out["efficiency"]["source_resident_adapter_parameters"]),
            "coreflow": core_out,
            "isvd": {
                "directory": str(isvd_dir),
                "isvd_bank_sha256": sha256_file(isvd_dir / "isvd_bank.safetensors"),
                "isvd_config_sha256": sha256_file(isvd_dir / "isvd_config.json"),
                "budget": isvd_config["budget"],
            },
        }
    write_json(args.output, lock)
    print(json.dumps({"status": "PASS", "pools": list(lock["pools"])}, ensure_ascii=False))


def module_output(module: AdapterModule, x: torch.Tensor) -> torch.Tensor:
    return ((x @ module.a.T) @ module.b.T) * module.scaling


def core_output(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((x @ v) @ c.T) @ u.T


def isvd_output(b: torch.Tensor, v: torch.Tensor, rank: int, x: torch.Tensor) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(x.shape[0], b.shape[0], dtype=x.dtype, device=x.device)
    return (x @ v[:, :rank]) @ b[:, :rank].T


def gate_weights(kind: str, k: int, rng: random.Random) -> list[float]:
    if kind == "uniform":
        return [1.0 / k] * k
    if kind == "one_hot":
        out = [0.0] * k
        out[rng.randrange(k)] = 1.0
        return out
    if kind == "top2_sparse":
        first = rng.randrange(k)
        second = rng.randrange(k - 1)
        if second >= first:
            second += 1
        w = rng.random()
        out = [0.0] * k
        out[first] = w
        out[second] = 1.0 - w
        return out
    if kind.startswith("dirichlet_"):
        alpha = float(kind.split("_", 1)[1])
        vals = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
        total = sum(vals)
        return [v / total for v in vals]
    raise ValueError(kind)


def analyze(args, cfg: dict) -> None:
    from safetensors.torch import load_file

    work = Path(args.work_root).resolve()
    source = Path(args.source_work_root).resolve()
    _, modules, meta = load_all_adapters(cfg, source)
    bank_lock = load_json(args.bank_lock)
    q = int(cfg["q"])
    samples = int(cfg["metrics"]["activation_samples_per_module"])
    gate_draws = int(cfg["metrics"]["fused_gate_draws_per_distribution"])
    rng = random.Random(int(cfg["metrics"]["random_seed"]))
    report = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "pools": {},
    }
    for pool_name, order in cfg["pools"].items():
        print(f"[ANALYZE] {pool_name}", flush=True)
        pool_modules = [modules[name] for name in order]
        names = sorted(pool_modules[0])
        core_dir = Path(bank_lock["pools"][pool_name]["coreflow"]["directory"])
        isvd_dir = Path(bank_lock["pools"][pool_name]["isvd"]["directory"])
        core_cfg = load_json(core_dir / "core_config.json")
        isvd_cfg = load_json(isvd_dir / "isvd_config.json")
        core_tensors = load_file(str(core_dir / "core_bank.safetensors"), device="cpu")
        isvd_tensors = load_file(str(isvd_dir / "isvd_bank.safetensors"), device="cpu")
        core_errors = []
        isvd_errors = []
        fused = {dist: {"coreflow": [], "isvd": []} for dist in cfg["metrics"]["gate_distributions"]}
        worst = {"coreflow": None, "isvd": None}
        for module_index, name in enumerate(names):
            prefix = f"m{module_index:03d}"
            u = core_tensors[f"{prefix}.U"].float()
            v = core_tensors[f"{prefix}.V"].float()
            c_all = core_tensors[f"{prefix}.C"].float()
            b_all = isvd_tensors[f"{prefix}.B"].float()
            v_all = isvd_tensors[f"{prefix}.V"].float()
            ranks = isvd_cfg["modules"][module_index]["expert_ranks"]
            generator = torch.Generator(device="cpu").manual_seed(10_000 + module_index)
            x = torch.randn(samples, pool_modules[0][name].in_features, generator=generator, dtype=torch.float32)
            ref_outputs = [module_output(group[name], x) for group in pool_modules]
            for expert_index, ref in enumerate(ref_outputs):
                ce = float(torch.linalg.vector_norm(core_output(u, v, c_all[expert_index], x) - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30))
                ie = float(torch.linalg.vector_norm(isvd_output(b_all[expert_index], v_all[expert_index], ranks[expert_index], x) - ref) / torch.linalg.vector_norm(ref).clamp_min(1e-30))
                core_errors.append(ce)
                isvd_errors.append(ie)
                item_c = {"expert": order[expert_index], "module": name, "error": ce}
                item_i = {"expert": order[expert_index], "module": name, "error": ie}
                if worst["coreflow"] is None or ce > worst["coreflow"]["error"]:
                    worst["coreflow"] = item_c
                if worst["isvd"] is None or ie > worst["isvd"]["error"]:
                    worst["isvd"] = item_i
            for dist in cfg["metrics"]["gate_distributions"]:
                for _ in range(gate_draws):
                    weights = gate_weights(dist, len(order), rng)
                    ref = sum(ref_outputs[i] * weights[i] for i in range(len(order)))
                    core = sum(core_output(u, v, c_all[i], x) * weights[i] for i in range(len(order)))
                    isvd = sum(isvd_output(b_all[i], v_all[i], ranks[i], x) * weights[i] for i in range(len(order)))
                    denom = torch.linalg.vector_norm(ref).clamp_min(1e-30)
                    fused[dist]["coreflow"].append(float(torch.linalg.vector_norm(core - ref) / denom))
                    fused[dist]["isvd"].append(float(torch.linalg.vector_norm(isvd - ref) / denom))
        report["pools"][pool_name] = {
            "expert_order": order,
            "source_full_parameters": bank_lock["pools"][pool_name]["source_full_parameters"],
            "coreflow_efficiency": bank_lock["pools"][pool_name]["coreflow"]["efficiency"],
            "isvd_budget": bank_lock["pools"][pool_name]["isvd"]["budget"],
            "per_expert_module_output_error": {
                "coreflow": summarize(core_errors),
                "isvd": summarize(isvd_errors),
                "coreflow_minus_isvd_mean": summarize(core_errors)["mean"] - summarize(isvd_errors)["mean"],
                "coreflow_worst": worst["coreflow"],
                "isvd_worst": worst["isvd"],
            },
            "fused_output_error": {
                dist: {
                    "coreflow": summarize(values["coreflow"]),
                    "isvd": summarize(values["isvd"]),
                    "coreflow_minus_isvd_mean": summarize(values["coreflow"])["mean"] - summarize(values["isvd"])["mean"],
                }
                for dist, values in fused.items()
            },
        }
    write_json(args.output, report)
    print(json.dumps({"status": "ANALYZED", "pools": list(report["pools"])}, ensure_ascii=False))


def benchmark(args, cfg: dict) -> None:
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark")
    device = torch.device("cuda")
    work = Path(args.work_root).resolve()
    source = Path(args.source_work_root).resolve()
    _, modules, _ = load_all_adapters(cfg, source)
    bank_lock = load_json(args.bank_lock)
    warmup = int(cfg["benchmark"]["warmup"])
    iterations = int(cfg["benchmark"]["iterations"])
    selected_modules = set(cfg["benchmark"]["modules"])
    report = {"status": "PASS", "protocol_id": cfg["protocol_id"], "timestamp_utc": utc_now(), "pools": {}}
    for pool_name, order in cfg["pools"].items():
        print(f"[BENCH] {pool_name}", flush=True)
        pool_modules = [{k: v for k, v in modules[name].items() if k in selected_modules} for name in order]
        names = sorted(set.intersection(*(set(group) for group in pool_modules)))
        core_dir = Path(bank_lock["pools"][pool_name]["coreflow"]["directory"])
        isvd_dir = Path(bank_lock["pools"][pool_name]["isvd"]["directory"])
        core_cfg = load_json(core_dir / "core_config.json")
        isvd_cfg = load_json(isvd_dir / "isvd_config.json")
        all_names = [m["name"] for m in core_cfg["modules"]]
        name_to_index = {name: idx for idx, name in enumerate(all_names)}
        core_tensors = load_file(str(core_dir / "core_bank.safetensors"), device="cpu")
        isvd_tensors = load_file(str(isvd_dir / "isvd_bank.safetensors"), device="cpu")
        rows = []
        for name in names:
            module_index = name_to_index[name]
            prefix = f"m{module_index:03d}"
            source_gpu = [
                AdapterModule(name, group[name].a.to(device), group[name].b.to(device), group[name].scaling)
                for group in pool_modules
            ]
            u = core_tensors[f"{prefix}.U"].float().to(device)
            v = core_tensors[f"{prefix}.V"].float().to(device)
            c_all = core_tensors[f"{prefix}.C"].float().to(device)
            b_all = isvd_tensors[f"{prefix}.B"].float().to(device)
            v_all = isvd_tensors[f"{prefix}.V"].float().to(device)
            ranks = isvd_cfg["modules"][module_index]["expert_ranks"]
            x = torch.randn(1, source_gpu[0].in_features, device=device)
            weights = torch.full((len(order),), 1.0 / len(order), device=device)

            def run_full():
                out = None
                for i, mod in enumerate(source_gpu):
                    val = module_output(mod, x) * weights[i]
                    out = val if out is None else out + val
                return out

            def run_core():
                out = None
                xv = x @ v
                for i in range(len(order)):
                    val = ((xv @ c_all[i].T) @ u.T) * weights[i]
                    out = val if out is None else out + val
                return out

            def run_isvd():
                out = None
                for i in range(len(order)):
                    val = isvd_output(b_all[i], v_all[i], ranks[i], x) * weights[i]
                    out = val if out is None else out + val
                return out

            timings = {}
            for label, fn in [("full", run_full), ("coreflow", run_core), ("isvd", run_isvd)]:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    fn()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                timings[label] = {
                    "seconds_total": elapsed,
                    "microseconds_per_iter": 1e6 * elapsed / iterations,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            rows.append({"module": name, "timings": timings})
        def mean_us(label: str) -> float:
            return sum(row["timings"][label]["microseconds_per_iter"] for row in rows) / max(1, len(rows))
        report["pools"][pool_name] = {
            "modules": rows,
            "mean_microseconds": {label: mean_us(label) for label in ("full", "coreflow", "isvd")},
            "coreflow_speedup_over_full": mean_us("full") / mean_us("coreflow") if mean_us("coreflow") else None,
            "coreflow_speedup_over_isvd": mean_us("isvd") / mean_us("coreflow") if mean_us("coreflow") else None,
        }
    write_json(args.output, report)
    print(json.dumps({"status": "PASS", "pools": list(report["pools"])}, ensure_ascii=False))


def decide(args, cfg: dict) -> None:
    metrics = load_json(args.metrics)
    bench = load_json(args.benchmark)
    k8 = metrics["pools"]["k8_strict"]
    fused_core = k8["fused_output_error"]["uniform"]["coreflow"]["mean"]
    fused_isvd = k8["fused_output_error"]["uniform"]["isvd"]["mean"]
    worst_core = k8["per_expert_module_output_error"]["coreflow_worst"]["error"]
    worst_isvd = k8["per_expert_module_output_error"]["isvd_worst"]["error"]
    speed = bench["pools"]["k8_strict"].get("coreflow_speedup_over_isvd")
    checks = {
        "k8_uniform_fused_core_le_isvd": fused_core <= fused_isvd,
        "k8_worst_core_not_over_isvd_by_10pct": worst_core <= worst_isvd * 1.10,
        "k8_core_speedup_over_isvd_ge_1.05": speed is not None and speed >= 1.05,
    }
    decision = "M3_MINI_GO_SCALING_SUPPORTED" if all(checks.values()) else "M3_MINI_MIXED_OR_NO_GO"
    payload = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "decision": decision,
        "checks": checks,
        "k8_key_values": {
            "uniform_fused_error_coreflow": fused_core,
            "uniform_fused_error_isvd": fused_isvd,
            "worst_expert_module_error_coreflow": worst_core,
            "worst_expert_module_error_isvd": worst_isvd,
            "coreflow_speedup_over_isvd": speed,
        },
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "ANALYZED", "decision": decision, "checks": checks}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "build", "analyze", "benchmark", "decide"))
    parser.add_argument("--config", default=str(ROOT / "config" / "m3_k_scaling_mini_protocol.json"))
    parser.add_argument("--source-work-root", default="/root/autodl-tmp/coreflow_m0")
    parser.add_argument("--work-root", default="/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank-lock")
    parser.add_argument("--metrics")
    parser.add_argument("--benchmark")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action == "preflight":
        check_assets(args, cfg)
    elif args.action == "build":
        build(args, cfg)
    elif args.action == "analyze":
        if not args.bank_lock:
            raise ValueError("--bank-lock required")
        analyze(args, cfg)
    elif args.action == "benchmark":
        if not args.bank_lock:
            raise ValueError("--bank-lock required")
        benchmark(args, cfg)
    elif args.action == "decide":
        if not args.metrics or not args.benchmark:
            raise ValueError("--metrics and --benchmark required")
        decide(args, cfg)


if __name__ == "__main__":
    main()
