#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import AdapterModule, load_adapter, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    xs = sorted(float(v) for v in values)
    def q(prob: float) -> float:
        pos = (len(xs) - 1) * prob
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return xs[lo]
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac
    return {"count": len(xs), "mean": sum(xs) / len(xs), "median": q(0.5), "p95": q(0.95), "min": xs[0], "max": xs[-1]}


def module_output(module: AdapterModule, x: torch.Tensor) -> torch.Tensor:
    return ((x @ module.a.T) @ module.b.T) * module.scaling


def core_output(u: torch.Tensor, v: torch.Tensor, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((x @ v) @ c.T) @ u.T


def isvd_output(b: torch.Tensor, v: torch.Tensor, rank: int, x: torch.Tensor) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros(x.shape[0], b.shape[0], dtype=x.dtype, device=x.device)
    return (x @ v[:, :rank]) @ b[:, :rank].T


def normalized_entropy(weights: list[float]) -> float:
    k = len(weights)
    h = -sum(w * math.log(max(w, 1e-30)) for w in weights if w > 0)
    return h / math.log(k) if k > 1 else 0.0


def gate_draws(cfg: dict, k: int, rng: random.Random) -> list[dict]:
    draws = []
    for index in range(int(cfg["gate_families"]["one_hot"]["draws"])):
        w = [0.0] * k
        w[index % k] = 1.0
        draws.append({"family": "one_hot", "label": "one_hot", "weights": w})
    topk_cfg = cfg["gate_families"]["topk"]
    for kk in topk_cfg["ks"]:
        kk = min(int(kk), k)
        for _ in range(int(topk_cfg["draws_per_k"])):
            ids = rng.sample(range(k), kk)
            vals = [rng.random() for _ in ids]
            total = sum(vals)
            w = [0.0] * k
            for idx, val in zip(ids, vals):
                w[idx] = val / total
            draws.append({"family": "topk", "label": f"top{kk}", "weights": w})
    dir_cfg = cfg["gate_families"]["dirichlet"]
    for alpha in dir_cfg["alphas"]:
        alpha = float(alpha)
        for _ in range(int(dir_cfg["draws_per_alpha"])):
            vals = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
            total = sum(vals)
            draws.append({"family": "dirichlet", "label": f"dirichlet_{alpha:g}", "weights": [v / total for v in vals]})
    soft_cfg = cfg["gate_families"]["softmax_temperature"]
    for temp in soft_cfg["temperatures"]:
        temp = float(temp)
        for _ in range(int(soft_cfg["draws_per_temperature"])):
            logits = [rng.gauss(0.0, 1.0) / max(temp, 1e-8) for _ in range(k)]
            m = max(logits)
            vals = [math.exp(x - m) for x in logits]
            total = sum(vals)
            draws.append({"family": "softmax_temperature", "label": f"softmax_T{temp:g}", "weights": [v / total for v in vals]})
    for draw in draws:
        draw["entropy"] = normalized_entropy(draw["weights"])
    return draws


def entropy_bin(value: float, edges: list[float]) -> str:
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return f">={edges[-1]:.2f}"


def pairwise_cosines(vectors: list[torch.Tensor]) -> list[float]:
    out = []
    flat = [v.reshape(-1).float() for v in vectors]
    norms = [torch.linalg.vector_norm(v).clamp_min(1e-30) for v in flat]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            out.append(float(torch.dot(flat[i], flat[j]) / (norms[i] * norms[j])))
    return out


def load_expert_modules(pool_order: list[str], source_work_root: Path, m3_upload_root: Path) -> list[dict[str, AdapterModule]]:
    paths = {
        "zh": source_work_root / "official_assets" / "LoRAs" / "zh_lora",
        "ru": source_work_root / "official_assets" / "LoRAs" / "ru_lora",
        "es": source_work_root / "official_assets" / "LoRAs" / "es_lora",
        "math": source_work_root / "official_assets" / "LoRAs" / "math_lora",
        "code": source_work_root / "official_assets" / "LoRAs" / "code_lora",
        "magicoder": m3_upload_root / "lora_assets" / "m3_k8_strict" / "magicoder",
        "openwebmath": m3_upload_root / "lora_assets" / "m3_k8_strict" / "openwebmath",
        "gsm8k_loftq": m3_upload_root / "lora_assets" / "m3_k8_strict" / "gsm8k_loftq",
    }
    return [load_adapter(paths[name])[1] for name in pool_order]


def analyze_pool(pool: str, cfg: dict, args: argparse.Namespace, bank_lock: dict) -> dict:
    from safetensors.torch import load_file

    source_work = Path(args.source_work_root)
    m3_upload = Path(args.m3_upload_root)
    order = bank_lock["pools"][pool]["expert_order"]
    groups = load_expert_modules(order, source_work, m3_upload)
    names = sorted(groups[0])[: int(cfg["max_modules_per_pool"])]
    core_dir = Path(bank_lock["pools"][pool]["coreflow"]["directory"])
    isvd_dir = Path(bank_lock["pools"][pool]["isvd"]["directory"])
    core_cfg = load_json(core_dir / "core_config.json")
    isvd_cfg = load_json(isvd_dir / "isvd_config.json")
    core_tensors = load_file(str(core_dir / "core_bank.safetensors"), device="cpu")
    isvd_tensors = load_file(str(isvd_dir / "isvd_bank.safetensors"), device="cpu")
    name_to_index = {item["name"]: index for index, item in enumerate(core_cfg["modules"])}
    rng = random.Random(int(cfg["random_seed"]) + len(order) * 101)
    draws = gate_draws(cfg, len(order), rng)
    edges = [float(x) for x in cfg["entropy_bins"]]

    family_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"coreflow": [], "isvd": [], "entropy": []})
    bin_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"coreflow": [], "isvd": [], "entropy": []})
    residual_cos = {"coreflow": [], "isvd": []}
    cancellation = {"coreflow": [], "isvd": []}
    wins = {"coreflow": 0, "isvd": 0, "ties": 0}
    module_rows = []

    for ordinal, name in enumerate(names):
        module_index = name_to_index[name]
        prefix = f"m{module_index:03d}"
        u = core_tensors[f"{prefix}.U"].float()
        v = core_tensors[f"{prefix}.V"].float()
        c_all = core_tensors[f"{prefix}.C"].float()
        b_all = isvd_tensors[f"{prefix}.B"].float()
        v_all = isvd_tensors[f"{prefix}.V"].float()
        ranks = isvd_cfg["modules"][module_index]["expert_ranks"]
        generator = torch.Generator(device="cpu").manual_seed(99_000 + ordinal)
        x = torch.randn(int(cfg["activation_samples_per_module"]), groups[0][name].in_features, generator=generator, dtype=torch.float32)
        refs = [module_output(group[name], x) for group in groups]
        core_residuals = [core_output(u, v, c_all[i], x) - refs[i] for i in range(len(order))]
        isvd_residuals = [isvd_output(b_all[i], v_all[i], ranks[i], x) - refs[i] for i in range(len(order))]
        residual_cos["coreflow"].extend(pairwise_cosines(core_residuals))
        residual_cos["isvd"].extend(pairwise_cosines(isvd_residuals))
        for label, residuals in [("coreflow", core_residuals), ("isvd", isvd_residuals)]:
            mean_norm = sum(float(torch.linalg.vector_norm(r)) for r in residuals) / len(residuals)
            uniform_res = sum(residuals) / len(residuals)
            cancellation[label].append(float(torch.linalg.vector_norm(uniform_res)) / max(mean_norm, 1e-30))
        for draw in draws:
            weights = draw["weights"]
            ref = sum(refs[i] * weights[i] for i in range(len(order)))
            core = sum(core_residuals[i] * weights[i] for i in range(len(order)))
            isvd = sum(isvd_residuals[i] * weights[i] for i in range(len(order)))
            denom = torch.linalg.vector_norm(ref).clamp_min(1e-30)
            ce = float(torch.linalg.vector_norm(core) / denom)
            ie = float(torch.linalg.vector_norm(isvd) / denom)
            family = draw["label"]
            family_errors[family]["coreflow"].append(ce)
            family_errors[family]["isvd"].append(ie)
            family_errors[family]["entropy"].append(draw["entropy"])
            b = entropy_bin(float(draw["entropy"]), edges)
            bin_errors[b]["coreflow"].append(ce)
            bin_errors[b]["isvd"].append(ie)
            bin_errors[b]["entropy"].append(draw["entropy"])
            if ce < ie:
                wins["coreflow"] += 1
            elif ie < ce:
                wins["isvd"] += 1
            else:
                wins["ties"] += 1
        if ordinal % 25 == 0:
            print(f"[POOL {pool}] modules={ordinal+1}/{len(names)}", flush=True)

    def summarize_block(block: dict[str, list[float]]) -> dict:
        core = summarize(block["coreflow"])
        isvd = summarize(block["isvd"])
        return {
            "draw_count": len(block["coreflow"]),
            "entropy": summarize(block["entropy"]),
            "coreflow": core,
            "isvd": isvd,
            "coreflow_minus_isvd_mean": None if core["mean"] is None or isvd["mean"] is None else core["mean"] - isvd["mean"],
            "winner": (
                "coreflow" if core["mean"] is not None and isvd["mean"] is not None and core["mean"] < isvd["mean"]
                else "isvd" if core["mean"] is not None and isvd["mean"] is not None and isvd["mean"] < core["mean"]
                else "tie"
            ),
        }

    family_summary = {label: summarize_block(values) for label, values in sorted(family_errors.items())}
    bin_summary = {label: summarize_block(values) for label, values in sorted(bin_errors.items())}
    sparse_bins = [item for label, item in bin_summary.items() if item["entropy"]["mean"] is not None and item["entropy"]["mean"] < 0.5]
    dense_bins = [item for label, item in bin_summary.items() if item["entropy"]["mean"] is not None and item["entropy"]["mean"] >= 0.75]
    sparse_core_wins = sum(item["winner"] == "coreflow" for item in sparse_bins)
    dense_isvd_wins = sum(item["winner"] == "isvd" for item in dense_bins)
    return {
        "pool": pool,
        "expert_order": order,
        "module_count": len(names),
        "draws_per_module": len(draws),
        "wins": wins,
        "family_summary": family_summary,
        "entropy_bin_summary": bin_summary,
        "residual_alignment": {
            "pairwise_cosine": {
                "coreflow": summarize(residual_cos["coreflow"]),
                "isvd": summarize(residual_cos["isvd"]),
            },
            "uniform_cancellation_ratio": {
                "coreflow": summarize(cancellation["coreflow"]),
                "isvd": summarize(cancellation["isvd"]),
                "lower_is_more_cancellation": True,
            },
        },
        "regime_summary": {
            "sparse_bins_coreflow_wins": sparse_core_wins,
            "dense_bins_isvd_wins": dense_isvd_wins,
            "gate_dependence_supported": sparse_core_wins > 0 and dense_isvd_wins > 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-work-root", required=True)
    parser.add_argument("--m3-upload-root", required=True)
    parser.add_argument("--m3-work-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    m3_work = Path(args.m3_work_root)
    bank_lock_path = m3_work / "reports" / "m3_k_scaling_mini" / "bank_lock.json"
    metrics_path = m3_work / "reports" / "m3_k_scaling_mini" / "reconstruction_metrics.json"
    if not bank_lock_path.exists():
        raise FileNotFoundError(bank_lock_path)
    bank_lock = load_json(bank_lock_path)
    results = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "config_sha256": sha256_file(args.config),
        "m3_bank_lock_sha256": sha256_file(bank_lock_path),
        "m3_reconstruction_metrics_sha256": sha256_file(metrics_path) if metrics_path.exists() else None,
        "pools": {},
    }
    for pool in cfg["pools"]:
        print(f"[M3R] analyze {pool}", flush=True)
        results["pools"][pool] = analyze_pool(pool, cfg, args, bank_lock)

    primary = results["pools"][cfg["primary_pool"]]["regime_summary"]
    if primary["gate_dependence_supported"]:
        decision = "M3R_GATE_DEPENDENCE_SUPPORTED"
    else:
        decision = "M3R_NO_CLEAR_GATE_REGIME_SPLIT"
    results["decision"] = decision
    write_json(args.output, results)
    print(json.dumps({"status": "ANALYZED", "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()
