#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import config, load_json, now_utc, paths, sha256_file, variant_dir, write_json


class Adapter:
    def __init__(self, a: torch.Tensor, b: torch.Tensor, scaling: float):
        self.a, self.b, self.scaling = a.float(), b.float(), float(scaling)


def checkpoint(root: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(root)


def canonical(raw: str) -> str:
    name = raw
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    marker = "model.layers."
    return name[name.index(marker):] if marker in name else name


def load_adapter(root: Path) -> dict[str, Adapter]:
    adapter_cfg = load_json(root / "adapter_config.json")
    path = checkpoint(root)
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    pairs = {}
    for raw, tensor in state.items():
        suffix = next((candidate for candidate in (".lora_A.weight", ".lora_B.weight") if raw.endswith(candidate)), None)
        if suffix is None:
            continue
        name = canonical(raw[:-len(suffix)])
        pairs.setdefault(name, {})["a" if "lora_A" in suffix else "b"] = tensor.detach().cpu()
    result = {}
    rank_pattern, alpha_pattern = adapter_cfg.get("rank_pattern") or {}, adapter_cfg.get("alpha_pattern") or {}
    for name, tensors in sorted(pairs.items()):
        rank = int(rank_pattern.get(name, rank_pattern.get(name.split(".")[-1], adapter_cfg["r"])))
        alpha = float(alpha_pattern.get(name, alpha_pattern.get(name.split(".")[-1], adapter_cfg["lora_alpha"])))
        result[name] = Adapter(tensors["a"], tensors["b"], alpha / rank)
    if not result:
        raise ValueError(f"No LoRA pairs in {path}")
    return result


def stats(values: list[float], *, high_is_worse: bool = True) -> dict:
    x = torch.tensor(values, dtype=torch.float64)
    return {"count": int(x.numel()), "mean": float(x.mean()), "median": float(x.median()),
            "std_population": float(x.std(unbiased=False)), "p10": float(torch.quantile(x, 0.10)),
            "p25": float(torch.quantile(x, 0.25)), "p75": float(torch.quantile(x, 0.75)),
            "p90": float(torch.quantile(x, 0.90)), "minimum": float(x.min()), "maximum": float(x.max()),
            "iqr": float(torch.quantile(x, 0.75) - torch.quantile(x, 0.25)),
            "max_minus_min": float(x.max() - x.min()),
            "worst": float(x.max() if high_is_worse else x.min())}


def canonical_cosines(gram: torch.Tensor, projected_gram: torch.Tensor) -> list[float]:
    gram = (gram + gram.T) * 0.5
    projected_gram = (projected_gram + projected_gram.T) * 0.5
    values, vectors = torch.linalg.eigh(gram)
    threshold = max(float(values.max()) * 1e-7, 1e-12)
    keep = values > threshold
    if not bool(keep.any()):
        return [0.0]
    basis = vectors[:, keep] * values[keep].rsqrt().unsqueeze(0)
    whitened = basis.T @ projected_gram @ basis
    cos2 = torch.linalg.eigvalsh((whitened + whitened.T) * 0.5).clamp(0.0, 1.0)
    return cos2.sqrt().detach().cpu().tolist()


def efficiency(config_payload: dict) -> tuple[int | None, int | None, float | None]:
    item = config_payload.get("efficiency") or {}
    parameters = next((item[key] for key in ("coreflow_resident_adapter_parameters", "variant_resident_adapter_parameters") if key in item), None)
    macs = next((item[key] for key in ("coreflow_adapter_macs_per_token", "variant_adapter_macs_per_token") if key in item), parameters)
    reduction = item.get("resident_parameter_reduction_pct")
    if reduction is None and parameters and item.get("source_resident_adapter_parameters"):
        reduction = 100 * (1 - parameters / item["source_resident_adapter_parameters"])
    return None if parameters is None else int(parameters), None if macs is None else int(macs), reduction


def main() -> None:
    cfg, p = config(), paths()
    seal = p["reports"] / "PRS5_SEAL.json"
    if not seal.is_file():
        raise RuntimeError("Run seal first")
    output_dir = p["results"] / "weight_audit"
    final = output_dir / "weight_audit.json"
    if final.exists():
        print(json.dumps({"status": "PASS", "reused": True, "output": str(final)}))
        return
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists() or output_dir.exists():
        raise RuntimeError(f"Retained partial/final output exists: {partial} or {output_dir}")
    partial.mkdir(parents=True)
    groups = []
    for name in cfg["expert_order"]:
        print(f"[LOAD] source expert={name}", flush=True)
        groups.append(load_adapter(p["asset_root"] / "LoRAs" / cfg["experts"][name]["relative"]))
    module_names = sorted(groups[0])
    if any(set(group) != set(module_names) for group in groups[1:]):
        raise ValueError("Source adapters do not share an identical module set")
    raw_rows, variant_reports = [], []
    for variant_index, (variant, item) in enumerate(cfg["variants"].items(), start=1):
        root = variant_dir(item, p)
        bank_cfg = load_json(root / item["config"])
        tensors = load_file(str(root / item["tensor"]), device="cpu")
        config_modules = {entry["name"]: entry for entry in bank_cfg["modules"]}
        if set(config_modules) != set(module_names):
            raise ValueError(f"Bank/source module mismatch for {variant}")
        expert_config_errors = [[] for _ in cfg["expert_order"]]
        capture_values = {"left": [], "right": [], "joint": [], "left_cos": [], "right_cos": []}
        zero_rank_entries = rank_entries = 0
        for module_index, name in enumerate(module_names, start=1):
            entry = config_modules[name]
            prefix = entry["tensor_prefix"]
            errors = entry.get("fp32_sampled_output_relative_errors") or []
            if len(errors) != len(groups):
                raise ValueError(f"Missing construction errors for {variant}/{name}")
            if item["kind"] == "bilateral":
                left_rank = int(entry.get("effective_left_rank", bank_cfg.get("q", 0)))
                right_rank = int(entry.get("effective_right_rank", bank_cfg.get("q", 0)))
                rank_entries += 2
                zero_rank_entries += int(left_rank == 0) + int(right_rank == 0)
                u = tensors[f"{prefix}.U"].to("cuda", torch.float32)
                v = tensors[f"{prefix}.V"].to("cuda", torch.float32)
            elif item["kind"] == "input":
                rank = int(entry["rank"]); rank_entries += 1; zero_rank_entries += int(rank == 0)
                u, v = None, tensors[f"{prefix}.V"].to("cuda", torch.float32)
            else:
                rank = int(entry["rank"]); rank_entries += 1; zero_rank_entries += int(rank == 0)
                u, v = tensors[f"{prefix}.U"].to("cuda", torch.float32), None
            gu = None if u is None else (u.T @ u)
            gv = None if v is None else (v.T @ v)
            for expert_index, expert_name in enumerate(cfg["expert_order"]):
                source = groups[expert_index][name]
                a = source.a.to("cuda", torch.float32)
                b = source.b.to("cuda", torch.float32)
                ga, gb = a @ a.T, b.T @ b
                denominator = float((gb * ga.T).sum().clamp_min(1e-30))
                left_capture = right_capture = joint_capture = None
                left_cos = right_cos = None
                if u is not None:
                    ub = u.T @ b
                    h_left = ub.T @ torch.linalg.solve(gu, ub)
                    left_capture = min(1.0, max(0.0, float((h_left * ga.T).sum() / denominator)))
                    left_values = canonical_cosines(gb, h_left)
                    left_cos = sum(left_values) / len(left_values)
                    capture_values["left"].append(left_capture)
                    capture_values["left_cos"].append(left_cos)
                if v is not None:
                    av = a @ v
                    h_right = av @ torch.linalg.solve(gv, av.T)
                    right_capture = min(1.0, max(0.0, float((gb * h_right.T).sum() / denominator)))
                    right_values = canonical_cosines(ga, h_right)
                    right_cos = sum(right_values) / len(right_values)
                    capture_values["right"].append(right_capture)
                    capture_values["right_cos"].append(right_cos)
                if u is not None and v is not None:
                    mixed = (u.T @ b) @ (a @ v)
                    coefficients = torch.linalg.solve(gu, mixed)
                    coefficients = torch.linalg.solve(gv, coefficients.T).T
                    projected_norm = (coefficients * (gu @ coefficients @ gv)).sum()
                    joint_capture = min(1.0, max(0.0, float(projected_norm / denominator)))
                    capture_values["joint"].append(joint_capture)
                error = float(errors[expert_index])
                expert_config_errors[expert_index].append(error)
                raw_rows.append({"variant": variant, "module": name, "module_type": name.rsplit(".", 1)[-1],
                                 "expert": expert_name, "construction_error_fp32": error,
                                 "left_energy_capture": left_capture, "right_energy_capture": right_capture,
                                 "joint_energy_capture": joint_capture, "left_mean_canonical_cosine": left_cos,
                                 "right_mean_canonical_cosine": right_cos})
                del a, b, ga, gb
            del u, v, gu, gv
            if module_index == 1 or module_index % 32 == 0 or module_index == len(module_names):
                print(f"[PRS5] variant={variant} modules={module_index}/{len(module_names)}", flush=True)
            torch.cuda.empty_cache()
        expert_means = [sum(values) / len(values) for values in expert_config_errors]
        parameters, macs, reduction = efficiency(bank_cfg)
        metric_label = "core_q185" if variant.startswith("A0_") else variant
        legacy_metric = next(row for row in load_json((Path(__file__).resolve().parents[1] / "evidence" / "phase2b_weight_metrics.json"))["methods"] if row["label"] == metric_label)
        report = {"variant": variant, "kind": item["kind"], "modules": len(module_names),
                  "logical_parameters": parameters, "adapter_macs_per_token": macs,
                  "parameter_reduction_pct": reduction, "tensor_bytes": (root / item["tensor"]).stat().st_size,
                  "artifact_bytes": (root / item["tensor"]).stat().st_size + (root / item["config"]).stat().st_size,
                  "tensor_sha256": sha256_file(root / item["tensor"]), "config_sha256": sha256_file(root / item["config"]),
                  "structural_zero_rank_entries": zero_rank_entries, "structural_rank_entries": rank_entries,
                  "expert_names": cfg["expert_order"], "construction_error_mean_by_expert": expert_means,
                  "construction_error_expert_dispersion": stats(expert_means),
                  "fused_relative_output_error": legacy_metric["fused_relative_output_error"],
                  "runtime_bf16_per_expert_relative_output_error": legacy_metric["per_expert_relative_output_error"]}
        for key, values in capture_values.items():
            report[key] = None if not values else stats(values, high_is_worse=False)
        variant_reports.append(report)
        del tensors
        torch.cuda.empty_cache()
    raw_path = partial / "module_expert_metrics.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader(); writer.writerows(raw_rows)
    payload = {"status": "ANALYZED_NO_GENERATION", "protocol_id": cfg["protocol_id"], "timestamp_utc": now_utc(),
               "selection_forbidden": True, "main_method_change_forbidden": True,
               "definitions": cfg["audit"], "variants": variant_reports,
               "raw_rows": len(raw_rows), "raw_csv": "module_expert_metrics.csv"}
    write_json(partial / "weight_audit.json", payload)
    partial.rename(output_dir)
    write_json(p["reports"] / "weight_audit_complete.json",
               {"status": "PASS", "protocol_id": cfg["protocol_id"], "variants": len(variant_reports), "raw_rows": len(raw_rows)})
    print(json.dumps({"status": payload["status"], "variants": len(variant_reports), "raw_rows": len(raw_rows)}))


if __name__ == "__main__":
    main()
