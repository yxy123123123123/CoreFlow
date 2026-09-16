from __future__ import annotations

import math
import types
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .decompose import build_module_bases, project_core, sampled_output_relative_error, stable_orthonormal_basis
from .io import AdapterModule, canonical_module_name, ensure_same_modules, load_json, sha256_file, write_json


def _metadata(source_names: Sequence[str], source_metadata: Sequence[dict]) -> list[dict]:
    if len(source_names) != len(source_metadata):
        raise ValueError("source metadata length mismatch")
    rows = []
    for index, (name, item) in enumerate(zip(source_names, source_metadata)):
        row = dict(item)
        row.setdefault("name", name)
        row.setdefault("expert_index", index)
        if not row.get("checkpoint_sha256") and row.get("adapter_sha256"):
            row["checkpoint_sha256"] = row["adapter_sha256"]
        rows.append(row)
    return rows


def _plain_bases(sources: Sequence[AdapterModule], weights: Sequence[float]):
    left = torch.cat(
        [source.b * (source.scaling * math.sqrt(float(weight))) for source, weight in zip(sources, weights)],
        dim=1,
    )
    right = torch.cat(
        [source.a.T * math.sqrt(float(weight)) for source, weight in zip(sources, weights)],
        dim=1,
    )
    u, su, ql, dl = stable_orthonormal_basis(left)
    v, sv, qr, dr = stable_orthonormal_basis(right)
    return u, v, {"basis_method": "plain_factor_concatenation", "q_left_numerical": ql,
                  "q_right_numerical": qr, "left": dl, "right": dr,
                  "left_singular_values": su.cpu().tolist(), "right_singular_values": sv.cpu().tolist()}


def build_bilateral_variant(
    source_groups: Sequence[Mapping[str, AdapterModule]],
    source_names: Sequence[str],
    output_dir: str | Path,
    q: int,
    source_metadata: Sequence[dict],
    *,
    variant: str,
    expert_weights: Sequence[float] | None = None,
    seed: int = 20260801,
    save_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> dict:
    from safetensors.torch import save_file

    names = ensure_same_modules(source_groups)
    k = len(source_groups)
    weights = list(expert_weights or [1.0 / k] * k)
    if len(weights) != k or any(value <= 0 for value in weights):
        raise ValueError("expert weights must be positive and follow expert order")
    total = sum(weights)
    weights = [value / total for value in weights]
    tensors = {}
    modules = []
    source_params = variant_params = 0
    for module_index, name in enumerate(names):
        sources = [
            AdapterModule(group[name].name, group[name].a.to(device), group[name].b.to(device), group[name].scaling)
            for group in source_groups
        ]
        if variant in {"A1_bilateral_no_norm"}:
            u_full, v_full, report = build_module_bases(sources, weights, balanced=False)
        elif variant in {"A4_plain_concat_svd"}:
            u_full, v_full, report = _plain_bases(sources, weights)
        elif variant in {"A5_gate_frequency_weighted"}:
            u_full, v_full, report = build_module_bases(sources, weights, balanced=True)
        elif variant == "A6_random_orthogonal":
            din, dout = sources[0].in_features, sources[0].out_features
            gen = torch.Generator(device=device).manual_seed(seed + module_index * 1009)
            u_full = torch.linalg.qr(torch.randn(dout, q, generator=gen, device=device), mode="reduced")[0]
            v_full = torch.linalg.qr(torch.randn(din, q, generator=gen, device=device), mode="reduced")[0]
            report = {"basis_method": "random_orthogonal_negative_control", "seed": seed + module_index * 1009,
                      "q_left_numerical": int(u_full.shape[1]), "q_right_numerical": int(v_full.shape[1])}
        else:
            raise ValueError(f"unsupported bilateral variant: {variant}")
        left_rank, right_rank = min(q, u_full.shape[1]), min(q, v_full.shape[1])
        u = u_full[:, :left_rank].contiguous()
        v = v_full[:, :right_rank].contiguous()
        cores = torch.stack([project_core(u, v, source) for source in sources], dim=0).contiguous()
        errors = [sampled_output_relative_error(u, v, cores[i], source, seed + module_index * 31 + i)
                  for i, source in enumerate(sources)]
        prefix = f"m{module_index:03d}"
        tensors[f"{prefix}.U"] = u.to(dtype=save_dtype, device="cpu")
        tensors[f"{prefix}.V"] = v.to(dtype=save_dtype, device="cpu")
        tensors[f"{prefix}.C"] = cores.to(dtype=save_dtype, device="cpu")
        din, dout = sources[0].in_features, sources[0].out_features
        ranks = [source.rank for source in sources]
        source_params += sum(rank * (din + dout) for rank in ranks)
        variant_params += left_rank * dout + right_rank * din + k * left_rank * right_rank
        modules.append({"name": name, "tensor_prefix": prefix, "in_features": din, "out_features": dout,
                        "source_ranks": ranks, "requested_q": q, "effective_left_rank": left_rank,
                        "effective_right_rank": right_rank, "invalid_included_direction_count": 0,
                        "fp32_sampled_output_relative_errors": errors, "basis_report": report})
        del sources, u_full, v_full, u, v, cores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "core_bank.safetensors"
    save_file(tensors, str(tensor_path), metadata={"format": "coreflow-oracle-fix-v1", "schema_version": "2"})
    config = {"format": "coreflow-oracle-fix-v1", "schema_version": 2, "variant": variant, "q": q,
              "expert_order": list(source_names), "source_order": list(source_names),
              "experts": _metadata(source_names, source_metadata), "expert_weights": weights,
              "scaling_absorbed_into_core": True, "runtime_must_not_apply_source_scaling": True,
              "scale_absorbed_exactly_once": True, "storage_dtype": str(save_dtype).replace("torch.", ""),
              "modules": modules, "efficiency": {"source_resident_adapter_parameters": source_params,
              "variant_resident_adapter_parameters": variant_params,
              "source_adapter_macs_per_token": source_params,
              "variant_adapter_macs_per_token": variant_params,
              "resident_parameter_reduction_pct": 100.0 * (1.0 - variant_params / source_params)}}
    write_json(output_dir / "core_config.json", config)
    return {"variant": variant, "bank_sha256": sha256_file(tensor_path), "config": config}


def build_one_sided_variant(
    source_groups: Sequence[Mapping[str, AdapterModule]], source_names: Sequence[str], output_dir: str | Path,
    source_metadata: Sequence[dict], *, side: str, q0: int = 185, seed: int = 20260801,
    save_dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
) -> dict:
    from safetensors.torch import save_file

    if side not in {"input", "output"}:
        raise ValueError(side)
    names = ensure_same_modules(source_groups)
    k = len(source_groups)
    weights = [1.0 / k] * k
    tensors, modules = {}, []
    source_params = target_params = used_params = 0
    for module_index, name in enumerate(names):
        sources = [AdapterModule(group[name].name, group[name].a.to(device), group[name].b.to(device), group[name].scaling)
                   for group in source_groups]
        u_full, v_full, _ = build_module_bases(sources, weights, balanced=True)
        din, dout = sources[0].in_features, sources[0].out_features
        a0_budget = q0 * (din + dout) + k * q0 * q0
        denom = din + k * dout if side == "input" else dout + k * din
        rank = min(int(a0_budget // denom), v_full.shape[1] if side == "input" else u_full.shape[1])
        if rank < 1:
            raise ValueError(f"one-sided rank collapsed at {name}")
        prefix = f"m{module_index:03d}"
        errors = []
        gen = torch.Generator(device=device).manual_seed(seed + module_index * 31)
        x = torch.randn(4, din, generator=gen, device=device)
        if side == "input":
            v = v_full[:, :rank].contiguous()
            factors = torch.stack([source.scaling * (source.b @ (source.a @ v)) for source in sources], dim=0)
            tensors[f"{prefix}.V"] = v.to(dtype=save_dtype, device="cpu")
            tensors[f"{prefix}.D"] = factors.to(dtype=save_dtype, device="cpu")
            for i, source in enumerate(sources):
                truth = ((x @ source.a.T) @ source.b.T) * source.scaling
                candidate = (x @ v) @ factors[i].T
                errors.append(float(torch.linalg.vector_norm(candidate - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-30)))
        else:
            u = u_full[:, :rank].contiguous()
            factors = torch.stack([source.scaling * ((u.T @ source.b) @ source.a) for source in sources], dim=0)
            tensors[f"{prefix}.U"] = u.to(dtype=save_dtype, device="cpu")
            tensors[f"{prefix}.E"] = factors.to(dtype=save_dtype, device="cpu")
            for i, source in enumerate(sources):
                truth = ((x @ source.a.T) @ source.b.T) * source.scaling
                candidate = (x @ factors[i].T) @ u.T
                errors.append(float(torch.linalg.vector_norm(candidate - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-30)))
        source_here = sum(source.rank * (din + dout) for source in sources)
        used_here = rank * denom
        source_params += source_here
        target_params += a0_budget
        used_params += used_here
        modules.append({"name": name, "tensor_prefix": prefix, "in_features": din, "out_features": dout,
                        "rank": rank, "a0_module_budget": a0_budget, "variant_module_parameters": used_here,
                        "budget_gap": a0_budget - used_here, "source_ranks": [source.rank for source in sources],
                        "fp32_sampled_output_relative_errors": errors})
        del sources, u_full, v_full, factors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "one_sided_bank.safetensors"
    save_file(tensors, str(tensor_path), metadata={"format": "coreflow-one-sided-v1", "schema_version": "1"})
    config = {"format": "coreflow-one-sided-v1", "schema_version": 1, "variant": f"A2_{side}_only" if side == "input" else "A3_output_only",
              "side": side, "q0_budget_reference": q0, "expert_order": list(source_names),
              "experts": _metadata(source_names, source_metadata), "scaling_absorbed": True, "modules": modules,
              "efficiency": {"source_resident_adapter_parameters": source_params,
              "a0_target_parameters": target_params, "variant_resident_adapter_parameters": used_params,
              "source_adapter_macs_per_token": source_params, "a0_target_macs_per_token": target_params,
              "variant_adapter_macs_per_token": used_params,
              "budget_gap_parameters": target_params - used_params}}
    write_json(output_dir / "one_sided_config.json", config)
    return {"variant": config["variant"], "bank_sha256": sha256_file(tensor_path), "config": config}


def one_sided_delta(x: torch.Tensor, weights: torch.Tensor, shared: torch.Tensor, factors: torch.Tensor, side: str):
    dtype = x.dtype
    work = x.to(shared.dtype)
    if weights.ndim == x.ndim + 1 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if side == "input":
        z = torch.matmul(work, shared)
        mixed = torch.einsum("...r,tor,...t->...o", z, factors, weights.to(z.dtype))
    else:
        z = torch.einsum("...i,tri->...tr", work, factors)
        mixed = torch.einsum("...tr,...t->...r", z, weights.to(z.dtype))
        mixed = torch.matmul(mixed, shared.T)
    return mixed.to(dtype)


def _one_sided_forward(self, x, lora_weights=None):
    result = self.base_layer(x)
    if lora_weights is None:
        raise ValueError("one-sided CoreFlow requires gate weights")
    delta = one_sided_delta(x, lora_weights, self._one_shared, self._one_factors, self._one_side)
    return (result + delta).to(x.dtype)


def install_one_sided(model, bank_dir: str | Path, expected_expert_order: Sequence[str], release_sources: bool = True):
    from safetensors.torch import load_file

    bank_dir = Path(bank_dir)
    config = load_json(bank_dir / "one_sided_config.json")
    if config.get("format") != "coreflow-one-sided-v1" or config.get("schema_version") != 1:
        raise ValueError("unsupported one-sided bank")
    if list(config.get("expert_order") or []) != list(expected_expert_order):
        raise ValueError("one-sided expert order mismatch")
    tensors = load_file(str(bank_dir / "one_sided_bank.safetensors"), device="cpu")
    module_config = {item["name"]: item for item in config["modules"]}
    installed = []
    for raw_name, module in model.named_modules():
        name = canonical_module_name(raw_name)
        if name not in module_config:
            continue
        item = module_config[name]
        prefix = item["tensor_prefix"]
        if config["side"] == "input":
            shared, factors = tensors[f"{prefix}.V"], tensors[f"{prefix}.D"]
        else:
            shared, factors = tensors[f"{prefix}.U"], tensors[f"{prefix}.E"]
        device, dtype = module.base_layer.weight.device, module.base_layer.weight.dtype
        module.register_buffer("_one_shared", shared.to(device=device, dtype=dtype), persistent=False)
        module.register_buffer("_one_factors", factors.to(device=device, dtype=dtype), persistent=False)
        module._one_side = config["side"]
        module.forward = types.MethodType(_one_sided_forward, module)
        if release_sources:
            module.lora_A.clear(); module.lora_B.clear()
            if hasattr(module, "lora_dropout"):
                module.lora_dropout.clear()
        installed.append(name)
    missing = sorted(set(module_config) - set(installed))
    if missing:
        raise ValueError(f"one-sided bank unmatched modules: {missing[:5]}")
    return {"installed_modules": len(installed), "side": config["side"], "bank_sha256": sha256_file(bank_dir / "one_sided_bank.safetensors")}
