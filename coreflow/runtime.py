from __future__ import annotations

import types
from pathlib import Path

import torch

from .io import canonical_module_name, load_json, sha256_file


def coreflow_delta(x: torch.Tensor, weights: torch.Tensor, u: torch.Tensor, v: torch.Tensor, cores: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_work = x.to(v.dtype)
    z = torch.matmul(x_work, v)
    if weights.ndim == x.ndim + 1 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape[:-1] != x.shape[:-1] or weights.shape[-1] != cores.shape[0]:
        raise ValueError(f"Bad gate shape {tuple(weights.shape)} for x={tuple(x.shape)}, cores={tuple(cores.shape)}")
    mixed = torch.einsum("...r,tlr,...t->...l", z, cores, weights.to(z.dtype))
    return torch.matmul(mixed, u.T).to(dtype)


def isvd_delta(x: torch.Tensor, weights: torch.Tensor, b: torch.Tensor, v: torch.Tensor, ranks: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_work = x.to(v.dtype)
    if weights.ndim == x.ndim + 1 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape[:-1] != x.shape[:-1] or weights.shape[-1] != b.shape[0]:
        raise ValueError(f"Bad gate shape {tuple(weights.shape)} for x={tuple(x.shape)}, factors={tuple(b.shape)}")
    z = torch.einsum("...i,tir->...tr", x_work, v)
    if ranks.numel() and int(ranks.max().item()) < v.shape[-1]:
        mask = torch.arange(v.shape[-1], device=x.device).unsqueeze(0) < ranks.to(device=x.device).unsqueeze(1)
        z = z * mask.to(z.dtype)
    mixed = torch.einsum("...tr,tor,...t->...o", z, b, weights.to(z.dtype))
    return mixed.to(dtype)


def _patched_forward(self, x: torch.Tensor, lora_weights=None):
    result = self.base_layer(x)
    if lora_weights is None:
        raise ValueError("CoreFlow requires layer-level LoRA-Flow weights")
    if self._coreflow_uniform:
        shape = (*x.shape[:-1], self._coreflow_cores.shape[0])
        lora_weights = torch.full(shape, 1.0 / self._coreflow_cores.shape[0], device=x.device, dtype=x.dtype)
    with torch.autograd.profiler.record_function("coreflow_adapter_path"):
        delta = coreflow_delta(
            x,
            lora_weights,
            self._coreflow_u,
            self._coreflow_v,
            self._coreflow_cores,
        )
    return (result + delta).to(x.dtype)


def _patched_isvd_forward(self, x: torch.Tensor, lora_weights=None):
    result = self.base_layer(x)
    if lora_weights is None:
        raise ValueError("Independent-SVD requires layer-level LoRA-Flow weights")
    delta = isvd_delta(x, lora_weights, self._isvd_b, self._isvd_v, self._isvd_ranks)
    return (result + delta).to(x.dtype)


def install_coreflow(
    model,
    bank_dir: str | Path,
    uniform: bool = False,
    release_sources: bool = True,
    expected_expert_order: list[str] | tuple[str, ...] = ("zh", "task"),
) -> dict:
    from safetensors.torch import load_file

    bank_dir = Path(bank_dir)
    config = load_json(bank_dir / "core_config.json")
    if config.get("format") != "coreflow-oracle-fix-v1" or config.get("schema_version") != 2:
        raise ValueError(f"Unsupported CoreFlow bank schema: {config.get('format')}/{config.get('schema_version')}")
    expert_order = config.get("expert_order")
    if list(expert_order or []) != list(expected_expert_order):
        raise ValueError(f"Expert order mismatch: bank={expert_order}, expected={list(expected_expert_order)}")
    experts = [dict(item) for item in (config.get("experts") or [])]
    for index, item in enumerate(experts):
        item.setdefault("expert_index", index)
        if not item.get("checkpoint_sha256") and item.get("adapter_sha256"):
            item["checkpoint_sha256"] = item["adapter_sha256"]
    if len(experts) != len(expert_order) or [item.get("expert_index") for item in experts] != list(range(len(expert_order))):
        raise ValueError(f"Expert index metadata is invalid: {experts}")
    if [item.get("name") for item in experts] != list(expert_order):
        raise ValueError(f"Expert name metadata is inconsistent: {experts}")
    if any(not item.get("checkpoint_sha256") for item in experts):
        raise ValueError("Expert checkpoint hashes are required")
    if not config.get("scaling_absorbed_into_core") or not config.get("runtime_must_not_apply_source_scaling"):
        raise ValueError("Bank scaling contract is missing or ambiguous")
    tensors = load_file(str(bank_dir / "core_bank.safetensors"), device="cpu")
    module_config = {item["name"]: item for item in config["modules"]}
    installed = []
    device_counts = {}
    for raw_name, module in model.named_modules():
        name = canonical_module_name(raw_name)
        if name not in module_config:
            continue
        if not hasattr(module, "base_layer") or not hasattr(module, "lora_A"):
            raise TypeError(f"Target is not a PEFT LoRA linear layer: {raw_name} ({type(module)})")
        item = module_config[name]
        prefix = item["tensor_prefix"]
        u_cpu = tensors[f"{prefix}.U"]
        v_cpu = tensors[f"{prefix}.V"]
        cores_cpu = tensors[f"{prefix}.C"]
        expected_left = int(item["effective_left_rank"])
        expected_right = int(item["effective_right_rank"])
        expected_shapes = {
            "U": (int(item["out_features"]), expected_left),
            "V": (int(item["in_features"]), expected_right),
            "C": (len(expert_order), expected_left, expected_right),
        }
        actual_shapes = {"U": tuple(u_cpu.shape), "V": tuple(v_cpu.shape), "C": tuple(cores_cpu.shape)}
        if actual_shapes != expected_shapes:
            raise ValueError(f"Bank tensor shape mismatch at {name}: actual={actual_shapes}, expected={expected_shapes}")
        if item.get("invalid_included_direction_count") != 0:
            raise ValueError(f"Invalid basis directions recorded at {name}")
        device = module.base_layer.weight.device
        dtype = module.base_layer.weight.dtype
        module.register_buffer("_coreflow_u", u_cpu.to(device=device, dtype=dtype), persistent=False)
        module.register_buffer("_coreflow_v", v_cpu.to(device=device, dtype=dtype), persistent=False)
        module.register_buffer("_coreflow_cores", cores_cpu.to(device=device, dtype=dtype), persistent=False)
        module._coreflow_uniform = bool(uniform)
        module.forward = types.MethodType(_patched_forward, module)
        if release_sources:
            module.lora_A.clear()
            module.lora_B.clear()
            if hasattr(module, "lora_dropout"):
                module.lora_dropout.clear()
        installed.append(name)
        device_counts[str(device)] = device_counts.get(str(device), 0) + 1
    missing = sorted(set(module_config) - set(installed))
    if missing:
        raise ValueError(f"Core bank has {len(missing)} unmatched modules, first={missing[:5]}")
    return {
        "installed_modules": len(installed),
        "uniform": uniform,
        "device_counts": device_counts,
        "q": config["q"],
        "schema_version": config["schema_version"],
        "expert_order": expert_order,
        "scaling_absorbed_into_core": True,
        "bank_sha256": sha256_file(bank_dir / "core_bank.safetensors"),
    }


def install_isvd(
    model,
    bank_dir: str | Path,
    release_sources: bool = True,
    expected_expert_order: list[str] | tuple[str, ...] = ("zh", "task"),
) -> dict:
    from safetensors.torch import load_file

    bank_dir = Path(bank_dir)
    config = load_json(bank_dir / "isvd_config.json")
    if config.get("format") != "coreflow-independent-svd-v1" or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported Independent-SVD bank schema: {config.get('format')}/{config.get('schema_version')}")
    expert_order = config.get("expert_order")
    if list(expert_order or []) != list(expected_expert_order):
        raise ValueError(f"Expert order mismatch: bank={expert_order}, expected={list(expected_expert_order)}")
    experts = [dict(item) for item in (config.get("experts") or [])]
    if len(experts) != len(expert_order) or [item.get("expert_index") for item in experts] != list(range(len(expert_order))):
        raise ValueError(f"Expert index metadata is invalid: {experts}")
    if [item.get("name") for item in experts] != list(expert_order):
        raise ValueError(f"Expert name metadata is inconsistent: {experts}")
    tensors = load_file(str(bank_dir / "isvd_bank.safetensors"), device="cpu")
    module_config = {item["name"]: item for item in config["modules"]}
    installed = []
    device_counts = {}
    for raw_name, module in model.named_modules():
        name = canonical_module_name(raw_name)
        if name not in module_config:
            continue
        if not hasattr(module, "base_layer") or not hasattr(module, "lora_A"):
            raise TypeError(f"Target is not a PEFT LoRA linear layer: {raw_name} ({type(module)})")
        item = module_config[name]
        prefix = item["tensor_prefix"]
        b_cpu = tensors[f"{prefix}.B"]
        v_cpu = tensors[f"{prefix}.V"]
        expected_shapes = {
            "B": (len(expert_order), int(item["out_features"]), int(item["max_rank"])),
            "V": (len(expert_order), int(item["in_features"]), int(item["max_rank"])),
        }
        actual_shapes = {"B": tuple(b_cpu.shape), "V": tuple(v_cpu.shape)}
        if actual_shapes != expected_shapes:
            raise ValueError(f"ISVD tensor shape mismatch at {name}: actual={actual_shapes}, expected={expected_shapes}")
        ranks = torch.tensor(item["expert_ranks"], dtype=torch.long)
        device = module.base_layer.weight.device
        dtype = module.base_layer.weight.dtype
        module.register_buffer("_isvd_b", b_cpu.to(device=device, dtype=dtype), persistent=False)
        module.register_buffer("_isvd_v", v_cpu.to(device=device, dtype=dtype), persistent=False)
        module.register_buffer("_isvd_ranks", ranks.to(device=device), persistent=False)
        module.forward = types.MethodType(_patched_isvd_forward, module)
        if release_sources:
            module.lora_A.clear()
            module.lora_B.clear()
            if hasattr(module, "lora_dropout"):
                module.lora_dropout.clear()
        installed.append(name)
        device_counts[str(device)] = device_counts.get(str(device), 0) + 1
    missing = sorted(set(module_config) - set(installed))
    if missing:
        raise ValueError(f"ISVD bank has {len(missing)} unmatched modules, first={missing[:5]}")
    return {
        "installed_modules": len(installed),
        "device_counts": device_counts,
        "schema_version": config["schema_version"],
        "expert_order": expert_order,
        "budget": config["budget"],
        "bank_sha256": sha256_file(bank_dir / "isvd_bank.safetensors"),
    }


def packed_delta(
    x: torch.Tensor,
    weights: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    dim_to_expert: torch.Tensor,
) -> torch.Tensor:
    """Apply a ragged multi-expert low-rank bank without expert padding.

    Every retained rank direction is stored once. ``dim_to_expert`` selects the
    gate weight that belongs to that direction, so the implementation uses two
    dense matmuls and one elementwise multiply instead of a Python expert loop.
    """
    dtype = x.dtype
    # A single concatenated FP16 GEMM changes the reduction tree relative to
    # the legacy expert-major contraction.  For heterogeneous ragged ranks this
    # can amplify cancellation error even though the algebra is identical.
    # Accumulating the packed path in FP32 restores (and improves) numerical
    # accuracy while retaining compact storage and a loop-free forward path.
    # The frozen BF16 deployment path is intentionally unchanged.
    accumulation_dtype = torch.float32 if dtype == torch.float16 else v.dtype
    x_work = x.to(accumulation_dtype)
    v_work = v.to(accumulation_dtype)
    b_work = b.to(accumulation_dtype)
    if weights.ndim == x.ndim + 1 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.shape[:-1] != x.shape[:-1]:
        raise ValueError(f"Bad gate prefix {tuple(weights.shape)} for x={tuple(x.shape)}")
    if dim_to_expert.numel() == 0:
        return torch.zeros((*x.shape[:-1], b.shape[0]), dtype=dtype, device=x.device)
    if int(dim_to_expert.max().item()) >= weights.shape[-1]:
        raise ValueError(
            f"Gate K={weights.shape[-1]} does not cover packed expert index "
            f"{int(dim_to_expert.max().item())}"
        )
    latent = torch.matmul(x_work, v_work)
    per_direction_weights = weights.to(latent.dtype).index_select(-1, dim_to_expert)
    return torch.matmul(latent * per_direction_weights, b_work.T).to(dtype)


def _patched_packed_forward(self, x: torch.Tensor, lora_weights=None):
    result = self.base_layer(x)
    if lora_weights is None:
        raise ValueError("Packed dynamic LoRA runtime requires layer-level LoRA-Flow weights")
    with torch.autograd.profiler.record_function("full_vectorized_adapter_path"):
        delta = packed_delta(
            x,
            lora_weights,
            self._packed_b,
            self._packed_v,
            self._packed_dim_to_expert,
        )
    return (result + delta).to(x.dtype)


def _source_scaling(module, adapter_name: str) -> float:
    scaling = getattr(module, "scaling", None)
    if scaling is None:
        raise TypeError("PEFT LoRA layer has no scaling map")
    if hasattr(scaling, "__getitem__"):
        value = scaling[adapter_name]
    else:
        raise TypeError(f"Unsupported scaling container: {type(scaling)}")
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _release_source_lora(module) -> None:
    module.lora_A.clear()
    module.lora_B.clear()
    if hasattr(module, "lora_dropout"):
        module.lora_dropout.clear()


def install_full_vectorized(
    model,
    *,
    expected_expert_order: list[str] | tuple[str, ...],
    release_sources: bool = True,
) -> dict:
    """Pack the exact source LoRAs into a no-expert-loop dynamic runtime."""
    order = list(expected_expert_order)
    installed = []
    logical_parameters = 0
    serialized_bytes = 0
    device_counts = {}
    for raw_name, module in model.named_modules():
        if not hasattr(module, "base_layer") or not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if not all(name in module.lora_A and name in module.lora_B for name in order):
            continue
        device = module.base_layer.weight.device
        dtype = module.base_layer.weight.dtype
        v_parts = []
        b_parts = []
        owner = []
        ranks = []
        for expert_index, name in enumerate(order):
            a = module.lora_A[name].weight.detach()
            b = module.lora_B[name].weight.detach()
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(
                    f"Invalid source LoRA shapes at {raw_name}/{name}: A={tuple(a.shape)}, B={tuple(b.shape)}"
                )
            rank = int(a.shape[0])
            ranks.append(rank)
            v_parts.append(a.T.to(device=device, dtype=dtype))
            b_parts.append((b * _source_scaling(module, name)).to(device=device, dtype=dtype))
            owner.extend([expert_index] * rank)
            logical_parameters += rank * (int(a.shape[1]) + int(b.shape[0]))
        v_cat = torch.cat(v_parts, dim=1).contiguous()
        b_cat = torch.cat(b_parts, dim=1).contiguous()
        owner_tensor = torch.tensor(owner, dtype=torch.long, device=device)
        module.register_buffer("_packed_v", v_cat, persistent=False)
        module.register_buffer("_packed_b", b_cat, persistent=False)
        module.register_buffer("_packed_dim_to_expert", owner_tensor, persistent=False)
        module._packed_runtime_kind = "full_vectorized"
        module.forward = types.MethodType(_patched_packed_forward, module)
        if release_sources:
            _release_source_lora(module)
        serialized_bytes += v_cat.numel() * v_cat.element_size()
        serialized_bytes += b_cat.numel() * b_cat.element_size()
        installed.append({"name": canonical_module_name(raw_name), "ranks": ranks, "packed_rank": len(owner)})
        device_counts[str(device)] = device_counts.get(str(device), 0) + 1
    if not installed:
        raise ValueError("No PEFT LoRA modules were eligible for Full-vectorized packing")
    return {
        "runtime_kind": "full_vectorized",
        "installed_modules": len(installed),
        "expert_order": order,
        "logical_adapter_parameters": int(logical_parameters),
        "packed_tensor_bytes": int(serialized_bytes),
        "device_counts": device_counts,
        "no_per_expert_python_forward_loop": True,
        "module_rank_patterns": sorted({tuple(item["ranks"]) for item in installed}),
    }


def install_isvd_compact(
    model,
    bank_dir: str | Path,
    *,
    expected_expert_order: list[str] | tuple[str, ...],
    release_sources: bool = True,
) -> dict:
    """Install the frozen ISVD matrices in ragged, padding-free form."""
    from safetensors import safe_open

    bank_dir = Path(bank_dir)
    config = load_json(bank_dir / "isvd_config.json")
    if config.get("format") != "coreflow-independent-svd-v1" or config.get("schema_version") != 1:
        raise ValueError(f"Unsupported Independent-SVD bank schema: {config.get('format')}/{config.get('schema_version')}")
    order = list(expected_expert_order)
    if list(config.get("expert_order") or []) != order:
        raise ValueError(f"Expert order mismatch: bank={config.get('expert_order')}, expected={order}")
    module_config = {item["name"]: item for item in config["modules"]}
    tensor_path = bank_dir / "isvd_bank.safetensors"
    installed = []
    compact_parameters = 0
    padded_parameters = 0
    compact_bytes = 0
    device_counts = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        for raw_name, module in model.named_modules():
            name = canonical_module_name(raw_name)
            if name not in module_config:
                continue
            if not hasattr(module, "base_layer") or not hasattr(module, "lora_A"):
                raise TypeError(f"Target is not a PEFT LoRA linear layer: {raw_name} ({type(module)})")
            item = module_config[name]
            prefix = item["tensor_prefix"]
            b_pad = handle.get_tensor(f"{prefix}.B")
            v_pad = handle.get_tensor(f"{prefix}.V")
            ranks = [int(value) for value in item["expert_ranks"]]
            if len(ranks) != len(order):
                raise ValueError(f"Rank list length mismatch at {name}: {ranks}")
            b_parts = []
            v_parts = []
            owner = []
            for expert_index, rank in enumerate(ranks):
                if rank <= 0:
                    continue
                b_parts.append(b_pad[expert_index, :, :rank])
                v_parts.append(v_pad[expert_index, :, :rank])
                owner.extend([expert_index] * rank)
            device = module.base_layer.weight.device
            dtype = module.base_layer.weight.dtype
            if owner:
                b_cat = torch.cat(b_parts, dim=1).to(device=device, dtype=dtype).contiguous()
                v_cat = torch.cat(v_parts, dim=1).to(device=device, dtype=dtype).contiguous()
            else:
                b_cat = torch.empty((int(item["out_features"]), 0), device=device, dtype=dtype)
                v_cat = torch.empty((int(item["in_features"]), 0), device=device, dtype=dtype)
            owner_tensor = torch.tensor(owner, dtype=torch.long, device=device)
            module.register_buffer("_packed_v", v_cat, persistent=False)
            module.register_buffer("_packed_b", b_cat, persistent=False)
            module.register_buffer("_packed_dim_to_expert", owner_tensor, persistent=False)
            module._packed_runtime_kind = "isvd_compact"
            module.forward = types.MethodType(_patched_packed_forward, module)
            if release_sources:
                _release_source_lora(module)
            compact_parameters += int(sum(ranks)) * (int(item["in_features"]) + int(item["out_features"]))
            padded_parameters += len(order) * int(item["max_rank"]) * (
                int(item["in_features"]) + int(item["out_features"])
            )
            compact_bytes += v_cat.numel() * v_cat.element_size()
            compact_bytes += b_cat.numel() * b_cat.element_size()
            installed.append({"name": name, "ranks": ranks, "packed_rank": len(owner)})
            device_counts[str(device)] = device_counts.get(str(device), 0) + 1
    missing = sorted(set(module_config) - {item["name"] for item in installed})
    if missing:
        raise ValueError(f"ISVD bank has {len(missing)} unmatched modules, first={missing[:5]}")
    return {
        "runtime_kind": "isvd_compact",
        "installed_modules": len(installed),
        "expert_order": order,
        "logical_adapter_parameters": int(compact_parameters),
        "legacy_padded_parameters": int(padded_parameters),
        "padding_parameters_eliminated": int(padded_parameters - compact_parameters),
        "padding_reduction_pct": 100.0 * (1.0 - compact_parameters / max(padded_parameters, 1)),
        "compact_tensor_bytes": int(compact_bytes),
        "device_counts": device_counts,
        "budget": config["budget"],
        "bank_sha256": sha256_file(tensor_path),
        "no_per_expert_python_forward_loop": True,
    }
