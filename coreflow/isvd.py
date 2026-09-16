from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .io import AdapterModule, ensure_same_modules, sha256_file, write_json


def _thin_product_svd(module: AdapterModule) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of scaling * B @ A without forming the dense out x in matrix."""
    b = (module.b * module.scaling).float()
    a_t = module.a.T.float()
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a_t, mode="reduced")
    u_small, singular, vh_small = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    u = (q_b @ u_small).contiguous()
    v = (q_a @ vh_small.T).contiguous()
    return u, singular.contiguous(), v


def _module_cost(module: AdapterModule) -> int:
    return int(module.in_features + module.out_features)


def allocate_isvd_ranks(
    source_groups: Sequence[Mapping[str, AdapterModule]],
    budget_parameters: int,
) -> tuple[dict[tuple[int, str], int], dict]:
    names = ensure_same_modules(source_groups)
    candidates: list[tuple[float, float, int, int, str, int]] = []
    spectra: dict[tuple[int, str], list[float]] = {}
    costs: dict[str, int] = {}
    source_full_parameters = 0
    for expert_index, group in enumerate(source_groups):
        for name in names:
            module = group[name]
            cost = _module_cost(module)
            costs[name] = cost
            source_full_parameters += module.rank * cost
            _, singular, _ = _thin_product_svd(module)
            values = [float(item) for item in singular.cpu()]
            spectra[(expert_index, name)] = values
            for rank_index, value in enumerate(values):
                gain = value * value
                candidates.append((gain / max(cost, 1), gain, expert_index, cost, name, rank_index + 1))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[4], item[5]))
    ranks = {(expert_index, name): 0 for expert_index in range(len(source_groups)) for name in names}
    used = 0
    for _, gain, expert_index, cost, name, next_rank in candidates:
        key = (expert_index, name)
        if ranks[key] + 1 != next_rank:
            continue
        if used + cost > budget_parameters:
            continue
        ranks[key] = next_rank
        used += cost
    retained_energy = {}
    for key, values in spectra.items():
        total = sum(value * value for value in values)
        kept = sum(value * value for value in values[: ranks[key]])
        retained_energy[f"expert{key[0]}::{key[1]}"] = None if total <= 0 else kept / total
    report = {
        "allocation_rule": "global greedy by singular_value_squared_per_parameter_cost; no quality feedback",
        "budget_parameters": int(budget_parameters),
        "used_parameters": int(used),
        "budget_gap_parameters": int(budget_parameters - used),
        "budget_gap_fraction_of_budget": (budget_parameters - used) / max(float(budget_parameters), 1.0),
        "source_full_parameters": int(source_full_parameters),
        "rank_entries": len(ranks),
        "zero_rank_entries": int(sum(value == 0 for value in ranks.values())),
        "mean_retained_energy": sum(value for value in retained_energy.values() if value is not None) / max(1, sum(value is not None for value in retained_energy.values())),
        "min_retained_energy": min(value for value in retained_energy.values() if value is not None),
    }
    return ranks, report


def build_isvd_bank(
    source_groups: Sequence[Mapping[str, AdapterModule]],
    source_names: Sequence[str],
    output_dir: str | Path,
    budget_parameters: int,
    source_metadata: Sequence[dict],
    save_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    from safetensors.torch import save_file

    if len(source_groups) != len(source_names) or len(source_groups) != len(source_metadata):
        raise ValueError("source_groups/source_names/source_metadata length mismatch")
    names = ensure_same_modules(source_groups)
    ranks, allocation_report = allocate_isvd_ranks(source_groups, int(budget_parameters))
    tensors: dict[str, torch.Tensor] = {}
    modules = []
    for module_index, name in enumerate(names):
        max_rank = max(ranks[(expert_index, name)] for expert_index in range(len(source_groups)))
        in_features = source_groups[0][name].in_features
        out_features = source_groups[0][name].out_features
        v_pad = torch.zeros((len(source_groups), in_features, max_rank), dtype=save_dtype)
        b_pad = torch.zeros((len(source_groups), out_features, max_rank), dtype=save_dtype)
        rank_list = []
        energy_list = []
        for expert_index, group in enumerate(source_groups):
            module = group[name]
            rank = ranks[(expert_index, name)]
            u, singular, v = _thin_product_svd(module)
            rank_list.append(rank)
            total_energy = float(torch.sum(singular.square()).item())
            kept_energy = float(torch.sum(singular[:rank].square()).item()) if rank else 0.0
            energy_list.append(None if total_energy <= 0 else kept_energy / total_energy)
            if rank:
                b_factor = u[:, :rank] * singular[:rank].unsqueeze(0)
                v_factor = v[:, :rank]
                b_pad[expert_index, :, :rank] = b_factor.to(dtype=save_dtype)
                v_pad[expert_index, :, :rank] = v_factor.to(dtype=save_dtype)
        prefix = f"m{module_index:03d}"
        tensors[f"{prefix}.B"] = b_pad.contiguous()
        tensors[f"{prefix}.V"] = v_pad.contiguous()
        modules.append({
            "name": name,
            "tensor_prefix": prefix,
            "in_features": int(in_features),
            "out_features": int(out_features),
            "expert_ranks": rank_list,
            "max_rank": int(max_rank),
            "retained_energy_by_expert": energy_list,
        })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = output_dir / "isvd_bank.safetensors"
    save_file(tensors, str(tensor_path), metadata={"format": "coreflow-independent-svd-v1", "schema_version": "1"})
    config = {
        "format": "coreflow-independent-svd-v1",
        "schema_version": 1,
        "expert_order": list(source_names),
        "experts": [
            {
                "expert_index": index,
                "name": name,
                **source_metadata[index],
            }
            for index, name in enumerate(source_names)
        ],
        "modules": modules,
        "budget": allocation_report,
        "isvd_bank_sha256": sha256_file(tensor_path),
    }
    write_json(output_dir / "isvd_config.json", config)
    config["isvd_config_sha256"] = sha256_file(output_dir / "isvd_config.json")
    write_json(output_dir / "isvd_config.json", config)
    return config
