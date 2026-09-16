from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .io import AdapterModule, ensure_same_modules, sha256_file, write_json


def delta_norm_sq(module: AdapterModule, precise: bool = False) -> torch.Tensor:
    dtype = torch.float64 if precise else torch.float32
    a = module.a.to(dtype=dtype)
    b = module.b.to(dtype=dtype)
    aat = a @ a.T
    btb = b.T @ b
    return (module.scaling**2) * torch.sum(aat * btb.T)


def _psd_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Stable PSD square root for the small LoRA-rank Gram factors.

    The old implementation clamped every null direction to a positive epsilon,
    which manufactured directions that do not exist in the source update.  We
    diagonalize these small matrices in float64 and clamp round-off negatives to
    exactly zero.
    """
    original_dtype = matrix.dtype
    work = matrix.double()
    values, vectors = torch.linalg.eigh((work + work.T) * 0.5)
    values = values.clamp_min(0.0)
    root = (vectors * values.sqrt().unsqueeze(0)) @ vectors.T
    return root.to(dtype=original_dtype)


def _left_factor(module: AdapterModule, weight: float, balanced: bool) -> torch.Tensor:
    scale = module.scaling * math.sqrt(weight)
    if balanced:
        scale /= math.sqrt(max(float(delta_norm_sq(module, precise=True)), 1e-30))
    return (module.b * scale) @ _psd_sqrt(module.a @ module.a.T)


def _right_factor(module: AdapterModule, weight: float, balanced: bool) -> torch.Tensor:
    b_scaled = module.b * module.scaling
    scale = math.sqrt(weight)
    if balanced:
        scale /= math.sqrt(max(float(delta_norm_sq(module, precise=True)), 1e-30))
    return module.a.T @ _psd_sqrt(b_scaled.T @ b_scaled) * scale


def stable_orthonormal_basis(
    factor: torch.Tensor,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
    """Return an ordered thin orthonormal basis without forming F.T @ F.

    A reduced QR followed by an SVD of the small R factor is backward stable,
    avoids squaring the condition number, and keeps only numerically valid
    directions.  The returned singular values remain in descending order, so
    truncating the basis preserves the original weighted spectral objective.
    """
    if factor.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape={tuple(factor.shape)}")
    if not torch.isfinite(factor).all():
        raise ValueError("Non-finite values in basis factor")
    work = factor.float()
    q, r = torch.linalg.qr(work, mode="reduced")
    u_r, singular_values, _ = torch.linalg.svd(r, full_matrices=False)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    # A dimension-scaled FP32 cutoff (~5e-4 for Llama projections) is much too
    # aggressive for this application and deletes real low-energy LoRA
    # directions. Ten machine epsilons filters QR round-off/null directions
    # while retaining the source subspace needed by the 1e-5 Oracle contract.
    relative = float(rtol) if rtol is not None else 10.0 * torch.finfo(work.dtype).eps
    tolerance = max(float(atol), relative * largest)
    effective_rank = int((singular_values > tolerance).sum().item())
    basis_before_reorthogonalization = (q @ u_r[:, :effective_rank]).contiguous()
    if effective_rank:
        # Q and U_r are individually orthogonal, but the FP32 product Q@U_r
        # accumulates enough error at q=128/320 to violate the strict Oracle
        # projection contract. A second reduced QR preserves the selected
        # subspace while restoring an actually orthonormal stored basis.
        pre_identity = torch.eye(effective_rank, device=work.device, dtype=work.dtype)
        pre_reorthogonality_error = float(
            torch.linalg.vector_norm(
                basis_before_reorthogonalization.T @ basis_before_reorthogonalization - pre_identity
            )
        )
        basis, _ = torch.linalg.qr(basis_before_reorthogonalization, mode="reduced")
        basis = basis.contiguous()
        identity = torch.eye(effective_rank, device=basis.device, dtype=basis.dtype)
        orthogonality_error = float(torch.linalg.vector_norm(basis.T @ basis - identity))
    else:
        basis = basis_before_reorthogonalization
        pre_reorthogonality_error = 0.0
        orthogonality_error = 0.0
    diagnostics = {
        "method": "thin_qr_plus_svd",
        "factor_shape": list(work.shape),
        "tolerance": tolerance,
        "relative_tolerance": relative,
        "effective_rank": effective_rank,
        "null_direction_count": int(singular_values.numel() - effective_rank),
        "invalid_included_direction_count": 0,
        "pre_reorthogonalization_error_fro": pre_reorthogonality_error,
        "orthogonality_error_fro": orthogonality_error,
        "singular_values": singular_values.detach().cpu().tolist(),
    }
    return basis, singular_values, effective_rank, diagnostics


def orthonormal_basis(factor: torch.Tensor, eps: float | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compatibility wrapper for existing callers and tests."""
    basis, singular_values, numerical_rank, _ = stable_orthonormal_basis(
        factor,
        rtol=eps,
    )
    return basis, singular_values.square(), numerical_rank


def build_module_bases(
    sources: Sequence[AdapterModule],
    weights: Sequence[float],
    balanced: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if len(sources) != len(weights) or not sources:
        raise ValueError("sources and weights must have equal non-zero length")
    left = torch.cat([_left_factor(source, weight, balanced) for source, weight in zip(sources, weights)], dim=1)
    right = torch.cat([_right_factor(source, weight, balanced) for source, weight in zip(sources, weights)], dim=1)
    u, left_singular, q_left, left_diagnostics = stable_orthonormal_basis(left)
    v, right_singular, q_right, right_diagnostics = stable_orthonormal_basis(right)
    report = {
        "basis_method": "thin_qr_plus_svd",
        "q_left_numerical": q_left,
        "q_right_numerical": q_right,
        "q_upper": int(sum(source.rank for source in sources)),
        "left_singular_values": left_singular.detach().cpu().tolist(),
        "right_singular_values": right_singular.detach().cpu().tolist(),
        "left": left_diagnostics,
        "right": right_diagnostics,
    }
    return u, v, report


def project_core(u: torch.Tensor, v: torch.Tensor, source: AdapterModule) -> torch.Tensor:
    return source.scaling * ((u.T @ source.b) @ (source.a @ v))


def projection_relative_error(u: torch.Tensor, v: torch.Tensor, core: torch.Tensor, source: AdapterModule) -> float:
    u64 = u.double()
    v64 = v.double()
    c64 = core.double()
    dense = source.scaling * (source.b.double() @ source.a.double())
    reconstructed = u64 @ c64 @ v64.T
    residual = torch.linalg.vector_norm(reconstructed - dense)
    reference = torch.linalg.vector_norm(dense).clamp_min(1e-30)
    return float(residual / reference)


def sampled_output_relative_error(
    u: torch.Tensor,
    v: torch.Tensor,
    core: torch.Tensor,
    source: AdapterModule,
    seed: int,
    samples: int = 4,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(samples, source.in_features, generator=generator, dtype=torch.float32).to(source.a.device)
    reference = ((x @ source.a.T) @ source.b.T) * source.scaling
    candidate = ((x @ v) @ core.T) @ u.T
    return float(torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-30))


def build_banks(
    source_groups: Sequence[Mapping[str, AdapterModule]],
    source_names: Sequence[str],
    ranks: Sequence[int],
    output_root: str | Path,
    beta: float = 0.5,
    save_dtype: torch.dtype = torch.bfloat16,
    balanced: bool = True,
    device: str | torch.device = "cpu",
    source_metadata: Sequence[dict] | None = None,
    bank_label_suffix: str = "",
    schema_format: str = "coreflow-oracle-fix-v1",
) -> dict:
    from safetensors.torch import save_file

    names = ensure_same_modules(source_groups)
    if len(source_groups) < 2 or len(source_names) != len(source_groups):
        raise ValueError("source_groups and source_names must describe the same K>=2 expert pool")
    if source_metadata is not None and len(source_metadata) != len(source_names):
        raise ValueError("source_metadata must follow source_names exactly")
    if beta != 0.5:
        raise ValueError("M2 uses equal source weights; beta is retained for M0 compatibility only")
    weights = [1.0 / float(len(source_groups)) for _ in source_groups]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    per_rank_tensors: dict[int, dict[str, torch.Tensor]] = {int(rank): {} for rank in ranks}
    per_rank_modules: dict[int, list[dict]] = {int(rank): [] for rank in ranks}
    audit_modules: list[dict] = []

    for module_index, name in enumerate(names):
        sources_cpu = [group[name] for group in source_groups]
        sources = [
            AdapterModule(source.name, source.a.to(device), source.b.to(device), source.scaling)
            for source in sources_cpu
        ]
        u_full, v_full, spectral = build_module_bases(sources, weights, balanced=balanced)
        tensor_prefix = f"m{module_index:03d}"
        module_audit = {"name": name, **spectral, "source_ranks": [source.rank for source in sources]}
        audit_modules.append(module_audit)
        for rank in ranks:
            rank = int(rank)
            left_rank = min(rank, int(u_full.shape[1]))
            right_rank = min(rank, int(v_full.shape[1]))
            u = u_full[:, :left_rank].contiguous()
            v = v_full[:, :right_rank].contiguous()
            cores = torch.stack([project_core(u, v, source) for source in sources], dim=0).contiguous()
            errors = [
                sampled_output_relative_error(u, v, cores[index], source, seed=100000 + module_index * 17 + index)
                for index, source in enumerate(sources)
            ]
            per_rank_tensors[rank][f"{tensor_prefix}.U"] = u.to(dtype=save_dtype, device="cpu")
            per_rank_tensors[rank][f"{tensor_prefix}.V"] = v.to(dtype=save_dtype, device="cpu")
            per_rank_tensors[rank][f"{tensor_prefix}.C"] = cores.to(dtype=save_dtype, device="cpu")
            per_rank_modules[rank].append(
                {
                    "name": name,
                    "tensor_prefix": tensor_prefix,
                    "in_features": sources[0].in_features,
                    "out_features": sources[0].out_features,
                    "source_ranks": [source.rank for source in sources],
                    "requested_q": rank,
                    "effective_left_rank": left_rank,
                    "effective_right_rank": right_rank,
                    "truncated_left_direction_count": max(int(u_full.shape[1]) - rank, 0),
                    "truncated_right_direction_count": max(int(v_full.shape[1]) - rank, 0),
                    "invalid_included_direction_count": 0,
                    "fp32_sampled_output_relative_errors": errors,
                }
            )
        del sources, u_full, v_full

    outputs = {}
    for rank in ranks:
        rank = int(rank)
        bank_dir = output_root / f"q{rank}{bank_label_suffix}"
        bank_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = bank_dir / "core_bank.safetensors"
        save_file(
            per_rank_tensors[rank],
            str(tensor_path),
            metadata={"format": schema_format, "schema_version": "2"},
        )
        source_params = 0
        core_params = 0
        source_macs = 0
        core_macs = 0
        for module in per_rank_modules[rank]:
            din = int(module["in_features"])
            dout = int(module["out_features"])
            source_ranks = module["source_ranks"]
            source_params += sum(r * (din + dout) for r in source_ranks)
            left_rank = int(module["effective_left_rank"])
            right_rank = int(module["effective_right_rank"])
            core_params += left_rank * dout + right_rank * din + len(source_ranks) * left_rank * right_rank
            source_macs += sum(r * (din + dout) for r in source_ranks)
            core_macs += right_rank * din + len(source_ranks) * left_rank * right_rank + left_rank * dout
        config = {
            "format": schema_format,
            "schema_version": 2,
            "q": rank,
            "beta": beta,
            "balanced_trace_normalization": balanced,
            "source_order": list(source_names),
            "expert_order": list(source_names),
            "experts": list(source_metadata or [{"name": name} for name in source_names]),
            "scale_absorbed_exactly_once": True,
            "scaling_absorbed_into_core": True,
            "runtime_must_not_apply_source_scaling": True,
            "storage_dtype": str(save_dtype).replace("torch.", ""),
            "modules": per_rank_modules[rank],
            "efficiency": {
                "source_resident_adapter_parameters": source_params,
                "coreflow_resident_adapter_parameters": core_params,
                "resident_parameter_reduction_pct": 100.0 * (1.0 - core_params / source_params),
                "source_adapter_macs_per_token": source_macs,
                "coreflow_adapter_macs_per_token": core_macs,
                "adapter_macs_reduction_pct": 100.0 * (1.0 - core_macs / source_macs),
            },
        }
        write_json(bank_dir / "core_config.json", config)
        outputs[str(rank)] = {
            "directory": str(bank_dir),
            "core_bank_sha256": sha256_file(tensor_path),
            "efficiency": config["efficiency"],
        }
    write_json(output_root / "basis_audit.json", {"modules": audit_modules, "ranks": list(map(int, ranks))})
    return outputs
