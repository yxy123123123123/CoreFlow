#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.decompose import build_module_bases, project_core
from coreflow.gate_io import write_json
from coreflow.io import AdapterModule, sha256_file
from coreflow.runtime import coreflow_delta


def make_module(name: str, in_features: int, out_features: int, rank: int, scale: float, seed: int) -> AdapterModule:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(rank, in_features, generator=generator) / (in_features ** 0.5)
    b = torch.randn(out_features, rank, generator=generator) / (rank ** 0.5)
    return AdapterModule(name=name, a=a.float(), b=(b * scale).float(), scaling=1.0)


def full_delta(x: torch.Tensor, weights: torch.Tensor, modules: list[AdapterModule]) -> torch.Tensor:
    out = None
    for expert, module in enumerate(modules):
        y = ((x @ module.a.T.to(x.dtype)) @ module.b.T.to(x.dtype)) * module.scaling
        y = y * weights[..., expert].unsqueeze(-1).to(y.dtype)
        out = y if out is None else out + y
    return out


def run_case(k: int, ranks: list[int], dtype_name: str, norm_spread: bool, seed: int) -> dict:
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    in_features, out_features = 96, 80
    scales = [1.0] * k
    if norm_spread:
        scales = [10.0 ** (idx / max(k - 1, 1)) for idx in range(k)]
    modules = [make_module("synthetic", in_features, out_features, ranks[idx], scales[idx], seed + idx * 13) for idx in range(k)]
    weights = [1.0 / k for _ in range(k)]
    u, v, report = build_module_bases(modules, weights, balanced=True)
    q_left = int(u.shape[1])
    q_right = int(v.shape[1])
    cores = torch.stack([project_core(u, v, module) for module in modules], dim=0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 999)
    x = torch.randn(3, 7, in_features, generator=generator).to(dtype)
    raw_gate = torch.randn(3, 7, k, generator=generator)
    gate = torch.softmax(raw_gate, dim=-1).to(dtype)
    reference = full_delta(x, gate, modules)
    candidate = coreflow_delta(x, gate, u.to(dtype), v.to(dtype), cores.to(dtype))
    abs_error = float(torch.linalg.vector_norm((candidate.float() - reference.float())))
    rel_error = abs_error / max(float(torch.linalg.vector_norm(reference.float())), 1e-30)
    if dtype_name == "fp32":
        threshold = 1e-5
    elif dtype_name == "bf16":
        threshold = 1e-2
    else:
        threshold = 1e-3
    return {
        "k": k,
        "ranks": ranks,
        "dtype": dtype_name,
        "norm_spread": norm_spread,
        "q_left": q_left,
        "q_right": q_right,
        "relative_error": rel_error,
        "threshold": threshold,
        "passed": rel_error <= threshold,
        "basis_report": {
            "q_upper": report["q_upper"],
            "q_left_numerical": report["q_left_numerical"],
            "q_right_numerical": report["q_right_numerical"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()
    cases = []
    specs = [
        (2, [8, 16], False),
        (3, [4, 12, 20], True),
        (5, [4, 8, 12, 16, 20], True),
    ]
    for k, ranks, spread in specs:
        for dtype in ("fp32", "bf16", "fp16"):
            cases.append(run_case(k, ranks, dtype, spread, 20260728 + k))
    payload = {
        "status": "PASS" if all(item["passed"] for item in cases) else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(args.config),
        "cases": cases,
        "elapsed_seconds": time.time() - started,
        "gate_tensor_identity_scope": "synthetic full-subspace only; compressed q185/q224 quality runs do not require gate tensor equality",
        "f0r_threshold_policy": "FP32 <=1e-5, FP16 <=1e-3, BF16 <=1e-2 based on pre-seal F0 observation that BF16 full-subspace relative error is stable around 6e-3.",
    }
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "cases": len(cases)}, ensure_ascii=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
