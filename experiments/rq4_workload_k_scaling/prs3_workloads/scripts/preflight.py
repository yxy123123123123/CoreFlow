#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import (
    adapter_checkpoint,
    bank_path,
    config,
    expert_path,
    gate_path,
    now_utc,
    paths,
    sha256_file,
    write_json,
)


def check_file(label, path, expected):
    path = Path(path)
    if not path.is_file():
        return {"label": label, "path": str(path), "status": "MISSING", "expected_sha256": expected}
    observed = sha256_file(path)
    return {
        "label": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": observed,
        "expected_sha256": expected,
        "status": "PASS" if observed == expected else "HASH_MISMATCH",
    }


def main():
    import torch
    import flask
    import safetensors
    import transformers

    cfg, p = config(), paths()
    checks = []
    checks.append(check_file("model_config", p["model"] / "config.json", cfg["assets"]["model_config_sha256"]))
    checks.append(check_file("gate_seed41", gate_path(cfg, p), cfg["assets"]["gate_sha256"]))
    for name, item in cfg["assets"]["experts"].items():
        root = expert_path(cfg, p, name)
        checks.append(check_file(f"{name}_config", root / "adapter_config.json", item["config_sha256"]))
        try:
            checkpoint = adapter_checkpoint(root)
        except FileNotFoundError:
            checkpoint = root / "adapter_model.MISSING"
        checks.append(check_file(f"{name}_checkpoint", checkpoint, item["checkpoint_sha256"]))
    for pool, item in cfg["assets"]["banks"].items():
        for kind, stem in (("core", "core"), ("isvd", "isvd")):
            root = bank_path(cfg, p, pool, kind)
            checks.append(check_file(f"{pool}_{kind}_bank", root / f"{stem}_bank.safetensors", item[f"{kind}_bank_sha256"]))
            checks.append(check_file(f"{pool}_{kind}_config", root / f"{stem}_config.json", item[f"{kind}_config_sha256"]))
    parent_files = {
        "prs0_seal": ROOT / "evidence" / "PRS0_SEALED_BEFORE_NEW_GPU_OUTPUTS.json",
        "prs0_assets": ROOT / "evidence" / "prs0_assets.json",
        "m3_bank_lock": ROOT / "evidence" / "m3_bank_lock.json",
    }
    for label, path in parent_files.items():
        checks.append(check_file(label, path, cfg["parent_evidence"][f"{label}_sha256"]))
    errors = [item for item in checks if item["status"] != "PASS"]
    if not p["runtime_python"].is_file():
        errors.append({"label": "runtime_python", "status": "MISSING", "path": str(p["runtime_python"])})
    if not torch.cuda.is_available():
        errors.append({"label": "cuda", "status": "UNAVAILABLE"})
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        errors.append({"label": "visible_cuda_device_count", "status": "EXPECTED_1", "observed": torch.cuda.device_count()})
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "flask": getattr(flask, "__version__", "imported"),
            "safetensors": getattr(safetensors, "__version__", "imported"),
            "visible_cuda_device_count": torch.cuda.device_count(),
            "gpu0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": nvidia.stdout.strip().splitlines(),
        },
        "resolved_paths": {key: str(value) for key, value in p.items() if isinstance(value, Path)},
        "checks": checks,
        "errors": errors,
    }
    write_json(p["reports"] / "preflight.json", payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks), "errors": len(errors), "gpu": payload["environment"]["gpu0"]}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
