#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.io import adapter_checkpoint_path
from scripts.common import (
    config,
    core_bank_path,
    gate_path,
    isvd_bank_path,
    load_jsonl,
    now_utc,
    paths,
    sha256_file,
    write_json,
)


def main() -> None:
    cfg, p = config(), paths()
    p["reports"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)
    if (p["reports"] / "PILOT_SEAL.json").exists():
        raise RuntimeError("Pilot is already sealed; do not rerun preflight in this workspace")

    forbidden_data = [
        item.relative_to(ROOT).as_posix()
        for item in ROOT.rglob("*")
        if item.is_file() and "formal164" in item.name.lower()
    ]
    if forbidden_data:
        raise RuntimeError(f"formal164 data is forbidden in this package: {forbidden_data}")

    pilot = p["data"] / cfg["data"]["pilot_file"]
    observed_pilot_hash = sha256_file(pilot)
    if observed_pilot_hash != cfg["data"]["pilot_sha256"]:
        raise ValueError(f"Pilot data hash mismatch: {observed_pilot_hash}")
    rows = load_jsonl(pilot)
    ids = [str(row["question_id"]) for row in rows]
    if len(rows) != cfg["data"]["pilot_rows"] or ids != cfg["data"]["pilot_question_ids"]:
        raise ValueError(f"Frozen pilot order mismatch: rows={len(rows)}, ids={ids}")

    required = [
        p["model"],
        p["asset_root"] / "LoRAs",
        p["official_root"],
        p["runtime_python"],
        p["m2_root"],
        p["formal_v2"],
        p["m2_lock"],
    ]
    for item in required:
        if not item.exists():
            raise FileNotFoundError(item)

    expert_hashes = {}
    for name in cfg["expert_order"]:
        spec = cfg["assets"]["experts"][name]
        root = p["asset_root"] / "LoRAs" / spec["relative"]
        checkpoint = adapter_checkpoint_path(root)
        observed = {
            "config_sha256": sha256_file(root / "adapter_config.json"),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        expected = {
            "config_sha256": spec["config_sha256"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
        }
        if observed != expected:
            raise ValueError(f"Expert hash mismatch for {name}: {observed}")
        expert_hashes[name] = observed

    gate = gate_path(cfg, p)
    core = core_bank_path(cfg, p) / "core_bank.safetensors"
    isvd = isvd_bank_path(cfg, p) / "isvd_bank.safetensors"
    checks = [
        (gate, cfg["assets"]["gate_sha256"], "gate"),
        (core, cfg["assets"]["core_bank_sha256"], "core bank"),
        (isvd, cfg["assets"]["isvd_bank_sha256"], "ISVD bank"),
    ]
    frozen_assets = {}
    for path, expected, label in checks:
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"{label} hash mismatch: {observed}")
        frozen_assets[label] = {"path": str(path), "sha256": observed}

    import os
    dependency_paths = [
        ROOT / "vendor_livecodebench",
        p["official_root"] / "UltraEval" / "lora-fusion" / "transformers" / "src",
        p["official_root"] / "UltraEval" / "lora-fusion" / "peft-group_lora" / "src",
        p["official_root"] / "UltraEval" / "human-eval-master",
        p["official_root"] / "UltraEval",
    ]
    dependency_pythonpath = os.pathsep.join(str(item) for item in dependency_paths)
    if os.environ.get("PYTHONPATH"):
        dependency_pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    dependency = subprocess.run(
        [
            str(p["runtime_python"]),
            "-c",
            "import torch,transformers,peft,flask,safetensors,numpy; "
            "from testing_util import run_test",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": dependency_pythonpath},
    )
    if dependency.returncode:
        raise RuntimeError(f"Dependency check failed: {dependency.stderr[-2000:]}")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != cfg["hardware"]["required_visible_gpu_count"]:
        raise RuntimeError("Exactly one visible CUDA GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if cfg["hardware"]["required_gpu_substring"] not in gpu:
        raise RuntimeError(f"Unexpected GPU: {gpu}")

    prior_outputs = list((p["results"] / "pilot_b").glob("*")) if (p["results"] / "pilot_b").exists() else []
    if prior_outputs:
        raise RuntimeError(f"Pilot outputs exist before seal: {[str(x) for x in prior_outputs]}")

    report = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "gpu": gpu,
        "pilot": {"rows": len(rows), "question_ids": ids, "sha256": observed_pilot_hash},
        "qualification64_source_sha256": cfg["data"]["qualification64_sha256"],
        "formal164_embedded": False,
        "formal164_runtime_path_resolved": False,
        "experts": expert_hashes,
        "assets": frozen_assets,
        "legacy_raw_workspace_present": p["legacy"].exists(),
        "python": sys.version,
    }
    write_json(p["reports"] / "preflight.json", report)
    print(json.dumps({"status": "PASS", "gpu": gpu, "pilot_rows": len(rows), "formal164_embedded": False}))


if __name__ == "__main__":
    main()
