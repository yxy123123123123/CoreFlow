#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from common import ROOT, config, load_json, now_utc, paths, sha256_file, variant_dir, write_json


def checkpoint(root: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter checkpoint under {root}")


def main() -> None:
    cfg, p = config(), paths()
    p["reports"].mkdir(parents=True, exist_ok=True)
    p["results"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(p["source"] / "runtime_env" / "bin" / "python"), "-c",
         "import torch,safetensors; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.cuda.device_count())"],
        text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Dependency/GPU check failed: {completed.stderr[-2000:]}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines or "4090" not in lines[0] or int(lines[-1]) != 1:
        raise RuntimeError(f"PRS-5 requires exactly one visible RTX 4090; observed={lines}")
    source_rows = []
    for name in cfg["expert_order"]:
        item = cfg["experts"][name]
        root = p["asset_root"] / "LoRAs" / item["relative"]
        config_path, checkpoint_path = root / "adapter_config.json", checkpoint(root)
        observed_config, observed_checkpoint = sha256_file(config_path), sha256_file(checkpoint_path)
        if observed_config != item["adapter_config_sha256"] or observed_checkpoint != item["checkpoint_sha256"]:
            raise ValueError(f"Source LoRA hash mismatch: {name}")
        source_rows.append({"expert": name, "adapter_config": str(config_path), "adapter_config_sha256": observed_config,
                            "checkpoint": str(checkpoint_path), "checkpoint_sha256": observed_checkpoint})
    bank_map_path = p["phase2b"] / "reports" / "phase2b" / "bank_map.json"
    if not bank_map_path.is_file():
        raise FileNotFoundError(f"Phase-2B bank map missing: {bank_map_path}")
    embedded_bank_map = ROOT / "evidence" / "phase2b_bank_map.json"
    if sha256_file(bank_map_path) != sha256_file(embedded_bank_map):
        raise ValueError("Phase-2B bank map differs from the locally frozen evidence copy")
    bank_map = load_json(bank_map_path)
    map_hashes = {name: item.get("sha256") for name, item in bank_map.get("ablation", {}).items()}
    bank_rows = []
    for name, item in cfg["variants"].items():
        root = variant_dir(item, p)
        tensor_path, config_path = root / item["tensor"], root / item["config"]
        if not tensor_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"PRS-5 bank incomplete for {name}: {root}")
        observed = sha256_file(tensor_path)
        expected = item.get("tensor_sha256") or map_hashes.get(name)
        if not expected:
            raise ValueError(f"No frozen bank hash for {name}")
        if observed != expected:
            raise ValueError(f"Bank hash mismatch for {name}: {observed}")
        bank_rows.append({"variant": name, "root": str(root), "kind": item["kind"],
                          "tensor_sha256": observed, "config_sha256": sha256_file(config_path),
                          "tensor_bytes": tensor_path.stat().st_size, "config_bytes": config_path.stat().st_size})
    forbidden = [path for path in ROOT.rglob("*") if path.is_file() and
                 ("jsonl" in path.suffix.lower() or "generate" in path.name.lower() or "serve" in path.name.lower())]
    if forbidden:
        raise RuntimeError(f"Task data/generation surface is forbidden in PRS-5: {forbidden}")
    evidence_rows = []
    for path in sorted((ROOT / "evidence").glob("*")):
        if path.is_file():
            evidence_rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    payload = {"status": "PASS", "protocol_id": cfg["protocol_id"], "timestamp_utc": now_utc(),
               "gpu": lines[0], "visible_gpu_count": int(lines[-1]), "sources": source_rows,
               "banks": bank_rows, "phase2b_bank_map_sha256": sha256_file(bank_map_path),
               "evidence": evidence_rows, "task_data_present": False, "generation_surface_present": False}
    write_json(p["reports"] / "preflight.json", payload)
    print(json.dumps({"status": "PASS", "gpu": lines[0], "experts": len(source_rows), "variants": len(bank_rows)}))


if __name__ == "__main__":
    main()
