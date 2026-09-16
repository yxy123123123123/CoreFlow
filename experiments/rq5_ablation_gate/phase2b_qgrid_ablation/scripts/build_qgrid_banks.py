#!/usr/bin/env python3
"""Phase-2B: build core banks for B1 q-grid {128,160,185,224,256} and K=5."""
from __future__ import annotations
import hashlib, json, sys, argparse, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

from coreflow.decompose import build_banks
from coreflow.gate_io import load_json, write_json
from coreflow.io import load_adapter, sha256_file, adapter_checkpoint_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-assets", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    cfg = load_json(ROOT / "config" / "protocol.json")
    order = cfg["expert_order"]
    qv = cfg["b1_qgrid"]["q_values"]
    asset_root = Path(args.source_assets) / "official_assets"

    groups = []
    metadata = []
    for name in order:
        adapter_dir = asset_root / "LoRAs" / f"{name}_lora"
        adapter_cfg, modules = load_adapter(adapter_dir)
        groups.append(modules)
        metadata.append({
            "name": name,
            "adapter_sha256": sha256_file(adapter_checkpoint_path(adapter_dir)),
            "rank": int(adapter_cfg["r"]),
        })

    out_root = Path(args.output_root)
    results = {}
    q185_sha = "1a3eaf175981a2c914af8400490ee603308851fd5ca22279a6ac76c2d121eaf2"
    for q in qv:
        qdir = out_root / f"q{q}_bf16"
        if q == 185 and qdir.exists():
            existing = load_json(qdir / "core_config.json") if (qdir / "core_config.json").exists() else {}
            if existing.get("tensor_sha256") == q185_sha:
                results[f"q{q}"] = {"status": "REUSE_EXISTING_FORMAL_BANK", "sha256": q185_sha}
                continue
        outputs = build_banks(
            groups, order, [q], str(qdir),
            save_dtype=torch.bfloat16, source_metadata=metadata, bank_label_suffix="_bf16",
        )
        bank_cfg = load_json(qdir / "core_config.json")
        results[f"q{q}"] = {
            "status": "BUILT",
            "sha256": sha256_file(qdir / "core_bank.safetensors"),
            "config_sha256": bank_cfg.get("tensor_sha256", "N/A"),
        }
        print(json.dumps({"q": q, **results[f"q{q}"]}, ensure_ascii=False))

    write_json(out_root / "build_report.json", {"status": "PASS", "q_values": qv, "results": results})
    print(json.dumps({"status": "PASS"}, ensure_ascii=False))

if __name__ == "__main__":
    main()