#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.data_audit import load_jsonl
from coreflow.gate_io import load_json, write_json


ADAPTERS = ("zh", "ru", "es", "math", "code")
CHUNK = 8 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_checkpoint_path(adapter_dir: str | Path) -> Path:
    root = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No adapter checkpoint under {root}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def copy_locked(source: Path, target: Path, expected_hash: str) -> None:
    if sha256_file(source) != expected_hash:
        raise ValueError(f"Embedded data hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != expected_hash:
            raise ValueError(f"Existing frozen data differs; refusing to overwrite: {target}")
        return
    shutil.copyfile(source, target)
    if sha256_file(target) != expected_hash:
        raise ValueError(f"Copied data hash mismatch: {target}")


def preflight(args, cfg: dict) -> None:
    formal_work = Path(args.formal_work_root).resolve()
    source_work = Path(args.source_work_root).resolve()
    model = Path(args.model).resolve()
    asset_root = formal_work / "m2_assets"
    expected_lock = load_json(ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json")
    installed_lock = load_json(formal_work / "reports" / "m2_assets" / "asset_lock.json")
    if canonical(expected_lock) != canonical(installed_lock):
        raise ValueError("Installed asset_lock.json differs from the package lock")
    if sha256_file(model / "config.json") != cfg["model_config_sha256"]:
        raise ValueError("Base model config hash mismatch")

    official_root = source_work / "vendor" / "LoRAFlow"
    required = [
        source_work / "runtime_env" / "bin" / "python",
        official_root / "UltraEval" / "main.py",
        model / "config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required source runtime files are missing: {missing}")

    source_hashes = {}
    for name in ADAPTERS:
        adapter_dir = source_work / "official_assets" / "LoRAs" / f"{name}_lora"
        config_file = adapter_dir / "adapter_config.json"
        if not config_file.exists():
            raise FileNotFoundError(config_file)
        checkpoint = adapter_checkpoint_path(adapter_dir)
        source_hashes[name] = {
            "adapter_config_sha256": sha256_file(config_file),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_path": str(checkpoint),
        }

    gates = {}
    for filename, expected_hash in sorted(expected_lock["gates"].items()):
        gate_path = asset_root / "gates" / filename
        gate_report_path = gate_path.with_suffix(".json")
        if sha256_file(gate_path) != expected_hash:
            raise ValueError(f"Gate hash mismatch: {filename}")
        report = load_json(gate_report_path)
        if report.get("status") != "PASS" or report.get("training_stability") != "CLEAN":
            raise ValueError(f"Gate is not PASS/CLEAN: {filename}")
        if report.get("gate_sha256") != expected_hash:
            raise ValueError(f"Gate report hash mismatch: {filename}")
        gates[filename] = {
            "sha256": expected_hash,
            "report_sha256": sha256_file(gate_report_path),
            "training_stability": report["training_stability"],
            "state_absmax": report.get("state_absmax"),
        }

    banks = {}
    for pool, q_items in sorted(expected_lock["banks"].items()):
        for q, expected in sorted(q_items.items()):
            bank_dir = asset_root / "core_banks" / pool / f"q{q}_bf16"
            bank_file = bank_dir / "core_bank.safetensors"
            config_file = bank_dir / "core_config.json"
            if sha256_file(bank_file) != expected["core_bank_sha256"]:
                raise ValueError(f"Core bank hash mismatch: {pool}/q{q}")
            metadata = load_json(config_file)
            if int(metadata.get("q", -1)) != int(q):
                raise ValueError(f"Core bank q mismatch: {pool}/q{q}")
            for expert in metadata.get("experts", []):
                recorded = expert.get("checkpoint_sha256", expert.get("adapter_sha256"))
                if recorded != source_hashes[expert["name"]]["checkpoint_sha256"]:
                    raise ValueError(f"Core bank/source adapter mismatch: {pool}/q{q}/{expert['name']}")
            banks[f"{pool}/q{q}"] = {
                "core_bank_sha256": expected["core_bank_sha256"],
                "core_config_sha256": sha256_file(config_file),
                "efficiency": expected["efficiency"],
            }

    dependency_code = (
        "import json,torch,numpy,flask,safetensors;"
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device':"
        "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}))"
    )
    completed = subprocess.run([sys.executable, "-c", dependency_code], capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Dependency check failed: {completed.stderr[-2000:]}")
    runtime = json.loads(completed.stdout.strip().splitlines()[-1])
    if not runtime["cuda_available"]:
        raise RuntimeError("CUDA is unavailable")
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "host_python": platform.python_version(),
        "runtime": runtime,
        "nvidia_smi": nvidia.stdout.strip() if nvidia.returncode == 0 else None,
        "formal_work_root": str(formal_work),
        "source_work_root": str(source_work),
        "model": str(model),
        "model_config_sha256": cfg["model_config_sha256"],
        "source_adapters": source_hashes,
        "gates": gates,
        "banks": banks,
        "config_sha256": sha256_file(args.config),
        "package_manifest_sha256": sha256_file(ROOT / "PACKAGE_MANIFEST.sha256"),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "gpu": runtime["device"], "gates": len(gates), "banks": len(banks)}, ensure_ascii=False))


def prepare(args, cfg: dict) -> None:
    formal_work = Path(args.formal_work_root).resolve()
    data_root = formal_work / "data" / "formal_v1"
    split_report = {}
    for name, spec in cfg["data"].items():
        source = ROOT / "data" / spec["file"]
        target = data_root / spec["file"]
        copy_locked(source, target, spec["sha256"])
        rows = load_jsonl(target)
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"{name} row count mismatch")
        task_ids = [row.get("task_id") for row in rows]
        if any(not isinstance(value, str) or not value for value in task_ids):
            raise ValueError(f"{name} has invalid task_id")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"{name} has duplicate task_id")
        if name.startswith("code"):
            required = {"task_id", "problem", "io_cases"}
        elif name == "legacy_m2_math":
            required = {"task_id", "problem", "answer"}
        else:
            required = {"task_id", "question", "golden"}
        for index, row in enumerate(rows):
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{name} row {index} missing {missing}")
        split_report[name] = {
            "path": str(target),
            "rows": len(rows),
            "sha256": sha256_file(target),
            "task_id_unique": True,
        }

    code_dev_ids = {row["task_id"] for row in load_jsonl(data_root / cfg["data"]["code_dev"]["file"])}
    code_formal_ids = {row["task_id"] for row in load_jsonl(data_root / cfg["data"]["code_formal"]["file"])}
    code_reserve_ids = {row["task_id"] for row in load_jsonl(data_root / cfg["data"]["code_reserve"]["file"])}
    if code_dev_ids & code_formal_ids or code_dev_ids & code_reserve_ids or code_formal_ids & code_reserve_ids:
        raise ValueError("Code split overlap detected")
    math_formal_ids = {row["task_id"] for row in load_jsonl(data_root / cfg["data"]["math_formal"]["file"])}
    math_reserve_ids = {row["task_id"] for row in load_jsonl(data_root / cfg["data"]["math_reserve"]["file"])}
    if math_formal_ids & math_reserve_ids:
        raise ValueError("Math split overlap detected")

    prompt_report = {}
    for task in ("code", "math"):
        prompt = ROOT / "data" / cfg["prompts"][f"{task}_file"]
        expected_hash = cfg["prompts"][f"{task}_sha256"]
        text = prompt.read_text(encoding="utf-8")
        if sha256_file(prompt) != expected_hash or text.count("{problem}") != 1:
            raise ValueError(f"{task} prompt lock/placeholder check failed")
        prompt_report[task] = {"sha256": expected_hash, "utf8": True, "placeholder_count": 1}

    local_audit = load_json(ROOT / "provenance" / "formal_data_audit.json")
    if local_audit.get("status") != "PASS":
        raise ValueError("Local formal data audit is not PASS")
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "splits": split_report,
        "code_split_intersections": 0,
        "math_formal_reserve_intersection": 0,
        "prompt_checks": prompt_report,
        "evaluator_only_fields": cfg["prompts"]["evaluator_only_fields"],
        "formal_content_human_inspection_required": False,
        "source_local_audit_sha256": sha256_file(ROOT / "provenance" / "formal_data_audit.json"),
        "dataset_lock_sha256": sha256_file(ROOT / "provenance" / "DATASET_LOCK.json"),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "splits": {key: item["rows"] for key, item in split_report.items()}}, ensure_ascii=False))


def seal(args, cfg: dict) -> None:
    required = {
        "preflight": Path(args.preflight),
        "data_audit": Path(args.data_audit),
        "legacy_pair": Path(args.legacy_pair),
        "smoke": Path(args.smoke),
    }
    for name, path in required.items():
        payload = load_json(path)
        if payload.get("status") not in {"PASS", "ANALYZED"}:
            raise ValueError(f"{name} prerequisite is not complete: {payload.get('status')}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Protocol is already sealed: {output}")
    seal_payload = {
        "status": "SEALED_BEFORE_FORMAL_OUTPUTS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "formal_results_seen_before_seal": False,
        "config_sha256": sha256_file(args.config),
        "package_manifest_sha256": sha256_file(ROOT / "PACKAGE_MANIFEST.sha256"),
        "dataset_lock_sha256": sha256_file(ROOT / "provenance" / "DATASET_LOCK.json"),
        "asset_payload_lock_sha256": sha256_file(ROOT / "provenance" / "ASSET_PAYLOAD_LOCK.json"),
        "expected_asset_lock_sha256": sha256_file(ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json"),
        "prerequisites": {name: sha256_file(path) for name, path in required.items()},
        "q224_policy": cfg["assets"]["q224_policy"],
        "code_formal_floor_correct": cfg["hypotheses"]["code_formal_full_k5_floor_correct"],
    }
    write_json(output, seal_payload)
    print(json.dumps({"status": "PASS", "seal": str(output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "seal"))
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--source-work-root")
    parser.add_argument("--model")
    parser.add_argument("--config", default=str(ROOT / "config" / "formal_protocol.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--data-audit")
    parser.add_argument("--legacy-pair")
    parser.add_argument("--smoke")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action == "preflight":
        if not args.source_work_root or not args.model:
            raise ValueError("preflight requires --source-work-root and --model")
        preflight(args, cfg)
    elif args.action == "prepare":
        prepare(args, cfg)
    else:
        if not all((args.preflight, args.data_audit, args.legacy_pair, args.smoke)):
            raise ValueError("seal requires all prerequisite paths")
        seal(args, cfg)


if __name__ == "__main__":
    main()
