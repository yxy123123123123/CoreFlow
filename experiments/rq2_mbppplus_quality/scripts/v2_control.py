#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import adapter_checkpoint_path, load_adapter, load_jsonl, sha256_file
from coreflow.isvd import build_isvd_bank
ADAPTERS = ("zh", "ru", "es", "math", "code")


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
    work = Path(args.work_root).resolve()
    source = Path(args.source_work_root).resolve()
    model = Path(args.model).resolve()
    asset_root = Path(args.m2_asset_root).resolve()
    asset_lock_path = Path(args.m2_asset_lock).resolve()
    expected_lock = load_json(ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json")
    installed_lock = load_json(asset_lock_path)
    if canonical(expected_lock) != canonical(installed_lock):
        raise ValueError("Installed m2 asset lock differs from expected formal-v1 lock")
    required = [
        source / "runtime_env" / "bin" / "python",
        source / "vendor" / "LoRAFlow" / "UltraEval" / "main.py",
        model / "config.json",
        asset_root / "gates",
        asset_root / "core_banks",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required runtime/assets missing: {missing}")
    if sha256_file(model / "config.json") != cfg["model_config_sha256"]:
        raise ValueError("Base model config hash mismatch")
    gates = {}
    for filename, expected_hash in sorted(expected_lock["gates"].items()):
        gate = asset_root / "gates" / filename
        if sha256_file(gate) != expected_hash:
            raise ValueError(f"Gate hash mismatch: {filename}")
        report = load_json(gate.with_suffix(".json"))
        if report.get("status") != "PASS" or report.get("training_stability") != "CLEAN":
            raise ValueError(f"Gate is not PASS/CLEAN: {filename}")
        gates[filename] = expected_hash
    banks = {}
    for pool, q_items in sorted(expected_lock["banks"].items()):
        for q, expected in sorted(q_items.items()):
            bank = asset_root / "core_banks" / pool / f"q{q}_bf16" / "core_bank.safetensors"
            if sha256_file(bank) != expected["core_bank_sha256"]:
                raise ValueError(f"Core bank hash mismatch: {pool}/q{q}")
            banks[f"{pool}/q{q}"] = expected["core_bank_sha256"]
    source_adapters = {}
    for name in ADAPTERS:
        adapter_dir = source / "official_assets" / "LoRAs" / f"{name}_lora"
        config_file = adapter_dir / "adapter_config.json"
        checkpoint = adapter_checkpoint_path(adapter_dir)
        source_adapters[name] = {
            "adapter_config_sha256": sha256_file(config_file),
            "checkpoint_sha256": sha256_file(checkpoint),
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
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "host_python": platform.python_version(),
        "runtime": runtime,
        "work_root": str(work),
        "source_work_root": str(source),
        "model": str(model),
        "m2_asset_root": str(asset_root),
        "m2_asset_lock": str(asset_lock_path),
        "config_sha256": sha256_file(args.config),
        "package_manifest_sha256": sha256_file(ROOT / "PACKAGE_MANIFEST.sha256") if (ROOT / "PACKAGE_MANIFEST.sha256").exists() else None,
        "source_adapters": source_adapters,
        "gates_checked": len(gates),
        "core_banks_checked": len(banks),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "gpu": runtime["device"], "gates": len(gates), "banks": len(banks)}, ensure_ascii=False))


def prepare(args, cfg: dict) -> None:
    work = Path(args.work_root).resolve()
    data_root = work / "data" / cfg.get("paths", {}).get("data_subdir", "formal_v2_f0r")
    split_report = {}
    all_task_ids = {}
    for name, spec in cfg["data"].items():
        source = ROOT / "data" / spec["file"]
        target = data_root / spec["file"]
        copy_locked(source, target, spec["sha256"])
        rows = load_jsonl(target)
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"{name} row count mismatch")
        ids = [row["task_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name} duplicate task_id")
        all_task_ids[name] = set(ids)
        if name.startswith("mbppplus"):
            for index, row in enumerate(rows):
                for field in ("problem", "entry_point", "canonical_solution", "hidden_test", "test_list"):
                    if field not in row:
                        raise ValueError(f"{name} row {index} missing {field}")
                if not isinstance(row["entry_point"], str) or not row["entry_point"].isidentifier():
                    raise ValueError(f"{name} row {index} has invalid entry_point")
                prompt_text = row["problem"]
                forbidden = [row["canonical_solution"], row["hidden_test"]]
                if any(item and item in prompt_text for item in forbidden):
                    raise ValueError(f"MBPP+ prompt leak at {row['task_id']}")
        split_report[name] = {"path": str(target), "rows": len(rows), "sha256": sha256_file(target)}
    if all_task_ids["mbppplus_dev128"] & all_task_ids["mbppplus_formal_candidate250"]:
        raise ValueError("MBPP+ dev/formal_candidate overlap")
    prompt_report = {}
    for task in ("code", "math"):
        prompt = ROOT / "data" / cfg["prompts"][f"{task}_file"]
        if sha256_file(prompt) != cfg["prompts"][f"{task}_sha256"]:
            raise ValueError(f"{task} prompt hash mismatch")
        if prompt.read_text(encoding="utf-8").count("{problem}") != 1:
            raise ValueError(f"{task} prompt must contain exactly one placeholder")
        if task == "code" and prompt.read_text(encoding="utf-8").count("{entry_point}") != 1:
            raise ValueError("code prompt must contain exactly one {entry_point} placeholder")
        prompt_report[task] = cfg["prompts"][f"{task}_sha256"]
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "splits": split_report,
        "prompt_checks": prompt_report,
        "mbppplus_manifest_sha256": sha256_file(ROOT / "provenance" / "MBPPPLUS_SOURCE_MANIFEST.json"),
        "formal_candidate_generation_allowed": bool(cfg.get("execution_contract", {}).get("formal_candidate_generation_allowed_after_seal", False)),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "splits": {k: v["rows"] for k, v in split_report.items()}}, ensure_ascii=False))


def build_isvd(args, cfg: dict) -> None:
    work = Path(args.work_root).resolve()
    source = Path(args.source_work_root).resolve()
    asset_lock = load_json(args.m2_asset_lock)
    bank_root = work / "isvd_banks"
    lock = {"status": "PASS", "protocol_id": cfg["protocol_id"], "timestamp_utc": utc_now(), "banks": {}}
    adapter_cache = {}
    metadata_cache = {}
    for name in ADAPTERS:
        adapter_dir = source / "official_assets" / "LoRAs" / f"{name}_lora"
        config, modules = load_adapter(adapter_dir)
        checkpoint = adapter_checkpoint_path(adapter_dir)
        adapter_cache[name] = modules
        metadata_cache[name] = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
            "rank": int(config["r"]),
            "lora_alpha": config.get("lora_alpha"),
        }
    built_by_signature: dict[tuple, Path] = {}
    for pool in ("k5_math", "k5_code"):
        order = cfg["expert_pools"][pool]
        budget = int(asset_lock["banks"][pool]["185"]["efficiency"]["coreflow_resident_adapter_parameters"])
        output_dir = bank_root / pool / "matched_q185_bf16"
        signature = (tuple(order), budget)
        if (output_dir / "isvd_config.json").exists():
            config = load_json(output_dir / "isvd_config.json")
        elif signature in built_by_signature:
            shutil.copytree(built_by_signature[signature], output_dir)
            config = load_json(output_dir / "isvd_config.json")
        else:
            config = build_isvd_bank(
                [adapter_cache[name] for name in order],
                order,
                output_dir,
                budget,
                [metadata_cache[name] for name in order],
            )
            built_by_signature[signature] = output_dir
        lock["banks"][pool] = {
            "path": str(output_dir),
            "budget_parameters": budget,
            "isvd_bank_sha256": sha256_file(output_dir / "isvd_bank.safetensors"),
            "isvd_config_sha256": sha256_file(output_dir / "isvd_config.json"),
            "budget": load_json(output_dir / "isvd_config.json")["budget"],
        }
    write_json(args.output, lock)
    print(json.dumps({"status": "PASS", "banks": sorted(lock["banks"])}, ensure_ascii=False))


def analyze(args, cfg: dict) -> None:
    root = Path(args.result_root)
    def load_summary(method: str) -> dict:
        return load_json(root / "go_nogo_quality" / method / "summary.json")
    methods = cfg["phase_orders"]["go_nogo_quality"]
    summaries = {method: load_summary(method) for method in methods}
    mbpp_full = summaries["code_full_k5_seed41"]
    correct = int(mbpp_full["correct"])
    if correct >= cfg["hypotheses"]["mbppplus_dev_floor_confirmatory_min_correct"]:
        code_floor = "CODE_CONFIRMATORY"
    elif correct >= cfg["hypotheses"]["mbppplus_dev_floor_supportive_min_correct"]:
        code_floor = "CODE_SUPPORTIVE"
    else:
        code_floor = "CODE_STRESS_ONLY"
    comparisons = {}
    for task in ("math", "code"):
        full = summaries[f"{task}_full_k5_seed41"]
        for candidate in (f"{task}_core_q185_k5_seed41", f"{task}_isvd_q185_k5_seed41"):
            item = summaries[candidate]
            comparisons[candidate] = {
                "full_correct": full["correct"],
                "candidate_correct": item["correct"],
                "rows": item["rows"],
                "diff_pp": 100.0 * (item["pass_at_1"] - full["pass_at_1"]),
                "retention": item["pass_at_1"] / full["pass_at_1"] if full["pass_at_1"] else None,
            }
    decision = "PROCEED_STRONG" if code_floor == "CODE_CONFIRMATORY" else "PROCEED_QUALIFIED" if code_floor == "CODE_SUPPORTIVE" else "STOP_CODE_STRESS_ONLY"
    payload = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "decision": decision,
        "mbppplus_dev_floor": code_floor,
        "mbppplus_full_correct": correct,
        "summaries": summaries,
        "comparisons": comparisons,
        "formal_candidate_opened": False,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "ANALYZED", "decision": decision, "mbppplus_floor": code_floor, "full_correct": correct}, ensure_ascii=False))


def seal(args, cfg: dict) -> None:
    for label, path in {
        "preflight": args.preflight,
        "data_audit": args.data_audit,
        "isvd_asset_lock": args.isvd_asset_lock,
        "e0_correctness": args.e0_correctness,
    }.items():
        payload = load_json(path)
        if payload.get("status") not in {"PASS", "ANALYZED"}:
            raise ValueError(f"{label} is not complete: {payload.get('status')}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Protocol already sealed: {output}")
    payload = {
        "status": (
            "SEALED_BEFORE_FORMAL_MODEL_OUTPUTS"
            if cfg.get("execution_contract", {}).get("formal_candidate_generation_allowed_after_seal", False)
            else "SEALED_BEFORE_DEV_MODEL_OUTPUTS"
        ),
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "formal_candidate_generation_allowed": bool(cfg.get("execution_contract", {}).get("formal_candidate_generation_allowed_after_seal", False)),
        "config_sha256": sha256_file(args.config),
        "prerequisites": {
            "preflight": sha256_file(args.preflight),
            "data_audit": sha256_file(args.data_audit),
            "isvd_asset_lock": sha256_file(args.isvd_asset_lock),
            "e0_correctness": sha256_file(args.e0_correctness),
        },
    }
    write_json(output, payload)
    print(json.dumps({"status": "PASS", "seal": str(output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "build-isvd", "seal", "analyze"))
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--source-work-root")
    parser.add_argument("--model")
    parser.add_argument("--m2-asset-root")
    parser.add_argument("--m2-asset-lock")
    parser.add_argument("--config", default=str(ROOT / "config" / "formal_v2_f0r_protocol.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--data-audit")
    parser.add_argument("--isvd-asset-lock")
    parser.add_argument("--e0-correctness")
    parser.add_argument("--result-root")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action == "preflight":
        preflight(args, cfg)
    elif args.action == "prepare":
        prepare(args, cfg)
    elif args.action == "build-isvd":
        build_isvd(args, cfg)
    elif args.action == "seal":
        seal(args, cfg)
    elif args.action == "analyze":
        analyze(args, cfg)


if __name__ == "__main__":
    main()
