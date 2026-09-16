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
from coreflow.io import adapter_checkpoint_path, load_jsonl, sha256_file

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
    isvd_root = Path(args.isvd_asset_root).resolve()
    expected_isvd = load_json(ROOT / "provenance" / "EXPECTED_ISVD_LOCK.json")
    isvd_checked = {}
    for pool, item in sorted(expected_isvd["banks"].items()):
        bank = isvd_root / pool / "matched_q185_bf16" / "isvd_bank.safetensors"
        if sha256_file(bank) != item["isvd_bank_sha256"]:
            raise ValueError(f"ISVD bank hash mismatch: {pool}")
        isvd_checked[pool] = item["isvd_bank_sha256"]
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
        "isvd_asset_root": str(isvd_root),
        "config_sha256": sha256_file(args.config),
        "package_manifest_sha256": sha256_file(ROOT / "PACKAGE_MANIFEST.sha256") if (ROOT / "PACKAGE_MANIFEST.sha256").exists() else None,
        "source_adapters": source_adapters,
        "gates_checked": len(gates),
        "core_banks_checked": len(banks),
        "isvd_banks_checked": isvd_checked,
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "gpu": runtime["device"], "gates": len(gates), "banks": len(banks), "isvd": len(isvd_checked)}, ensure_ascii=False))


def load_raw_classes() -> dict[str, dict]:
    raw = load_json(ROOT / "raw_sources" / "ClassEval_data.json")
    return {str(cls["task_id"]): cls for cls in raw}


def validate_classeval_rows(rows: list[dict], split: str) -> None:
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split} duplicate task_id")
    classes: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        required = {
            "task_id", "class_id", "class_name", "split", "entry_point", "method_index",
            "method_order", "problem", "prompt_sha256", "evaluator", "import_statement",
            "class_constructor", "method_description", "test_class", "test_code", "dependencies",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{split} row {index} missing {missing}")
        if row["evaluator"] != "classeval_method":
            raise ValueError(f"{split} row {index} evaluator is not classeval_method")
        if not isinstance(row["entry_point"], str) or not row["entry_point"].isidentifier():
            raise ValueError(f"{split} row {index} invalid entry_point")
        if row["split"] != split:
            raise ValueError(f"{split} row {index} split field mismatch")
        if not isinstance(row["test_code"], str) or not row["test_code"].strip():
            raise ValueError(f"{split} row {index} missing test_code")
        if not isinstance(row["class_constructor"], str) or not row["class_constructor"].strip():
            raise ValueError(f"{split} row {index} missing class_constructor")
        classes.setdefault(str(row["class_id"]), []).append(row)
    for class_id, class_rows in classes.items():
        first = class_rows[0]
        for row in class_rows[1:]:
            if row["method_order"] != first["method_order"]:
                raise ValueError(f"{class_id} inconsistent method_order")
            if row["class_constructor"] != first["class_constructor"]:
                raise ValueError(f"{class_id} inconsistent class_constructor")
            if row["import_statement"] != first["import_statement"]:
                raise ValueError(f"{class_id} inconsistent import_statement")
        entry_points = [row["entry_point"] for row in class_rows]
        if sorted(entry_points) != sorted(first["method_order"]):
            raise ValueError(f"{class_id} entry_point set does not match method_order")
        if len(set(entry_points)) != len(entry_points):
            raise ValueError(f"{class_id} duplicate entry_point")


def leak_scan(rows: list[dict], raw_classes: dict[str, dict]) -> None:
    for row in rows:
        problem = row["problem"]
        cls = raw_classes[str(row["class_id"])]
        forbidden = [cls.get("solution_code", ""), cls.get("test", "")]
        for method in cls["methods_info"]:
            forbidden.append(method.get("solution_code", ""))
            forbidden.append(method.get("test_code", ""))
        hits = [text[:120] for text in forbidden if text and text in problem]
        if hits:
            raise ValueError(f"Prompt leak at {row['task_id']}: {hits[:2]}")


def prepare(args, cfg: dict) -> None:
    work = Path(args.work_root).resolve()
    data_root = work / "data" / cfg.get("paths", {}).get("data_subdir", "indep_confirm_classeval")
    split_report = {}
    all_task_ids = {}
    raw_classes = load_raw_classes()
    split_names = ("qualification", "formal", "reserve")
    for name in split_names:
        spec = cfg["data"][name]
        source = ROOT / "data" / spec["file"]
        target = data_root / spec["file"]
        copy_locked(source, target, spec["sha256"])
        rows = load_jsonl(target)
        if len(rows) != int(spec["rows"]):
            raise ValueError(f"{name} row count mismatch")
        validate_classeval_rows(rows, spec["split_role"])
        all_task_ids[name] = {row["task_id"] for row in rows}
        split_report[name] = {"path": str(target), "rows": len(rows), "sha256": sha256_file(target)}
    names = list(split_names)
    for index in range(len(names)):
        for other in range(index + 1, len(names)):
            if all_task_ids[names[index]] & all_task_ids[names[other]]:
                raise ValueError(f"Split overlap: {names[index]} / {names[other]}")
    for name in names:
        leak_scan(load_jsonl(data_root / cfg["data"][name]["file"]), raw_classes)
    prompt = ROOT / "data" / cfg["prompts"]["code_file"]
    if sha256_file(prompt) != cfg["prompts"]["code_sha256"]:
        raise ValueError("code prompt hash mismatch")
    text = prompt.read_text(encoding="utf-8")
    if text.count("{problem}") != 1:
        raise ValueError("code prompt must contain exactly one {problem}")
    if "{entry_point}" in text:
        raise ValueError("ClassEval prompt must not require {entry_point}")
    manifest = ROOT / "data" / cfg["data"]["manifest"]["file"]
    if sha256_file(manifest) != cfg["data"]["manifest"]["sha256"]:
        raise ValueError("split manifest hash mismatch")
    payload = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "splits": split_report,
        "prompt_checks": {"code_file": cfg["prompts"]["code_file"], "code_sha256": cfg["prompts"]["code_sha256"]},
        "manifest_sha256": cfg["data"]["manifest"]["sha256"],
        "leak_scan": "solution/test strings not found in prompts",
        "formal_candidate_generation_allowed": bool(cfg.get("execution_contract", {}).get("formal_candidate_generation_allowed_after_seal", False)),
    }
    write_json(args.output, payload)
    print(json.dumps({"status": "PASS", "splits": {k: v["rows"] for k, v in split_report.items()}}, ensure_ascii=False))


def isvd_bank_projection(banks: dict) -> dict:
    return {
        pool: {
            "budget_parameters": item.get("budget_parameters"),
            "isvd_bank_sha256": item.get("isvd_bank_sha256"),
            "isvd_config_sha256": item.get("isvd_config_sha256"),
        }
        for pool, item in sorted((banks or {}).items())
    }


def check_isvd(args, cfg: dict) -> None:
    expected = load_json(ROOT / "provenance" / "EXPECTED_ISVD_LOCK.json")
    if args.isvd_lock:
        primary = load_json(args.isvd_lock)
        if canonical(isvd_bank_projection(primary.get("banks", {}))) != canonical(isvd_bank_projection(expected["banks"])):
            raise ValueError("Primary isvd_asset_lock differs from EXPECTED_ISVD_LOCK")
    isvd_root = Path(args.isvd_asset_root).resolve()
    banks = {}
    for pool, item in sorted(expected["banks"].items()):
        bank = isvd_root / pool / "matched_q185_bf16" / "isvd_bank.safetensors"
        if sha256_file(bank) != item["isvd_bank_sha256"]:
            raise ValueError(f"ISVD bank hash mismatch: {pool}")
        banks[pool] = {
            "path": str(bank),
            "isvd_bank_sha256": item["isvd_bank_sha256"],
            "budget_parameters": item["budget_parameters"],
        }
    lock = {
        "status": "PASS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "banks": banks,
        "source_primary_lock": str(args.isvd_lock) if args.isvd_lock else None,
    }
    write_json(args.output, lock)
    print(json.dumps({"status": "PASS", "banks": sorted(lock["banks"])}, ensure_ascii=False))


def seal(args, cfg: dict) -> None:
    prerequisites = {
        "preflight": args.preflight,
        "data_audit": args.data_audit,
        "isvd_asset_lock": args.isvd_asset_lock,
        "evaluator_seal": args.evaluator_seal,
    }
    for label, path in prerequisites.items():
        payload = load_json(path)
        if payload.get("status") not in {"PASS", "ANALYZED"}:
            raise ValueError(f"{label} is not complete: {payload.get('status')}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Protocol already sealed: {output}")
    payload = {
        "status": "SEALED_BEFORE_FORMAL_MODEL_OUTPUTS",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": utc_now(),
        "formal_candidate_generation_allowed": bool(cfg.get("execution_contract", {}).get("formal_candidate_generation_allowed_after_seal", False)),
        "config_sha256": sha256_file(args.config),
        "prerequisites": {label: sha256_file(path) for label, path in prerequisites.items()},
    }
    write_json(output, payload)
    print(json.dumps({"status": "PASS", "seal": str(output)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "prepare", "check-isvd", "seal"))
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--source-work-root")
    parser.add_argument("--model")
    parser.add_argument("--m2-asset-root")
    parser.add_argument("--m2-asset-lock")
    parser.add_argument("--isvd-asset-root")
    parser.add_argument("--isvd-lock")
    parser.add_argument("--isvd-asset-lock")
    parser.add_argument("--config", default=str(ROOT / "config" / "indep_confirm_classeval_protocol.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight")
    parser.add_argument("--data-audit")
    parser.add_argument("--evaluator-seal")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.action == "preflight":
        preflight(args, cfg)
    elif args.action == "prepare":
        prepare(args, cfg)
    elif args.action == "check-isvd":
        check_isvd(args, cfg)
    elif args.action == "seal":
        seal(args, cfg)


if __name__ == "__main__":
    main()
