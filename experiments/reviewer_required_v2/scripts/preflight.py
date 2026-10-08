from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def main() -> None:
    work = env_path("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")
    model = env_path("MODEL", "/root/autodl-tmp/Model")
    source = env_path("SOURCE_WORK_ROOT", "/root/autodl-tmp/coreflow_m0")
    formal_work = env_path("FORMAL_V2_WORK_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    formal_pkg = env_path("FORMAL_V2_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    classeval_pkg = env_path("CLASSEVAL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_upload")
    classeval_work = env_path("CLASSEVAL_WORK_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_workspace")
    qwen_pkg = env_path("QWEN_MATCH_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_upload")
    qwen_work = env_path("QWEN_MATCH_WORK_ROOT", "/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")

    required = {
        "model_config": model / "config.json",
        "source_official_assets": source / "official_assets",
        "source_vendor": source / "vendor" / "LoRAFlow",
        "formal_m2_assets": formal_work / "m2_assets",
        "formal_package": formal_pkg / "scripts" / "run_formal_method.py",
        "mbpp_formal": formal_pkg / "data" / "mbppplus_formal_candidate250.jsonl",
        "classeval_package": classeval_pkg / "scripts" / "run_formal_method.py",
        "classeval_formal": classeval_pkg / "data" / "classeval_formal.jsonl",
        "classeval_existing_results": classeval_work / "reports" / "indep_confirm_classeval" / "primary_decision.json",
        "qwen_package": qwen_pkg / "scripts" / "system_core_worker.py",
        "qwen_group1_results": qwen_work / "groups" / "group1",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites: " + ", ".join(f"{name}={required[name]}" for name in missing))

    q224_bank = formal_work / "m2_assets" / "core_banks" / "k5_code" / "q224_bf16" / "core_bank.safetensors"
    q224_gate = formal_work / "m2_assets" / "gates" / "code_k5_code_seed41.pt"
    if not q224_bank.exists() or not q224_gate.exists():
        raise FileNotFoundError(f"q224 assets missing: bank={q224_bank}, gate={q224_gate}")

    report = {
        "status": "PASS",
        "protocol_id": "coreflow-reviewer-minimal-phasea-v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "paths": {key: str(path) for key, path in required.items()},
        "hashes": {
            "model_config": sha256(required["model_config"]),
            "mbpp_formal": sha256(required["mbpp_formal"]),
            "classeval_formal": sha256(required["classeval_formal"]),
            "q224_bank": sha256(q224_bank),
            "q224_gate_seed41": sha256(q224_gate),
        },
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "work_root": str(work),
    }
    out = work / "reports" / "reviewer_minimal_phasea" / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "protocol_id": report["protocol_id"], "q224": True, "qwen_group1": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
