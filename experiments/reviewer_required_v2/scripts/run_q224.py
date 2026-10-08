from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def run_one(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[START] " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"q224 method failed with exit={result.returncode}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mbpp", "classeval", "both"], default="both")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = path("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_required_v2_workspace")
    source = path("SOURCE_WORK_ROOT", "/root/autodl-tmp/coreflow_m0")
    model = path("MODEL", "/root/autodl-tmp/Model")
    formal_work = path("FORMAL_V2_WORK_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    formal_pkg = path("FORMAL_V2_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    classeval_pkg = path("CLASSEVAL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_upload")
    asset_lock = formal_work / "reports" / "m2_assets" / "asset_lock.json"
    report = work / "reports" / "coreflow-reviewer-required-v2"
    if not (report / "PROTOCOL_SEAL.json").exists():
        raise RuntimeError("Run seal first")
    prepare = json.loads((report / "prepare.json").read_text(encoding="utf-8"))
    data = work / "data"
    py = os.environ.get("PYTHON", sys.executable)

    if args.dataset in {"mbpp", "both"}:
        external = formal_pkg / "scripts" / "run_formal_method.py"
        config = formal_pkg / "config" / "formal_v2_primary_protocol.json"
        frozen = data / "llama_mbppplus_formal_q224.jsonl"
        for index, seed in enumerate((41, 42, 43)):
            name = f"code_core_q224_k5_seed{seed}"
            final = work / "results" / "coreflow-reviewer-required-v2" / "q224_mbpp" / name
            command = [
                py, str(external), "--formal-work-root", str(formal_work), "--source-work-root", str(source),
                "--model", str(model), "--config", str(config), "--asset-lock", str(asset_lock),
                "--frozen-data", str(frozen), "--method", name, "--output", str(final), "--port", str(6200 + index),
            ]
            run_one(command, work / "logs" / f"q224_mbpp_seed{seed}.log")

    if args.dataset in {"classeval", "both"}:
        external = classeval_pkg / "scripts" / "run_formal_method.py"
        config = classeval_pkg / "config" / "indep_confirm_classeval_protocol.json"
        frozen = data / "llama_classeval_formal_q224.jsonl"
        for index, seed in enumerate((41, 42, 43)):
            name = f"code_core_q224_k5_seed{seed}"
            final = work / "results" / "coreflow-reviewer-required-v2" / "q224_classeval" / name
            command = [
                py, str(external), "--formal-work-root", str(formal_work), "--source-work-root", str(source),
                "--model", str(model), "--config", str(config), "--asset-lock", str(asset_lock),
                "--frozen-data", str(frozen), "--method", name, "--output", str(final), "--port", str(6300 + index),
            ]
            run_one(command, work / "logs" / f"q224_classeval_seed{seed}.log")

    marker = report / "q224_complete.json"
    marker.write_text(json.dumps({"status": "PASS", "datasets": [args.dataset], "prepare_sha256": prepare["data"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage": "q224", "dataset": args.dataset}, ensure_ascii=False))


if __name__ == "__main__":
    main()
