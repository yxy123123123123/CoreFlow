from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def p(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def run(command: list[str], env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("[START] " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"service method failed with exit={result.returncode}; see {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", default=os.environ.get("HARDWARE_LABEL", "current_gpu"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_required_v2_workspace")
    qwen_root = work / "qwen_group1"
    model = p("QWEN_MODEL_PATH", "/root/autodl-tmp/coreflow_qwen3_load_smoke_v1_upload/assets/base_model")
    comol = p("COMOL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload")
    py = os.environ.get("QWEN_PYTHON", os.environ.get("PYTHON", sys.executable))
    service_env = os.environ.copy()
    service_env.update({
        "WORK_ROOT": str(qwen_root),
        "MODEL_PATH": str(model),
        "COMOL_PACKAGE_ROOT": str(comol),
        "ACTIVE_GROUPS": "1",
        "PYTHONPATH": str(root / "coreflow") + os.pathsep + str(root / "scripts") + os.pathsep + service_env.get("PYTHONPATH", ""),
    })
    output_root = work / "results" / "coreflow-reviewer-required-v2" / "service" / args.hardware
    configs = [("c1_l512_o128", 1, 512, 128), ("c4_l512_o128", 4, 512, 128), ("c1_l1024_o128", 1, 1024, 128)]
    methods = [("full", "system_core_worker.py", ["--method", "full", "--q", "16"]), ("core_q16", "system_core_worker.py", ["--method", "core", "--q", "16"]), ("comol_native", "system_comol_worker.py", [])]
    for config_name, concurrency, input_tokens, output_tokens in configs:
        for method, script, extra in methods:
            env = dict(service_env)
            env.update({
                "SERVICE_CONCURRENCY": str(concurrency),
                "SERVICE_INPUT_TOKENS": str(input_tokens),
                "SERVICE_OUTPUT_TOKENS": str(output_tokens),
                "SERVICE_MEASURED_REQUESTS": "50",
            })
            command = [py, str(root / "scripts" / script), "--group", "1", "--concurrency", str(concurrency), "--input-tokens", str(input_tokens), "--output-tokens", str(output_tokens), "--measured-requests", "50"] + extra
            log = work / "logs" / f"service_{args.hardware}_{config_name}_{method}.log"
            run(command, env, log)
            if method == "full":
                source = qwen_root / "groups" / "group1" / "system" / "independent_full" / "metrics.json"
            elif method == "core_q16":
                source = qwen_root / "groups" / "group1" / "system" / "core_q16" / "metrics.json"
            else:
                source = qwen_root / "groups" / "group1" / "system" / "comol_native" / "metrics.json"
            dest = output_root / config_name / f"{method}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            payload.update({"hardware": args.hardware, "config": config_name, "source_metrics": str(source)})
            dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marker = output_root / "COMPLETE.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"status": "PASS", "hardware": args.hardware, "configs": [x[0] for x in configs]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage": "service", "hardware": args.hardware}, ensure_ascii=False))


if __name__ == "__main__":
    main()
