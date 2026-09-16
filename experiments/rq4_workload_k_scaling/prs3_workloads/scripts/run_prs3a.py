#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, load_json, paths, write_json


def main():
    cfg, p = config(), paths()
    seal = p["reports"] / "PROTOCOL_SEAL.json"
    if load_json(seal).get("status") != "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS":
        raise RuntimeError("Run seal first")
    root = p["results"] / "prs3a"
    root.mkdir(parents=True, exist_ok=True)
    completed, ooms = [], []
    port = 7100
    for workload in cfg["prs3a"]["workloads"]:
        for block_index, order in enumerate(workload["orders"], start=1):
            block = f"block{block_index:02d}"
            for position, method in enumerate(order, start=1):
                port += 1
                final = root / workload["name"] / block / f"{position}_{method}"
                command = [
                    str(p["runtime_python"]), str(ROOT / "scripts" / "prs3a_worker.py"),
                    "--method", method,
                    "--workload", workload["name"],
                    "--block-id", block,
                    "--order-position", str(position),
                    "--batch-size", str(workload["batch_size"]),
                    "--input-length", str(workload["input_length"]),
                    "--output-length", str(workload["output_length"]),
                    "--warmup", str(workload["warmup"]),
                    "--measurements", str(workload["measurements"]),
                    "--port", str(port),
                    "--output", str(final),
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = p["cuda_device"]
                print(f"[START PRS3A] workload={workload['name']} block={block} position={position} method={method}", flush=True)
                code = subprocess.run(command, env=environment).returncode
                if code == 20:
                    if workload["name"] == cfg["prs3a"]["anchor"]:
                        raise RuntimeError(f"Anchor workload OOM is not an accepted pilot outcome: {final}")
                    ooms.append(str(final) + ".oom")
                    continue
                if code:
                    raise SystemExit(code)
                completed.append(str(final))
    payload = {
        "status": "PASS_WITH_RECORDED_BOUNDARY_OOM" if ooms else "PASS",
        "protocol_id": cfg["protocol_id"],
        "completed_runs": len(completed),
        "recorded_oom": ooms,
        "expected_runs": sum(len(item["orders"]) * 3 for item in cfg["prs3a"]["workloads"]),
        "execution": "single_gpu_exclusive_serial",
    }
    write_json(p["reports"] / "prs3a_complete.json", payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
