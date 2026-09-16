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


def invoke(cfg, p, pool, method, launch, sentinel_only, output):
    command = [
        str(p["runtime_python"]),
        str(ROOT / "scripts" / "prs3b_worker.py"),
        "--pool", pool,
        "--method", method,
        "--launch", str(launch),
        "--output", str(output),
    ]
    if sentinel_only:
        command.append("--sentinel-only")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = p["cuda_device"]
    print(f"[START PRS3B] pool={pool} launch={launch} sentinel_only={sentinel_only} method={method}", flush=True)
    return subprocess.run(command, env=environment).returncode


def main():
    cfg, p = config(), paths()
    if load_json(p["reports"] / "PROTOCOL_SEAL.json").get("status") != "SEALED_BEFORE_PRS3_PILOT_GPU_OUTPUTS":
        raise RuntimeError("Run seal first")
    result_root = p["results"] / "prs3b"
    result_root.mkdir(parents=True, exist_ok=True)
    orders = {
        "k3_strict": ["full_legacy_original", "coreflow_q185_frozen", "isvd_legacy_padded_matched_q185"],
        "k5_formal": ["coreflow_q185_frozen", "isvd_legacy_padded_matched_q185", "full_legacy_original"],
        "k8_strict": ["isvd_legacy_padded_matched_q185", "full_legacy_original", "coreflow_q185_frozen"],
    }
    complete, ooms = [], []
    for pool in cfg["prs3b"]["pools"]:
        for method in orders[pool]:
            output = result_root / f"main_{pool}_{method}_launch1"
            code = invoke(cfg, p, pool, method, 1, False, output)
            if code == 20:
                ooms.append(str(output) + ".oom")
            elif code:
                raise SystemExit(code)
            else:
                complete.append(str(output))
    pool = cfg["prs3b"]["cv_sentinel_pool"]
    extra_orders = {
        2: ["isvd_legacy_padded_matched_q185", "full_legacy_original", "coreflow_q185_frozen"],
        3: ["full_legacy_original", "coreflow_q185_frozen", "isvd_legacy_padded_matched_q185"],
    }
    for launch in range(2, int(cfg["prs3b"]["cv_total_launches"]) + 1):
        for method in extra_orders[launch]:
            output = result_root / f"sentinel_{pool}_{method}_launch{launch}"
            code = invoke(cfg, p, pool, method, launch, True, output)
            if code == 20:
                ooms.append(str(output) + ".oom")
            elif code:
                raise SystemExit(code)
            else:
                complete.append(str(output))
    payload = {
        "status": "PASS_WITH_RECORDED_OOM" if ooms else "PASS",
        "protocol_id": cfg["protocol_id"],
        "completed_runs": len(complete),
        "recorded_oom": ooms,
        "expected_runs": 15,
        "execution": "single_gpu_exclusive_serial",
    }
    write_json(p["reports"] / "prs3b_complete.json", payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
