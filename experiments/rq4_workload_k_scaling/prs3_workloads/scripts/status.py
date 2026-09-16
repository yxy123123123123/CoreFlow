#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, paths


def marker(path):
    if not path.is_file():
        return "MISSING"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status", "PRESENT")
    except Exception:
        return "PRESENT_UNREADABLE"


def main():
    cfg, p = config(), paths()
    print("=== PRS-3 pilot reports ===")
    for name in ("preflight.json", "prepare.json", "PROTOCOL_SEAL.json", "prs3a_complete.json", "prs3b_complete.json", "final/prs3_pilot_summary.json"):
        print(f"{name}: {marker(p['reports'] / name)}")
    print("\n=== PRS-3A progress ===")
    for workload in cfg["prs3a"]["workloads"]:
        root = p["results"] / "prs3a" / workload["name"]
        expected = len(workload["orders"]) * 3
        complete = len(list(root.glob("block*/*/COMPLETE.json"))) if root.exists() else 0
        partial = len(list(root.glob("block*/*.partial/PARTIAL.json"))) if root.exists() else 0
        oom = len(list(root.glob("block*/*.oom/OOM.json"))) if root.exists() else 0
        print(f"{workload['name']}: complete={complete}/{expected}, oom={oom}, partial={partial}")
    print("\n=== PRS-3B progress ===")
    root = p["results"] / "prs3b"
    complete = len(list(root.glob("*/COMPLETE.json"))) if root.exists() else 0
    partial = len(list(root.glob("*.partial/PARTIAL.json"))) if root.exists() else 0
    oom = len(list(root.glob("*.oom/OOM.json"))) if root.exists() else 0
    print(f"runs: complete={complete}/15, oom={oom}, partial={partial}")
    if root.exists():
        for item in sorted(root.glob("*.partial")):
            measurement = item / "measurements.jsonl"
            lines = sum(1 for _ in measurement.open(encoding="utf-8")) if measurement.is_file() else 0
            print(f"  ACTIVE_OR_FAILED_PARTIAL {item.name}: measurement_rows={lines}")
    print("\n=== related processes ===")
    result = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etime,stat,%cpu,%mem,cmd | grep -E 'run_prs3|prs3a_worker|prs3b_worker|serve_method.py' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stdout.rstrip() or "none")


if __name__ == "__main__":
    main()
