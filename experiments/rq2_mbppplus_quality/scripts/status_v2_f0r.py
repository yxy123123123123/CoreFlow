#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()
    work = Path(args.work_root)
    report = work / "reports" / "formal_v2_f0r"
    result = work / "results" / "formal_v2_f0r" / "go_nogo_quality"
    names = [
        "preflight.json",
        "data_audit.json",
        "isvd_asset_lock.json",
        "e0_correctness.json",
        "PROTOCOL_SEAL.json",
        "go_nogo_decision.json",
    ]
    print("=== v2 F0R MBPP+ interface reports ===")
    for name in names:
        payload = load(report / name)
        status = "MISSING" if payload is None else payload.get("status", "UNKNOWN")
        extra = ""
        if name == "go_nogo_decision.json" and payload:
            extra = f" decision={payload.get('decision')} floor={payload.get('mbppplus_dev_floor')} full={payload.get('mbppplus_full_correct')}/128"
        print(f"{name}: {status}{extra}")
    print("\n=== method progress ===")
    methods = [
        "math_full_k5_seed41",
        "math_core_q185_k5_seed41",
        "math_isvd_q185_k5_seed41",
        "code_full_k5_seed41",
        "code_core_q185_k5_seed41",
        "code_isvd_q185_k5_seed41",
    ]
    for method in methods:
        complete = result / method / "COMPLETE.json"
        partial = result / f"{method}.partial"
        if complete.exists():
            summary = load(result / method / "summary.json") or {}
            print(f"{method}: COMPLETE correct={summary.get('correct')}/{summary.get('rows')}")
        elif partial.exists():
            lines = partial / "executor_results.jsonl"
            count = 0
            if lines.exists():
                with lines.open("r", encoding="utf-8", errors="replace") as handle:
                    count = sum(1 for _ in handle)
            print(f"{method}: PARTIAL rows={count}")
        else:
            print(f"{method}: PENDING")


if __name__ == "__main__":
    main()

