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
    report = work / "reports" / "indep_confirm_classeval"
    result = work / "results" / "indep_confirm_classeval"
    names = [
        "preflight.json",
        "data_audit.json",
        "isvd_asset_lock.json",
        "evaluator_seal.json",
        "PROTOCOL_SEAL.json",
        "qualification_decision.json",
        "primary_decision.json",
    ]
    print("=== indep-confirm classeval reports ===")
    for name in names:
        payload = load(report / name)
        status = "MISSING" if payload is None else payload.get("status", payload.get("decision", "UNKNOWN"))
        extra = ""
        if name == "qualification_decision.json" and payload:
            extra = f" decision={payload.get('decision')} correct={payload.get('correct')}/{payload.get('rows')}"
        if name == "primary_decision.json" and payload:
            extra = f" decision={payload.get('decision')}"
        print(f"{name}: {status}{extra}")
    print("\n=== qualification method progress ===")
    qual_method = "code_full_k5_seed41"
    qual_dir = result / "qualification" / qual_method
    complete = qual_dir / "COMPLETE.json"
    partial = result / "qualification" / f"{qual_method}.partial"
    if complete.exists():
        summary = load(qual_dir / "summary.json") or {}
        print(f"{qual_method}: COMPLETE correct={summary.get('correct')}/{summary.get('rows')}")
    elif partial.exists():
        lines = partial / "executor_results.jsonl"
        count = 0
        if lines.exists():
            with lines.open("r", encoding="utf-8", errors="replace") as handle:
                count = sum(1 for _ in handle)
        print(f"{qual_method}: PARTIAL rows={count}")
    else:
        print(f"{qual_method}: PENDING")
    print("\n=== formal method progress ===")
    methods = [
        "code_full_k5_seed41",
        "code_core_q185_k5_seed41",
        "code_isvd_q185_k5_seed41",
        "code_full_k5_seed42",
        "code_core_q185_k5_seed42",
        "code_isvd_q185_k5_seed42",
        "code_full_k5_seed43",
        "code_core_q185_k5_seed43",
        "code_isvd_q185_k5_seed43",
    ]
    for method in methods:
        complete = result / "formal_main" / method / "COMPLETE.json"
        partial = result / "formal_main" / f"{method}.partial"
        if complete.exists():
            summary = load(result / "formal_main" / method / "summary.json") or {}
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
