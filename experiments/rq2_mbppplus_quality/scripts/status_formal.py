#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def json_status(path: Path) -> str:
    if not path.exists():
        return "PENDING"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
    except Exception:
        return "INVALID"


def method_status(root: Path, method: str) -> str:
    if (root / method / "COMPLETE.json").exists():
        return "COMPLETE"
    if (root / f"{method}.partial").exists():
        return "FAILED_OR_RUNNING_PARTIAL"
    return "PENDING"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-work-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    work = Path(args.formal_work_root)
    reports = work / "reports" / "formal_v1"
    results = work / "results" / "formal_v1"
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("=== F0 / seals ===")
    for name in (
        "assets_installed.json",
        "preflight.json",
        "data_audit.json",
        "legacy_math_k5_seed42_pair.json",
        "smoke_decision.json",
        "PROTOCOL_SEAL.json",
        "code_floor_decision.json",
        "primary_decision.json",
        "q224_activation.json",
        "microbenchmark_decision.json",
        "secondary_decision.json",
    ):
        print(f"{name}: {json_status(reports / name)}")
    print()
    print("=== method progress ===")
    groups = {
        "f0_legacy_pair": config["phase_orders"]["legacy_pair"],
        "f0_smoke": ["code_full_k5_seed41", "code_core_q185_k5_seed41"],
        "code_floor": [config["code_floor"]["method"]],
        "math_main": config["phase_orders"]["math_main"],
        "code_main": config["phase_orders"]["code_main"],
        "k_extension": config["phase_orders"]["k_extension"],
        "baselines": config["phase_orders"]["baselines"],
        "q224": config["phase_orders"]["q224_conditional"],
    }
    for group, methods in groups.items():
        states = {method: method_status(results / group, method) for method in methods}
        complete = sum(state == "COMPLETE" for state in states.values())
        partial = sum(state == "FAILED_OR_RUNNING_PARTIAL" for state in states.values())
        print(f"{group}: complete={complete}/{len(methods)}, partial={partial}")
        for method, state in states.items():
            if state != "PENDING":
                print(f"  {method}: {state}")
    micro_root = results / "microbenchmark"
    micro_methods = config["microbenchmark"]["methods"]
    complete = sum(method_status(micro_root, name) == "COMPLETE" for name in micro_methods)
    partial = sum(method_status(micro_root, name) == "FAILED_OR_RUNNING_PARTIAL" for name in micro_methods)
    print(f"microbenchmark: complete={complete}/{len(micro_methods)}, partial={partial}")


if __name__ == "__main__":
    main()
