#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the q224 fallback scope before any q224 formal output is generated."
    )
    parser.add_argument("--primary-decision", required=True)
    parser.add_argument("--q224-result-root", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated subset of math,code")
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    decision = json.loads(Path(args.primary_decision).read_text(encoding="utf-8"))
    required = sorted(decision.get("q224_decision_required_tasks") or [])
    tasks = sorted(set(part.strip() for part in args.tasks.split(",") if part.strip()))
    if not tasks or any(task not in {"math", "code"} for task in tasks):
        raise ValueError("--tasks must contain math and/or code")
    if tasks != required:
        raise ValueError(f"Activation tasks must exactly match q185 failed tasks: required={required}, requested={tasks}")
    qroot = Path(args.q224_result_root)
    if qroot.exists() and any(qroot.iterdir()):
        raise ValueError("q224 output already exists; activation is no longer pre-outcome")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "status": "ACTIVATED_BEFORE_Q224_OUTPUTS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "gate_seed": 41,
        "q": 224,
        "rationale": args.rationale,
        "three_seed_upgrade_forbidden_after_any_q224_output": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
