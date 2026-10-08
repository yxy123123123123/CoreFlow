from __future__ import annotations

import json
import os
from pathlib import Path


def p(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def state(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("status", "PRESENT"))
    except Exception:
        return "PRESENT"


def main() -> None:
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")
    report = work / "reports" / "reviewer_minimal_phasea"
    result = work / "results" / "reviewer_minimal_phasea"
    print("=== CoreFlow reviewer minimal Phase-A ===")
    for name in ("preflight.json", "prepare.json", "PROTOCOL_SEAL.json", "q224_complete.json", "drift_diagnostics.json", "analysis.json"):
        print(f"{name}: {state(report / name)}")
    for label, path in (
        ("q224_mbpp", result / "q224_mbpp"),
        ("q224_classeval", result / "q224_classeval"),
        ("service", result / "service"),
    ):
        complete = len(list(path.rglob("COMPLETE.json"))) if path.exists() else 0
        metrics = len(list(path.rglob("metrics.json"))) if path.exists() else 0
        partial = len(list(path.rglob("*.partial"))) if path.exists() else 0
        print(f"{label}: complete_markers={complete} metrics={metrics} partial_dirs={partial}")
    print("=== recent logs ===")
    logs = work / "logs"
    if logs.exists():
        for path in sorted(logs.glob("*.log"), key=lambda x: x.stat().st_mtime)[-8:]:
            print(path.name)


if __name__ == "__main__":
    main()
