#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import config, paths


def value(path: Path, key: str = "status") -> str:
    if not path.is_file():
        return "MISSING"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get(key, "PRESENT"))
    except Exception:
        return "UNREADABLE"


def main() -> None:
    cfg, p = config(), paths()
    print("=== PRS-4A / PRS-4P reports ===")
    for name in (
        "preflight.json",
        "scheme_a_legacy_qualification_closure.json",
        "PILOT_SEAL.json",
        "pilot_b_complete.json",
        "final/prs4a_prs4p_summary.json",
    ):
        print(f"{name}: {value(p['reports'] / name)}")
    print("\n=== Pilot B method progress ===")
    for method in cfg["methods"]:
        root = p["results"] / "pilot_b" / method["id"]
        partial = root.with_name(root.name + ".partial")
        if (root / "COMPLETE.json").is_file():
            item = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
            print(f"{method['id']}: COMPLETE correct={item['correct']}/{item['rows']}")
        elif partial.exists():
            rows = 0
            executor = partial / "executor_results.jsonl"
            if executor.is_file():
                rows = sum(1 for line in executor.open("r", encoding="utf-8") if line.strip())
            print(f"{method['id']}: PARTIAL rows={rows}/{cfg['data']['pilot_rows']}")
        else:
            print(f"{method['id']}: PENDING")
    print("\nformal164: CLOSED / NOT EMBEDDED / NOT ACCESSED")


if __name__ == "__main__":
    main()
