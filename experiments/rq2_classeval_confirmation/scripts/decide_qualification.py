#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import load_jsonl, sha256_file


def binomial_sf(n: int, k: int, p: float) -> float:
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="code_full_k5_seed41")
    args = parser.parse_args()

    cfg = load_json(args.config)
    result_root = Path(args.result_root)
    report_root = Path(args.report_root)
    method_dir = result_root / "qualification" / args.method
    if not (method_dir / "COMPLETE.json").exists():
        raise FileNotFoundError(f"Qualification method not complete: {method_dir}")
    rows = load_jsonl(method_dir / "executor_results.jsonl")
    outcomes = [bool(row["evaluation"]["passed"]) for row in rows]
    correct = sum(outcomes)
    n = len(outcomes)
    thresholds = cfg["hypotheses"]["qualification"]
    k = int(thresholds["min_correct"])
    p_null = float(thresholds["p_null"])
    p_target = float(thresholds["p_target"])
    passed = n >= int(thresholds["min_rows"]) and correct >= k
    status_counts = dict(sorted(Counter(row["evaluation"]["status"] for row in rows).items()))
    parseable = sum(1 for row in rows if row["evaluation"]["status"] not in {"PARSE_FAILURE"})
    payload = {
        "status": "PASS" if passed else "FAIL",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "OPEN_FORMAL" if passed else "DO_NOT_OPEN_FORMAL",
        "method": args.method,
        "rows": n,
        "correct": correct,
        "pass_at_1": correct / n if n else None,
        "threshold": {"min_rows": thresholds["min_rows"], "min_correct": k, "p_null": p_null, "p_target": p_target},
        "binomial": {
            "p_value_at_p_null": binomial_sf(n, correct, p_null),
            "power_at_p_target": binomial_sf(n, k, p_target),
        },
        "diagnostics": {
            "status_counts": status_counts,
            "parse_rate": parseable / n if n else None,
            "executable_rate": sum(1 for row in rows if row["evaluation"].get("execution_attempted", False)) / n if n else None,
        },
        "result_sha256": sha256_file(method_dir / "executor_results.jsonl"),
        "formal_opened": passed,
    }
    write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "decision": payload["decision"],
        "correct": correct,
        "rows": n,
        "k": k,
        "formal_opened": passed,
    }, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
