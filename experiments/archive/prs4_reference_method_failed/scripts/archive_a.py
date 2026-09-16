#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, load_json, now_utc, paths, sha256_file, write_json


def main() -> None:
    cfg, p = config(), paths()
    if not (p["reports"] / "preflight.json").is_file():
        raise RuntimeError("Run preflight first")
    target = p["reports"] / "scheme_a_legacy_qualification_closure.json"
    if target.exists():
        print(f"[SKIP] {target}")
        return

    spec = cfg["legacy_A"]
    raw_dir = p["legacy"] / spec["raw_archive_optional_path"]
    complete_path = raw_dir / "COMPLETE.json"
    summary_path = raw_dir / "summary.json"
    executor_path = raw_dir / "executor_results.jsonl"
    generation_path = raw_dir / "generations.jsonl"
    raw_archive = {"available": False}
    evidence_strength = "ATTESTED_CONSOLE_DIAGNOSTICS_WITH_FROZEN_DATA_IDENTITY"

    if complete_path.is_file() and executor_path.is_file() and generation_path.is_file():
        complete = load_json(complete_path)
        summary = load_json(summary_path) if summary_path.is_file() else complete
        observed = {
            "method": summary.get("method"),
            "seed": int(summary.get("seed", -1)),
            "rows": int(summary.get("rows", -1)),
            "correct": int(summary.get("correct", -1)),
        }
        expected = {
            "method": spec["actual_method"],
            "seed": spec["gate_seed"],
            "rows": spec["rows"],
            "correct": spec["correct"],
        }
        if observed != expected:
            raise ValueError(f"Legacy raw archive does not match attestation: observed={observed}, expected={expected}")
        executor_rows = sum(1 for line in executor_path.open("r", encoding="utf-8") if line.strip())
        generation_rows = sum(1 for line in generation_path.open("r", encoding="utf-8") if line.strip())
        if executor_rows != spec["rows"] or generation_rows != spec["rows"]:
            raise ValueError(f"Legacy raw archive row mismatch: executor={executor_rows}, generations={generation_rows}")
        raw_archive = {
            "available": True,
            "path": str(raw_dir),
            "complete_sha256": sha256_file(complete_path),
            "executor_sha256": sha256_file(executor_path),
            "generations_sha256": sha256_file(generation_path),
            "executor_rows": executor_rows,
            "generation_rows": generation_rows,
        }
        evidence_strength = "FULL_RAW_ARCHIVE_VERIFIED"

    diagnostics = ROOT / "evidence" / "legacy_stage1_qualification_console_diagnostics.txt"
    payload = {
        "status": "ARCHIVED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "scheme": "A",
        "legacy_protocol": spec["source_protocol"],
        "legacy_actual_method": spec["actual_method"],
        "legacy_gate_seed": spec["gate_seed"],
        "legacy_result": {
            "rows": spec["rows"],
            "correct": spec["correct"],
            "minimum_correct": spec["minimum_correct"],
            "stage1_decision": spec["archival_decision"],
        },
        "qualification64_sha256": cfg["data"]["qualification64_sha256"],
        "console_diagnostics": {
            "path": str(diagnostics),
            "sha256": sha256_file(diagnostics),
        },
        "raw_archive": raw_archive,
        "evidence_strength": evidence_strength,
        "prs4q_original_loraflow_status": "NOT_RUN_BY_THIS_ARCHIVE",
        "formal164_branch": "CLOSED",
        "formal164_may_be_opened": False,
        "interpretation": (
            "The previous Stage-1 full-vectorized run observed a 0/64 floor and closed that branch. "
            "It is preserved as failure evidence, but it is not relabelled as the PRS-v2 Original LoRA-Flow qualification."
        ),
        "method_mismatch_note": spec["method_mismatch_note"],
    }
    write_json(target, payload)
    print(json.dumps({"status": "ARCHIVED", "legacy_decision": spec["archival_decision"], "evidence_strength": evidence_strength, "formal164": "CLOSED"}))


if __name__ == "__main__":
    main()
