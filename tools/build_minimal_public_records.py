"""Build privacy- and license-conscious public evidence records.

The input manifest points to local raw result files that are intentionally not
redistributed.  The generated files retain only task identifiers, binary
outcomes, hashes, numerical request measurements, and lifecycle timings.  No
benchmark prompts, tests, model generations, checkpoints, or private paths are
written to the output package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_quality_records(sources: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source["path"])
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                evaluation = record.get("evaluation", {})
                rows.append(
                    {
                        "benchmark": source["benchmark"],
                        "method": source["method"],
                        "gate_checkpoint": int(source["gate_checkpoint"]),
                        "task_id": str(record["task_id"]),
                        "passed": bool(evaluation.get("passed", False)),
                        "status": str(evaluation.get("status", "UNKNOWN")),
                        "raw_output_sha256": str(record.get("raw_output_sha256", "")),
                    }
                )
                count += 1
        source_audit.append(
            {
                "benchmark": source["benchmark"],
                "method": source["method"],
                "gate_checkpoint": int(source["gate_checkpoint"]),
                "source_sha256": sha256(path),
                "record_count": count,
            }
        )

    rows.sort(key=lambda r: (r["benchmark"], r["method"], r["gate_checkpoint"], r["task_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"output": output.name, "record_count": len(rows), "sources": source_audit}


def build_service_records(sources: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source["path"])
        raw = read_json(path)
        payloads.append(
            {
                "hardware": raw["hardware"],
                "config": raw["config"],
                "method": raw["method"],
                "measurement_scope": raw["measurement_scope"],
                "input_tokens": raw["input_tokens"],
                "output_tokens": raw["output_tokens"],
                "concurrency_label": raw["concurrency_label"],
                "startup_health_ready_seconds": raw["startup_health_ready_seconds"],
                "latencies_seconds": raw["latencies_seconds"],
                "decode_tokens_per_second": raw["decode_tokens_per_second"],
                "peak_allocated_gib": raw["peak_allocated_gib"],
                "peak_reserved_gib": raw["peak_reserved_gib"],
                "average_power_watts": raw["average_power_watts"],
                "joules_per_request": raw["joules_per_request"],
                "joules_per_token": raw["joules_per_token"],
            }
        )
        source_audit.append({"source_sha256": sha256(path), "config": raw["config"], "method": raw["method"]})
    payloads.sort(key=lambda r: (r["config"], r["method"]))
    output.write_text(json.dumps({"records": payloads}, indent=2) + "\n", encoding="utf-8")
    return {"output": output.name, "configuration_count": len(payloads), "sources": source_audit}


def build_lifecycle_records(sources: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source["path"])
        raw = read_json(path)
        records.append(
            {
                "group": str(source["group"]),
                "route": source["route"],
                "status": raw.get("status"),
                "elapsed_seconds": raw.get("elapsed_seconds"),
                "seed": raw.get("seed"),
                "expert_count": len(raw.get("records", [])) or source.get("expert_count"),
                "rank": raw.get("training", {}).get("rank"),
                "epochs": raw.get("training", {}).get("epochs"),
            }
        )
        source_audit.append(
            {"group": str(source["group"]), "route": source["route"], "source_sha256": sha256(path)}
        )
    records.sort(key=lambda r: (r["group"], r["route"]))
    output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    return {"output": output.name, "record_count": len(records), "sources": source_audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit = {
        "schema_version": 1,
        "privacy_boundary": "No prompts, tests, generations, weights, or private source paths are exported.",
        "quality": build_quality_records(
            manifest["quality_sources"], args.output_dir / "quality_task_outcomes.csv"
        ),
        "service": build_service_records(
            manifest["service_sources"], args.output_dir / "rtx4080_request_records.json"
        ),
        "lifecycle": build_lifecycle_records(
            manifest["lifecycle_sources"], args.output_dir / "qwen_training_lifecycle.json"
        ),
    }
    (args.output_dir / "source_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
