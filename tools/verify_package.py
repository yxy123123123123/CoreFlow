"""Static release checks for the lightweight CoreFlow reproduction package."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".safetensors", ".db", ".sqlite", ".zip", ".gz"}
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "generic_api_key": re.compile(r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
}


def main() -> int:
    failures: list[str] = []
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    for path in files:
        relative = path.relative_to(ROOT)
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            failures.append(f"cache file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary/archive: {relative}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            failures.append(f"file >=100 MiB: {relative}")
        if path.suffix.lower() in {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"possible {label}: {relative}")

    required = [
        ROOT / "coreflow" / "decompose.py",
        ROOT / "coreflow" / "runtime.py",
        ROOT / "README.md",
        ROOT / "docs" / "EXPERIMENT_MAP.md",
        ROOT / "results" / "rq2_mbppplus" / "primary_decision.json",
        ROOT / "results" / "rq3_system" / "original_baselines_summary.json",
        ROOT / "results" / "submission_reproducibility_package" / "quality_task_outcomes.csv",
        ROOT / "results" / "submission_reproducibility_package" / "rtx4080_request_records.json",
        ROOT / "results" / "submission_reproducibility_package" / "qwen_training_lifecycle.json",
        ROOT / "results" / "submission_reproducibility_package" / "RELEASE_STATUS.json",
        ROOT / "CITATION.cff",
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"missing required file: {path.relative_to(ROOT)}")

    public_results = ROOT / "results" / "submission_reproducibility_package"
    quality_file = public_results / "quality_task_outcomes.csv"
    if quality_file.is_file():
        with quality_file.open("r", encoding="utf-8", newline="") as handle:
            quality_rows = list(csv.DictReader(handle))
        if len(quality_rows) != 5424:
            failures.append(f"unexpected public quality row count: {len(quality_rows)} != 5424")

    service_file = public_results / "rtx4080_request_records.json"
    if service_file.is_file():
        service_records = json.loads(service_file.read_text(encoding="utf-8")).get("records", [])
        if len(service_records) != 9:
            failures.append(f"unexpected service configuration count: {len(service_records)} != 9")
        for record in service_records:
            if len(record.get("latencies_seconds", [])) != 50:
                failures.append(f"service latency count is not 50: {record.get('config')} {record.get('method')}")
            if len(record.get("decode_tokens_per_second", [])) != 50:
                failures.append(f"service throughput count is not 50: {record.get('config')} {record.get('method')}")

    lifecycle_file = public_results / "qwen_training_lifecycle.json"
    if lifecycle_file.is_file():
        lifecycle_records = json.loads(lifecycle_file.read_text(encoding="utf-8")).get("records", [])
        if len(lifecycle_records) != 4 or any(record.get("status") != "PASS" for record in lifecycle_records):
            failures.append("Qwen group-2/group-3 lifecycle records are incomplete")

    citation_file = ROOT / "CITATION.cff"
    if citation_file.is_file():
        citation_text = citation_file.read_text(encoding="utf-8")
        if "family-names: Hong" not in citation_text or "given-names: Han" not in citation_text:
            failures.append("CITATION.cff does not include Han Hong")

    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"PASS files={len(files)} size_mib={total_bytes / 1024 / 1024:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
