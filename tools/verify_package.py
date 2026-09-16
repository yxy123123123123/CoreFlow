"""Static release checks for the lightweight CoreFlow reproduction package."""

from __future__ import annotations

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
    files = [path for path in ROOT.rglob("*") if path.is_file()]
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
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"missing required file: {path.relative_to(ROOT)}")

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

