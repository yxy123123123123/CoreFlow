#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
import zlib
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_FILES = {
    "v4": ("test4.jsonl", "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"),
    "v5": ("test5.jsonl", "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"),
    "v6": ("test6.jsonl", "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
}
REPO_COMMIT = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SPLIT_SALT = "coreflow-submission-final-stage1-v1|20260802"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(row: dict) -> str:
    token = f"{SPLIT_SALT}|{row['source_version']}|{row['question_id']}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def allocate(total: int, sizes: dict[tuple, int]) -> dict[tuple, int]:
    population = sum(sizes.values())
    raw = {key: total * size / population for key, size in sizes.items()}
    out = {key: int(value) for key, value in raw.items()}
    left = total - sum(out.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - out[key]), key))
    for key in order[:left]:
        out[key] += 1
    return out


def decode_tests(value: str) -> list[dict]:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    source_root, output_root = Path(args.source_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    eligible: list[dict] = []
    source_manifest = {}
    for version, (filename, expected) in SOURCE_FILES.items():
        path = source_root / filename
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Official source hash mismatch for {filename}: {observed}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                count += 1
                if row["difficulty"] not in {"easy", "medium"}:
                    continue
                row["source_version"] = version
                row["source_repo_commit"] = REPO_COMMIT
                row["split_hash"] = stable_key(row)
                eligible.append(row)
        source_manifest[filename] = {"sha256": observed, "rows": count, "version": version}
    if len(eligible) != 244 or len({row["question_id"] for row in eligible}) != 244:
        raise ValueError(f"Eligible pool contract failed: rows={len(eligible)}")
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in eligible:
        strata[(row["source_version"], row["difficulty"], row["platform"])].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: (row["split_hash"], row["question_id"]))
    sizes = {key: len(rows) for key, rows in strata.items()}
    q_alloc = allocate(64, sizes)
    qualification, remaining = [], {}
    for key, rows in sorted(strata.items()):
        qualification.extend(rows[: q_alloc[key]])
        remaining[key] = rows[q_alloc[key] :]
    f_alloc = allocate(164, {key: len(rows) for key, rows in remaining.items()})
    formal, reserve = [], []
    for key, rows in sorted(remaining.items()):
        formal.extend(rows[: f_alloc[key]])
        reserve.extend(rows[f_alloc[key] :])
    for rows in (qualification, formal, reserve):
        rows.sort(key=lambda row: (row["split_hash"], row["question_id"]))
    if [len(qualification), len(formal), len(reserve)] != [64, 164, 16]:
        raise AssertionError((len(qualification), len(formal), len(reserve)))
    all_ids = [row["question_id"] for row in qualification + formal + reserve]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split overlap detected")
    for row in eligible:
        public = json.loads(row["public_test_cases"])
        private = decode_tests(row["private_test_cases"])
        if not public or not private:
            raise ValueError(f"Missing tests: {row['question_id']}")
        if {item["testtype"] for item in public + private} - {"stdin", "functional"}:
            raise ValueError(f"Unknown test type: {row['question_id']}")
    outputs = {
        "livecodebench_qualification64.jsonl": qualification,
        "livecodebench_formal164.jsonl": formal,
        "livecodebench_reserve16.jsonl": reserve,
    }
    for name, rows in outputs.items():
        write_jsonl(output_root / name, rows)
    manifest = {
        "status": "FROZEN_BEFORE_ANY_GPU_OUTPUT",
        "dataset": "livecodebench/code_generation_lite",
        "repository_commit": REPO_COMMIT,
        "source_versions": ["v4", "v5", "v6"],
        "eligibility": {"difficulty": ["easy", "medium"], "rows": 244},
        "split_rule": "stratify by source_version,difficulty,platform; SHA-256 order using frozen salt; proportional largest remainder",
        "split_salt_sha256": hashlib.sha256(SPLIT_SALT.encode()).hexdigest(),
        "sources": source_manifest,
        "splits": {},
    }
    for name, rows in outputs.items():
        manifest["splits"][name] = {
            "rows": len(rows),
            "sha256": sha256_file(output_root / name),
            "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
            "platform": dict(sorted(Counter(row["platform"] for row in rows).items())),
            "source_version": dict(sorted(Counter(row["source_version"] for row in rows).items())),
            "question_ids_sha256": hashlib.sha256("\n".join(row["question_id"] for row in rows).encode()).hexdigest(),
        }
    (output_root / "livecodebench_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": "PASS", "splits": {k: len(v) for k, v in outputs.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
