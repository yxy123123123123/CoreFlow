#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import config, paths, sha256_file, write_json


def main():
    cfg, p = config(), paths()
    p["data"].mkdir(parents=True, exist_ok=True)
    p["reports"].mkdir(parents=True, exist_ok=True)
    p["results"].mkdir(parents=True, exist_ok=True)
    seal = p["reports"] / "PROTOCOL_SEAL.json"
    marker = p["reports"] / "prepare.json"
    if seal.is_file():
        if not marker.is_file():
            raise RuntimeError("Seal exists but prepare marker is missing")
        print(json.dumps({"status": "SKIP_ALREADY_SEALED", "prepare": str(marker)}))
        return
    source = ROOT / "data" / cfg["data"]["source_file"]
    prompt_file = ROOT / "data" / cfg["data"]["prompt_file"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != cfg["data"]["source_rows"] or sha256_file(source) != cfg["data"]["source_sha256"]:
        raise ValueError("Frozen dev128 contract failed")
    if sha256_file(prompt_file) != cfg["data"]["prompt_sha256"]:
        raise ValueError("Prompt hash mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(p["model"]), local_files_only=True)
    template = prompt_file.read_text(encoding="utf-8")
    filler = tokenizer(
        "\n# CoreFlow deterministic systems benchmark context. This comment is intentionally inert.",
        add_special_tokens=False,
    )["input_ids"]
    if not filler:
        raise ValueError("Tokenizer produced empty filler")
    lengths = sorted({int(item["input_length"]) for item in cfg["prs3a"]["workloads"]})
    files = {}
    for length in lengths:
        frozen = []
        for row in rows[: cfg["data"]["benchmark_rows"]]:
            prompt = template.format(problem=row["problem"], entry_point=row["entry_point"])
            ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=length)["input_ids"]
            while len(ids) < length:
                ids.extend(filler[: length - len(ids)])
            ids = ids[:length]
            frozen.append(
                {
                    "task_id": row["task_id"],
                    "source_task_id": row.get("source_task_id"),
                    "input_length": length,
                    "input_ids": ids,
                    "attention_mask": [1] * length,
                }
            )
        if any(len(item["input_ids"]) != length or sum(item["attention_mask"]) != length for item in frozen):
            raise AssertionError("Effective input contract failed")
        path = p["data"] / f"effective_inputs_l{length}.json"
        write_json(path, frozen)
        files[path.name] = {
            "rows": len(frozen),
            "sha256": sha256_file(path),
            "all_attention_tokens_active": True,
        }
    write_json(
        marker,
        {
            "status": "PASS",
            "protocol_id": cfg["protocol_id"],
            "source_sha256": sha256_file(source),
            "prompt_sha256": sha256_file(prompt_file),
            "effective_inputs": files,
        },
    )
    print(json.dumps({"status": "PASS", "effective_inputs": files}, ensure_ascii=False))


if __name__ == "__main__":
    main()
