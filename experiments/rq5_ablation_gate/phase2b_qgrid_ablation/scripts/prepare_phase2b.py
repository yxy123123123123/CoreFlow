#!/usr/bin/env python3
"""Phase-2B preparation: dev set selection + q-grid bank build."""
from __future__ import annotations
import json, sys, argparse, hashlib, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import load_json, write_json, sha256_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--source-assets", required=True)
    parser.add_argument("--apps-raw", default="/root/autodl-tmp/external_data/apps_raw")
    parser.add_argument("--cmath-raw", default="/root/autodl-tmp/external_data/cmath_raw")
    parser.add_argument("--output-dev", required=True)
    parser.add_argument("--frozen-formal-ids", default="/root/autodl-tmp/m2_clean_source")
    args = parser.parse_args()
    cfg = load_json(ROOT / "config" / "protocol.json")
    random.seed(20260731)

    # -- Load formal-v2 frozen IDs --
    formal_apps = set()
    formal_math = set()
    fp = Path(args.frozen_formal_ids)
    af = fp / "fresh_apps_intro_heldout.jsonl"
    mf = fp / "fresh_cmath_heldout.jsonl"
    if af.exists():
        for line in af.read_text(encoding="utf-8").splitlines():
            if line.strip(): formal_apps.add(json.loads(line).get("task_id", ""))
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            if line.strip(): formal_math.add(json.loads(line).get("task_id", ""))

    # -- Select 128 APPS interview questions --
    apps_src = Path(args.apps_raw) / "test.jsonl"
    apps_candidates = []
    for line in apps_src.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("difficulty") != "interview": continue
        tid = f"apps-{row['id']}"
        if tid in formal_apps: continue
        apps_candidates.append({"task_id": tid, "problem": row["problem"], "io_cases": row.get("io_cases", [])})
    apps_selected = random.sample(apps_candidates, min(128, len(apps_candidates)))
    print(f"apps dev: selected {len(apps_selected)} / {len(apps_candidates)} candidates (overlap with formal: 0)")

    # -- Select 128 CMATH questions --
    cmath_src = Path(args.cmath_raw) / "cmath_test.jsonl"
    cmath_candidates = []
    for line in cmath_src.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        tid = str(row.get("id", row.get("task_id", "")))
        if tid in formal_math: continue
        q = row.get("question", row.get("problem", ""))
        a = str(row.get("answer", ""))
        cmath_candidates.append({"task_id": tid, "problem": q, "answer": a})
    cmath_selected = random.sample(cmath_candidates, min(128, len(cmath_candidates)))
    print(f"cmath dev: selected {len(cmath_selected)} / {len(cmath_candidates)} candidates (overlap with formal: 0)")

    # -- Write frozen dev sets --
    out = Path(args.output_dev)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("code_dev128.jsonl", apps_selected), ("math_dev128.jsonl", cmath_selected)]:
        f = out / name
        with f.open("w", encoding="utf-8", newline="\n") as fh:
            for row in data: fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    # audit
    audit = {
        "status": "PASS",
        "code_dev_ids": sorted(r["task_id"] for r in apps_selected),
        "math_dev_ids": sorted(r["task_id"] for r in cmath_selected),
        "code_formal_overlap": sorted(formal_apps & {r["task_id"] for r in apps_selected}),
        "math_formal_overlap": sorted(formal_math & {r["task_id"] for r in cmath_selected}),
    }
    assert audit["code_formal_overlap"] == [], f"CODE OVERLAP: {audit['code_formal_overlap']}"
    assert audit["math_formal_overlap"] == [], f"MATH OVERLAP: {audit['math_formal_overlap']}"
    write_json(out / "dev_set_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))

if __name__ == "__main__":
    main()