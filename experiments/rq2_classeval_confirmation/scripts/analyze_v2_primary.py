#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.gate_io import load_json, write_json
from coreflow.io import load_jsonl, sha256_file
from coreflow.statistics import paired_bootstrap_difference, paired_transitions

RESULT_GROUP = "formal_main"


def outcomes(result_root: Path, method: str) -> dict[str, bool]:
    rows = load_jsonl(result_root / RESULT_GROUP / method / "executor_results.jsonl")
    return {row["task_id"]: bool(row["evaluation"]["passed"]) for row in rows}


def status_counts(result_root: Path, method: str) -> dict:
    rows = load_jsonl(result_root / RESULT_GROUP / method / "executor_results.jsonl")
    return dict(sorted(Counter(row["evaluation"]["status"] for row in rows).items()))


def parse_counts(result_root: Path, method: str) -> dict:
    rows = load_jsonl(result_root / RESULT_GROUP / method / "executor_results.jsonl")
    return dict(sorted(Counter(row["evaluation"].get("parse_mode") for row in rows).items()))


def compare(reference: dict[str, bool], candidate: dict[str, bool], cfg: dict, seed_offset: int) -> dict:
    if set(reference) != set(candidate):
        raise ValueError("Task IDs differ between paired methods")
    task_ids = sorted(reference)
    ref = [reference[task_id] for task_id in task_ids]
    cand = [candidate[task_id] for task_id in task_ids]
    boot = paired_bootstrap_difference(
        ref,
        cand,
        samples=int(cfg["hypotheses"]["paired_cluster_bootstrap_samples"]),
        confidence=float(cfg["hypotheses"]["bootstrap_confidence"]),
        seed=int(cfg["hypotheses"]["bootstrap_seed"]) + seed_offset,
    )
    trans = paired_transitions(reference, candidate)
    margin = -float(cfg["hypotheses"]["noninferiority_margin_pp"])
    sensitivity = {}
    for label in cfg["hypotheses"]["sensitivity_margins_pp"]:
        sensitivity[label] = 100.0 * boot["lower"] >= -float(label)
    return {
        "rows": len(task_ids),
        "reference_correct": sum(ref),
        "candidate_correct": sum(cand),
        "candidate_minus_reference_pp": 100.0 * boot["estimate"],
        "ci95_pp": [100.0 * boot["lower"], 100.0 * boot["upper"]],
        "noninferiority_margin_pp": margin,
        "passes_noninferiority": 100.0 * boot["lower"] >= margin,
        "sensitivity": sensitivity,
        "transitions": trans,
    }


def pooled(items: list[dict[str, bool]]) -> dict[str, bool]:
    pooled_out = {}
    for seed_index, item in enumerate(items):
        for task_id, passed in item.items():
            pooled_out[f"seed{seed_index}:{task_id}"] = passed
    return pooled_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    cfg = load_json(args.config)
    result_root = Path(args.result_root)
    report_root = Path(args.report_root)
    seeds = [41, 42, 43]
    methods = cfg["phase_orders"]["formal_main"]
    missing = [m for m in methods if not (result_root / RESULT_GROUP / m / "COMPLETE.json").exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete formal methods: {missing}")

    summaries = {method: load_json(result_root / RESULT_GROUP / method / "summary.json") for method in methods}
    complete_hashes = {method: sha256_file(result_root / RESULT_GROUP / method / "COMPLETE.json") for method in methods}
    full_by_seed = []
    core_by_seed = []
    isvd_by_seed = []
    by_seed = {}
    for idx, seed in enumerate(seeds):
        full = outcomes(result_root, f"code_full_k5_seed{seed}")
        core = outcomes(result_root, f"code_core_q185_k5_seed{seed}")
        isvd = outcomes(result_root, f"code_isvd_q185_k5_seed{seed}")
        full_by_seed.append(full)
        core_by_seed.append(core)
        isvd_by_seed.append(isvd)
        by_seed[str(seed)] = {
            "full_correct": sum(full.values()),
            "core_q185_correct": sum(core.values()),
            "isvd_q185_correct": sum(isvd.values()),
            "core_vs_full": compare(full, core, cfg, idx * 10 + 1),
            "core_vs_isvd": compare(isvd, core, cfg, idx * 10 + 2),
            "isvd_vs_full": compare(full, isvd, cfg, idx * 10 + 3),
        }
    pooled_full = pooled(full_by_seed)
    pooled_core = pooled(core_by_seed)
    pooled_isvd = pooled(isvd_by_seed)
    pooled_comparisons = {
        "core_vs_full": compare(pooled_full, pooled_core, cfg, 101),
        "core_vs_isvd": compare(pooled_isvd, pooled_core, cfg, 102),
        "isvd_vs_full": compare(pooled_full, pooled_isvd, cfg, 103),
    }
    primary_pass = pooled_comparisons["core_vs_full"]["passes_noninferiority"]
    decision = "COREFLOW_Q185_PRIMARY_NONINFERIOR" if primary_pass else "COREFLOW_Q185_PRIMARY_QUALITY_RISK_OR_FAIL"
    payload = {
        "status": "ANALYZED",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "primary_method": "CoreFlow q185 K5",
        "reference_method": "Original LoRA-Flow (full K5)",
        "fair_baseline": "Independent-SVD matched-q185 K5",
        "dataset": "ClassEval method-level formal split",
        "rows_per_seed": summaries["code_full_k5_seed41"]["rows"],
        "seeds": seeds,
        "summaries": summaries,
        "status_counts": {method: status_counts(result_root, method) for method in methods},
        "parse_mode_counts": {method: parse_counts(result_root, method) for method in methods},
        "by_seed": by_seed,
        "pooled": pooled_comparisons,
        "complete_hashes": complete_hashes,
        "interpretation_contract": {
            "qual_pass_and_core_pass": "claim may be extended to MBPP+ and ClassEval",
            "qual_fail_or_core_fail": "claim stays MBPP+-limited",
            "core_pass_but_svd_fail": "secondary difference reported without weakening primary claim",
        },
    }
    write_json(args.output, payload)

    lines = [
        "# CoreFlow independent confirmation: ClassEval method-level analysis",
        "",
        f"- decision: `{decision}`",
        f"- protocol: `{cfg['protocol_id']}`",
        f"- rows per seed: {payload['rows_per_seed']}",
        f"- seeds: {', '.join(map(str, seeds))}",
        "",
        "## Seed-level correct counts",
        "",
        "| seed | Full | CoreFlow q185 | Independent-SVD |",
        "|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        item = by_seed[str(seed)]
        lines.append(f"| {seed} | {item['full_correct']} | {item['core_q185_correct']} | {item['isvd_q185_correct']} |")
    lines.extend([
        "",
        "## Pooled paired comparisons",
        "",
        "| comparison | diff pp | 95% CI pp | noninferiority (-3pp) |",
        "|---|---:|---:|---|",
    ])
    for name, item in pooled_comparisons.items():
        ci = item["ci95_pp"]
        lines.append(
            f"| {name} | {item['candidate_minus_reference_pp']:.2f} | "
            f"[{ci[0]:.2f}, {ci[1]:.2f}] | {item['passes_noninferiority']} |"
        )
    lines.extend([
        "",
        "## Sensitivity (pooled CoreFlow - Full)",
        "",
        "| margin | CI lower >= margin |",
        "|---|---|",
    ])
    for label, passed in pooled_comparisons["core_vs_full"]["sensitivity"].items():
        lines.append(f"| -{label} pp | {passed} |")
    lines.append("")
    Path(args.summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_md).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "ANALYZED", "decision": decision, "primary_noninferior": primary_pass}, ensure_ascii=False))


if __name__ == "__main__":
    main()
