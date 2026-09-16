#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.data_audit import load_jsonl
from coreflow.gate_io import load_json, write_json
from coreflow.statistics import paired_bootstrap_difference, paired_transitions, quantile


def load_run(root: Path, name: str) -> dict:
    complete = load_json(root / name / "COMPLETE.json")
    summary = load_json(root / name / "summary.json")
    rows = load_jsonl(root / name / "executor_results.jsonl")
    if complete.get("status") != "PASS" or len(rows) != int(complete["row_count"]):
        raise ValueError(f"Incomplete run: {name}")
    outcomes = {row["task_id"]: bool(row["evaluation"]["passed"]) for row in rows}
    if len(outcomes) != len(rows):
        raise ValueError(f"Duplicate task IDs in run: {name}")
    return {"complete": complete, "summary": summary, "outcomes": outcomes}


def paired_compare(reference: dict, candidate: dict, cfg: dict, seed_offset: int = 0) -> dict:
    if set(reference["outcomes"]) != set(candidate["outcomes"]):
        raise ValueError("Paired task IDs differ")
    ids = sorted(reference["outcomes"])
    ref = [reference["outcomes"][task_id] for task_id in ids]
    cand = [candidate["outcomes"][task_id] for task_id in ids]
    ref_rate = sum(ref) / len(ref)
    cand_rate = sum(cand) / len(cand)
    boot = paired_bootstrap_difference(
        ref,
        cand,
        samples=int(cfg["hypotheses"]["paired_cluster_bootstrap_samples"]),
        confidence=float(cfg["hypotheses"]["bootstrap_confidence"]),
        seed=int(cfg["hypotheses"]["bootstrap_seed"]) + seed_offset,
    )
    retention = None if ref_rate == 0 else cand_rate / ref_rate
    margin = float(cfg["hypotheses"]["noninferiority_margin_pp"]) / 100.0
    checks = {
        "nonzero_reference": ref_rate > 0,
        "point_difference": cand_rate - ref_rate >= -margin,
        "ci_lower_bound": boot["lower"] >= -margin,
        "retention": retention is not None and retention >= float(cfg["hypotheses"]["minimum_retention_ratio"]),
    }
    return {
        "rows": len(ids),
        "reference_correct": sum(ref),
        "candidate_correct": sum(cand),
        "reference_rate": ref_rate,
        "candidate_rate": cand_rate,
        "difference_pp": (cand_rate - ref_rate) * 100.0,
        "retention_ratio": retention,
        "paired_bootstrap_ci_pp": {"lower": boot["lower"] * 100.0, "upper": boot["upper"] * 100.0},
        "transitions": paired_transitions(reference["outcomes"], candidate["outcomes"]),
        "checks": checks,
        "pass": all(checks.values()),
    }


def cluster_compare(full_runs: dict[int, dict], core_runs: dict[int, dict], cfg: dict, seed_offset: int) -> dict:
    seeds = sorted(full_runs)
    if seeds != sorted(core_runs):
        raise ValueError("Full/Core seed sets differ")
    ids = sorted(full_runs[seeds[0]]["outcomes"])
    for run in [*full_runs.values(), *core_runs.values()]:
        if sorted(run["outcomes"]) != ids:
            raise ValueError("Task IDs differ across primary runs")
    item_differences = [
        sum(
            int(core_runs[seed]["outcomes"][task_id]) - int(full_runs[seed]["outcomes"][task_id])
            for seed in seeds
        ) / len(seeds)
        for task_id in ids
    ]
    full_correct = sum(sum(run["outcomes"].values()) for run in full_runs.values())
    core_correct = sum(sum(run["outcomes"].values()) for run in core_runs.values())
    denominator = len(ids) * len(seeds)
    full_rate = full_correct / denominator
    core_rate = core_correct / denominator
    rng = random.Random(int(cfg["hypotheses"]["bootstrap_seed"]) + seed_offset)
    samples = int(cfg["hypotheses"]["paired_cluster_bootstrap_samples"])
    estimates = [
        sum(item_differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
        for _ in range(samples)
    ]
    alpha = (1.0 - float(cfg["hypotheses"]["bootstrap_confidence"])) / 2.0
    lower = quantile(estimates, alpha)
    upper = quantile(estimates, 1.0 - alpha)
    retention = None if full_rate == 0 else core_rate / full_rate
    margin = float(cfg["hypotheses"]["noninferiority_margin_pp"]) / 100.0
    checks = {
        "nonzero_reference": full_rate > 0,
        "point_difference": core_rate - full_rate >= -margin,
        "cluster_ci_lower_bound": lower >= -margin,
        "retention": retention is not None and retention >= float(cfg["hypotheses"]["minimum_retention_ratio"]),
    }
    by_seed = {
        str(seed): paired_compare(full_runs[seed], core_runs[seed], cfg, seed_offset + seed)
        for seed in seeds
    }
    return {
        "task_clusters": len(ids),
        "seeds": seeds,
        "binary_observations": denominator,
        "full_correct": full_correct,
        "core_correct": core_correct,
        "full_rate": full_rate,
        "core_rate": core_rate,
        "difference_pp": (core_rate - full_rate) * 100.0,
        "retention_ratio": retention,
        "cluster_paired_bootstrap_ci_pp": {"lower": lower * 100.0, "upper": upper * 100.0},
        "checks": checks,
        "by_seed": by_seed,
        "pass": all(checks.values()),
    }


def analyze_legacy(args, cfg: dict) -> dict:
    root = Path(args.result_root)
    full = load_run(root, "math_full_k5_seed42")
    core = load_run(root, "math_core_q185_k5_seed42")
    return {
        "status": "ANALYZED",
        "evidence_label": "F0_LEGACY_PAIR_COMPLETION",
        "comparison": paired_compare(full, core, cfg, 4200),
        "note": "This completes the previously missing M2 math K5 seed42 pair; it is not part of formal-v1 H1.",
    }


def analyze_smoke(args, cfg: dict) -> dict:
    root = Path(args.result_root)
    full = load_run(root, "code_full_k5_seed41")
    core = load_run(root, "code_core_q185_k5_seed41")
    if full["summary"]["rows"] != 8 or core["summary"]["rows"] != 8:
        raise ValueError("Smoke runs must each contain exactly 8 rows")
    return {
        "status": "PASS",
        "evidence_label": "IMPLEMENTATION_SMOKE_ONLY",
        "full_summary": full["summary"],
        "core_summary": core["summary"],
        "quality_not_used_in_formal_analysis": True,
    }


def analyze_floor(args, cfg: dict) -> dict:
    run = load_run(Path(args.result_root), cfg["code_floor"]["method"])
    correct = int(run["summary"]["correct"])
    if correct >= int(cfg["code_floor"]["confirmatory_min_correct"]):
        role = "CODE_CONFIRMATORY"
    elif correct >= int(cfg["code_floor"]["supportive_min_correct"]):
        role = "CODE_SUPPORTIVE"
    else:
        role = "CODE_STRESS_ONLY"
    return {
        "status": "PASS",
        "decision": role,
        "correct": correct,
        "rows": int(run["summary"]["rows"]),
        "formal_code_may_open": role != "CODE_STRESS_ONLY",
        "decision_is_irreversible_after_formal_code_outputs": True,
    }


def primary_task(root: Path, task: str, cfg: dict, seed_offset: int) -> dict:
    seeds = [41, 42, 43]
    full = {seed: load_run(root, f"{task}_full_k5_seed{seed}") for seed in seeds}
    core = {seed: load_run(root, f"{task}_core_q185_k5_seed{seed}") for seed in seeds}
    result = cluster_compare(full, core, cfg, seed_offset)
    result["full_correct_by_seed"] = {str(seed): full[seed]["summary"]["correct"] for seed in seeds}
    result["full_summary_by_seed"] = {str(seed): full[seed]["summary"] for seed in seeds}
    result["core_summary_by_seed"] = {str(seed): core[seed]["summary"] for seed in seeds}
    asset_lock = load_json(ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json")
    result["theoretical_efficiency"] = asset_lock["banks"][f"k5_{task}"]["185"]["efficiency"]
    if task == "code":
        floor = int(cfg["hypotheses"]["code_formal_full_k5_floor_correct"])
        result["formal_full_floor_correct"] = floor
        result["formal_full_floor_pass"] = all(int(full[seed]["summary"]["correct"]) >= floor for seed in seeds)
        result["pass"] = bool(result["pass"] and result["formal_full_floor_pass"])
    return result


def analyze_primary(args, cfg: dict) -> dict:
    root = Path(args.result_root)
    floor = load_json(args.floor_decision)
    math = primary_task(root / "math_main", "math", cfg, 10000)
    code = None
    if floor["decision"] != "CODE_STRESS_ONLY":
        code = primary_task(root / "code_main", "code", cfg, 20000)
    q224_tasks = []
    if not math["pass"]:
        q224_tasks.append("math")
    if code is not None and not code["pass"]:
        q224_tasks.append("code")
    status = "PASS_PRIMARY_Q185" if not q224_tasks else "Q185_PRIMARY_INCONCLUSIVE_OR_FAIL"
    return {
        "status": status,
        "quality_status": status,
        "efficiency_status": "EFFICIENCY_PENDING",
        "code_floor_decision": floor,
        "math_h1": math,
        "code_h1": code,
        "code_formal_was_kept_closed": floor["decision"] == "CODE_STRESS_ONLY",
        "q224_decision_required_tasks": q224_tasks,
        "q224_has_not_been_read_or_run_by_this_analysis": True,
    }


def analyze_secondary(args, cfg: dict) -> dict:
    result_root = Path(args.result_root)
    primary = load_json(args.primary_decision)
    code_closed = bool(primary.get("code_formal_was_kept_closed"))
    extension_root = result_root / "k_extension"
    comparisons = {}
    for task, k in (("code", 2), ("code", 3), ("math", 2)):
        if task == "code" and code_closed:
            comparisons[f"{task}_k{k}"] = {"status": "NOT_RUN_CODE_FORMAL_CLOSED"}
            continue
        full = load_run(extension_root, f"{task}_full_k{k}_seed41")
        core = load_run(extension_root, f"{task}_core_q185_k{k}_seed41")
        comparison = paired_compare(full, core, cfg, 30000 + k)
        asset_lock = load_json(ROOT / "provenance" / "EXPECTED_ASSET_LOCK.json")
        comparison["theoretical_efficiency"] = asset_lock["banks"][f"k{k}_{task}"]["185"]["efficiency"]
        comparisons[f"{task}_k{k}"] = comparison
    baseline_root = result_root / "baselines"
    baselines = {}
    for name in cfg["phase_orders"]["baselines"]:
        if name.startswith("code_") and code_closed:
            baselines[name] = {"status": "NOT_RUN_CODE_FORMAL_CLOSED"}
            continue
        baselines[name] = load_run(baseline_root, name)["summary"]
    micro = load_json(args.microbenchmark)
    if micro.get("status") != "PASS":
        raise ValueError("Microbenchmark is incomplete")
    q224 = None
    if args.q224_activation:
        activation = load_json(args.q224_activation)
        qroot = result_root / "q224"
        q224 = {"activation": activation, "comparisons": {}}
        for task in activation["tasks"]:
            full = load_run(result_root / f"{task}_main", f"{task}_full_k5_seed41")
            core = load_run(qroot, f"{task}_core_q224_k5_seed41")
            q224["comparisons"][task] = paired_compare(full, core, cfg, 40000)
    q224_pending = bool(primary.get("q224_decision_required_tasks")) and not args.q224_activation
    return {
        "status": "Q224_DECISION_PENDING" if q224_pending else "PASS",
        "k_extension": comparisons,
        "baselines": baselines,
        "microbenchmark": micro,
        "q224": q224,
        "q224_decision_pending_tasks": primary.get("q224_decision_required_tasks") if q224_pending else [],
    }


def write_markdown(path: Path, payload: dict, mode: str) -> None:
    lines = [
        f"# CoreFlow formal-v1 {mode} analysis",
        "",
        f"- Status: `{payload['status']}`",
        f"- Protocol: `coreflow-formal-v1.1`",
        "",
        "This file is generated from locked machine-readable outputs. See the adjacent JSON for full details.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("legacy", "smoke", "floor", "primary", "secondary"))
    parser.add_argument("--config", default=str(ROOT / "config" / "formal_protocol.json"))
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--floor-decision")
    parser.add_argument("--primary-decision")
    parser.add_argument("--microbenchmark")
    parser.add_argument("--q224-activation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.mode == "legacy":
        payload = analyze_legacy(args, cfg)
    elif args.mode == "smoke":
        payload = analyze_smoke(args, cfg)
    elif args.mode == "floor":
        payload = analyze_floor(args, cfg)
    elif args.mode == "primary":
        if not args.floor_decision:
            raise ValueError("primary requires --floor-decision")
        payload = analyze_primary(args, cfg)
    else:
        if not args.microbenchmark or not args.primary_decision:
            raise ValueError("secondary requires --microbenchmark and --primary-decision")
        payload = analyze_secondary(args, cfg)
    write_json(args.output, payload)
    write_markdown(Path(args.summary_md), payload, args.mode)
    print(json.dumps({"status": payload["status"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
