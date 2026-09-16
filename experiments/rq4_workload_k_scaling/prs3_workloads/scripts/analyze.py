#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.common import adapter_checkpoint, bank_path, config, expert_path, load_json, now_utc, paths, write_json


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values)) if values else None


def cv_pct(values):
    return 100.0 * statistics.stdev(values) / statistics.fmean(values) if len(values) > 1 and statistics.fmean(values) else None


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_geomean(values, samples, seed):
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(geometric_mean([values[rng.randrange(len(values))] for _ in values]))
    return {
        "estimate": geometric_mean(values),
        "diagnostic_ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "samples": samples,
        "seed": seed,
        "unit_count": len(values),
        "role": "pilot diagnostic only; not formal inferential evidence",
    }


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def linear_slope(xs, ys):
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator if denominator else None


def analyze_prs3a(cfg, p):
    root = p["results"] / "prs3a"
    launches = []
    for path in root.rglob("summary.json"):
        item = load_json(path)
        item["path"] = str(path.parent)
        launches.append(item)
    ooms = [load_json(path) for path in root.rglob("OOM.json")]
    partials = [str(path.parent) for path in root.rglob("PARTIAL.json")]
    expected = sum(len(item["orders"]) * 3 for item in cfg["prs3a"]["workloads"])
    grouped = defaultdict(list)
    by_block = defaultdict(dict)
    for item in launches:
        throughput = item["statistics"]["decode_tokens_per_second"]["median"]
        grouped[(item["workload"], item["method"])].append(item)
        by_block[(item["workload"], item["block_id"])][item["method"]] = throughput
    method_summary = {}
    for (workload, method), items in sorted(grouped.items()):
        medians = [value["statistics"]["decode_tokens_per_second"]["median"] for value in items]
        method_summary.setdefault(workload, {})[method] = {
            "complete_launches": len(items),
            "median_of_launch_medians_decode_tokens_per_second": statistics.median(medians),
            "launch_mean_decode_tokens_per_second": statistics.fmean(medians),
            "launch_cv_pct": cv_pct(medians),
            "median_end_to_end_latency_seconds": statistics.median(value["statistics"]["client_total_seconds"]["median"] for value in items),
            "median_ttft_seconds": statistics.median(value["statistics"]["server_ttft_seconds"]["median"] for value in items),
            "median_tpot_seconds": statistics.median(value["statistics"]["tpot_seconds"]["median"] for value in items),
            "peak_allocated_gib": max(value["statistics"]["peak_memory_allocated_bytes"] for value in items) / 2**30,
            "peak_reserved_gib": max(value["statistics"]["peak_memory_reserved_bytes"] for value in items) / 2**30,
            "startup_allocated_gib": max(value["statistics"]["startup_resident_allocated_bytes"] for value in items) / 2**30,
            "incremental_peak_allocated_gib": max(value["statistics"]["incremental_peak_allocated_bytes"] for value in items) / 2**30,
            "order_positions": sorted(value["order_position"] for value in items),
            "all_attention_mask_tokens_active": all(value.get("all_attention_mask_tokens_active") for value in items),
        }
    comparisons = {}
    for workload in [item["name"] for item in cfg["prs3a"]["workloads"]]:
        pairs = {"core_over_full": [], "core_over_isvd": []}
        for (name, _), methods in by_block.items():
            if name != workload:
                continue
            full = methods.get("full_legacy_original")
            core = methods.get("coreflow_q185_frozen")
            isvd = methods.get("isvd_legacy_padded_matched_q185")
            if core and full:
                pairs["core_over_full"].append(core / full)
            if core and isvd:
                pairs["core_over_isvd"].append(core / isvd)
        comparisons[workload] = {
            name: {"block_ratios": values, "geometric_mean_ratio": geometric_mean(values)}
            for name, values in pairs.items()
        }
    anchor = cfg["prs3a"]["anchor"]
    for index, name in enumerate(("core_over_full", "core_over_isvd")):
        values = comparisons[anchor][name]["block_ratios"]
        if values:
            comparisons[anchor][name]["pilot_bootstrap"] = bootstrap_geomean(
                values,
                int(cfg["prs3a"]["bootstrap_samples"]),
                int(cfg["prs3a"]["bootstrap_seed"]) + index,
            )
    def throughput(workload, method):
        return method_summary.get(workload, {}).get(method, {}).get("median_of_launch_medians_decode_tokens_per_second")
    sensitivity = {"context": {}, "decode_length": {}, "batch_scaling_efficiency": {}}
    for method in cfg["methods"]:
        anchor_value = throughput(anchor, method)
        sensitivity["context"][method] = {
            name: (throughput(name, method) / anchor_value if throughput(name, method) and anchor_value else None)
            for name in ("w0_b1_l128_o64", anchor, "w3_b1_l2048_o128")
        }
        sensitivity["decode_length"][method] = {
            "w4_over_w1": throughput("w4_b1_l512_o256", method) / anchor_value if throughput("w4_b1_l512_o256", method) and anchor_value else None
        }
        sensitivity["batch_scaling_efficiency"][method] = {
            "b4": throughput("w5_b4_l512_o128", method) / (4 * anchor_value) if throughput("w5_b4_l512_o128", method) and anchor_value else None,
            "b8": throughput("w6_b8_l512_o128", method) / (8 * anchor_value) if throughput("w6_b8_l512_o128", method) and anchor_value else None,
        }
    coverage = {
        "decode_tokens_per_second": bool(launches),
        "geometric_ratios": any(value["core_over_full"]["block_ratios"] for value in comparisons.values()),
        "end_to_end_latency": bool(launches),
        "ttft": bool(launches),
        "tpot": bool(launches),
        "peak_allocated": bool(launches),
        "peak_reserved": bool(launches),
        "startup_allocated": bool(launches),
        "incremental_peak": bool(launches),
        "launch_cv_anchor": all(method_summary.get(anchor, {}).get(method, {}).get("launch_cv_pct") is not None for method in cfg["methods"]),
        "oom_status": True,
        "batch_scaling_efficiency": True,
        "context_sensitivity": True,
        "decode_length_sensitivity": True,
    }
    return {
        "status": "ANALYZED_PILOT_ONLY",
        "expected_launches": expected,
        "complete_launches": len(launches),
        "recorded_oom": ooms,
        "partial_outputs": partials,
        "method_summary": method_summary,
        "comparisons": comparisons,
        "sensitivity": sensitivity,
        "metric_coverage": coverage,
        "all_metrics_covered": all(coverage.values()),
    }


def analyze_prs3b(cfg, p):
    root = p["results"] / "prs3b"
    rows = []
    for path in root.rglob("measurements.jsonl"):
        rows.extend(load_jsonl(path))
    ooms = [load_json(path) for path in root.rglob("OOM.json")]
    partials = [str(path.parent) for path in root.rglob("PARTIAL.json")]
    main_rows = [row for row in rows if not row["sentinel_only"]]
    grouped = defaultdict(list)
    for row in main_rows:
        grouped[(row["pool"], row["method"], int(row["token_block"]))].append(row)
    aggregate = {}
    for (pool, method, token_block), values in sorted(grouped.items()):
        aggregate.setdefault(pool, {}).setdefault(method, {})[str(token_block)] = {
            "module_count": len(values),
            "adapter_wall_time_us_seven_module_sum": sum(row["wall_microseconds"]["median"] for row in values),
            "adapter_cuda_time_us_seven_module_sum": sum(row["cuda_event_microseconds"]["median"] for row in values),
            "representative_resident_bytes_sum": sum(row["resident_selected_module_bytes"] for row in values),
            "representative_logical_elements_sum": sum(row["logical_tensor_elements"] for row in values),
            "representative_stored_elements_sum": sum(row["stored_tensor_elements"] for row in values),
            "representative_executed_elements_sum": sum(row["executed_tensor_elements"] for row in values),
            "peak_allocated_bytes_max": max(row["absolute_peak_allocated_bytes"] for row in values),
            "peak_reserved_bytes_max": max(row["absolute_peak_reserved_bytes"] for row in values),
            "incremental_peak_allocated_bytes_max": max(row["incremental_peak_allocated_bytes"] for row in values),
            "logical_adapter_scope_invocations": sum(row["adapter_scope_invocations_measured"] for row in values),
            "logical_expert_branches_per_seven_module_invocation": sum(row["logical_expert_branches_per_invocation"] for row in values),
            "logical_matmuls_per_seven_module_invocation": sum(row["logical_matmuls_per_invocation"] for row in values),
            "operator_calls": sum((row["profiler"] or {}).get("operator_calls", 0) for row in values),
            "cuda_kernel_events": sum((row["profiler"] or {}).get("cuda_kernel_events", 0) for row in values),
            "adapter_scope_cuda_time_us": sum((row["profiler"] or {}).get("adapter_scope_cuda_time_us", 0.0) for row in values),
            "profiled_modules": sum(row["profiler"] is not None for row in values),
        }
    ratios = {}
    for pool, methods in aggregate.items():
        ratios[pool] = {}
        for token_block in map(str, cfg["prs3b"]["token_blocks"]):
            if not all(token_block in methods.get(method, {}) for method in cfg["methods"]):
                continue
            full = methods["full_legacy_original"][token_block]
            core = methods["coreflow_q185_frozen"][token_block]
            isvd = methods["isvd_legacy_padded_matched_q185"][token_block]
            ratios[pool][token_block] = {
                "core_over_full_time_ratio": core["adapter_cuda_time_us_seven_module_sum"] / full["adapter_cuda_time_us_seven_module_sum"],
                "core_over_isvd_time_ratio": core["adapter_cuda_time_us_seven_module_sum"] / isvd["adapter_cuda_time_us_seven_module_sum"],
                "core_speedup_over_full": full["adapter_cuda_time_us_seven_module_sum"] / core["adapter_cuda_time_us_seven_module_sum"],
                "core_speedup_over_isvd": isvd["adapter_cuda_time_us_seven_module_sum"] / core["adapter_cuda_time_us_seven_module_sum"],
            }
    bank_lock = load_json(ROOT / "evidence" / "m3_bank_lock.json")
    static = {}
    for pool, experts in cfg["prs3b"]["pools"].items():
        item = bank_lock["pools"][pool]
        full_artifact = sum(adapter_checkpoint(expert_path(cfg, p, name)).stat().st_size for name in experts)
        core_root = bank_path(cfg, p, pool, "core")
        isvd_root = bank_path(cfg, p, pool, "isvd")
        static[pool] = {
            "k": len(experts),
            "full_logical_parameters": item["source_full_parameters"],
            "core_logical_parameters": item["coreflow"]["efficiency"]["coreflow_resident_adapter_parameters"],
            "isvd_logical_parameters": item["isvd"]["budget"]["used_parameters"],
            "full_effective_bf16_payload_gib": item["source_full_parameters"] * 2 / 2**30,
            "core_effective_bf16_payload_gib": item["coreflow"]["efficiency"]["coreflow_resident_adapter_parameters"] * 2 / 2**30,
            "isvd_effective_bf16_payload_gib": item["isvd"]["budget"]["used_parameters"] * 2 / 2**30,
            "full_artifact_gib": full_artifact / 2**30,
            "core_artifact_gib": (core_root / "core_bank.safetensors").stat().st_size / 2**30,
            "isvd_artifact_gib": (isvd_root / "isvd_bank.safetensors").stat().st_size / 2**30,
            "core_parameter_reduction_pct_vs_full": 100 * (1 - item["coreflow"]["efficiency"]["coreflow_resident_adapter_parameters"] / item["source_full_parameters"]),
        }
    slopes = {}
    for method in cfg["methods"]:
        slopes[method] = {}
        for token_block in map(str, cfg["prs3b"]["token_blocks"]):
            points = []
            for pool in cfg["prs3b"]["pools"]:
                value = aggregate.get(pool, {}).get(method, {}).get(token_block)
                if value:
                    points.append((static[pool]["k"], value["adapter_cuda_time_us_seven_module_sum"]))
            slopes[method][token_block] = linear_slope([x for x, _ in points], [y for _, y in points]) if len(points) == 3 else None
    memory_slopes = {}
    for method in cfg["methods"]:
        low = aggregate.get("k3_strict", {}).get(method, {}).get("128")
        high = aggregate.get("k8_strict", {}).get(method, {}).get("128")
        memory_slopes[method] = (high["representative_resident_bytes_sum"] - low["representative_resident_bytes_sum"]) / 5 if low and high else None
    sentinel = defaultdict(dict)
    for row in rows:
        if row["pool"] == cfg["prs3b"]["cv_sentinel_pool"] and int(row["token_block"]) == int(cfg["prs3b"]["cv_sentinel_token_block"]):
            sentinel[(row["method"], int(row["launch"]))][row["module"]] = row["cuda_event_microseconds"]["median"]
    cv = {}
    for method in cfg["methods"]:
        launch_totals = []
        for launch in range(1, int(cfg["prs3b"]["cv_total_launches"]) + 1):
            modules = sentinel.get((method, launch), {})
            if len(modules) == len(cfg["prs3b"]["modules"]):
                launch_totals.append(sum(modules.values()))
        cv[method] = {"launch_totals_cuda_us": launch_totals, "independent_launch_cv_pct": cv_pct(launch_totals)}
    expected_main_rows = len(cfg["prs3b"]["pools"]) * len(cfg["methods"]) * len(cfg["prs3b"]["token_blocks"]) * len(cfg["prs3b"]["modules"])
    profiler_rows = [row for row in main_rows if row["profiler"] is not None]
    coverage = {
        "logical_parameters": bool(static),
        "effective_bf16_payload": bool(static),
        "artifact_gib": bool(static),
        "adapter_wall_time": bool(main_rows),
        "cuda_event_time": bool(main_rows),
        "core_full_time_ratio": bool(ratios),
        "core_isvd_time_ratio": bool(ratios),
        "peak_allocated_reserved": bool(main_rows),
        "executed_tensor_elements": bool(main_rows),
        "adapter_call_count": bool(main_rows),
        "cuda_kernel_events": bool(profiler_rows) and all((row["profiler"] or {}).get("cuda_kernel_events", 0) > 0 for row in profiler_rows),
        "k_memory_slope": all(value is not None for value in memory_slopes.values()),
        "k_runtime_slope": all(value is not None for method in slopes.values() for value in method.values()),
        "token_block_interaction": all(len(value) == len(cfg["prs3b"]["token_blocks"]) for value in ratios.values()),
        "independent_launch_cv": all(item["independent_launch_cv_pct"] is not None for item in cv.values()),
    }
    return {
        "status": "ANALYZED_PILOT_ONLY",
        "main_rows_expected": expected_main_rows,
        "main_rows_observed": len(main_rows),
        "all_rows_observed": len(rows),
        "recorded_oom": ooms,
        "partial_outputs": partials,
        "static_resources": static,
        "aggregate": aggregate,
        "time_ratios": ratios,
        "k_runtime_slope_cuda_us_per_added_expert": slopes,
        "k_memory_slope_bytes_per_added_expert_at_t128": memory_slopes,
        "independent_launch_cv": cv,
        "profiler_rows": len(profiler_rows),
        "metric_coverage": coverage,
        "all_metrics_covered": all(coverage.values()),
    }


def write_tables(p, prs3a, prs3b):
    table_root = p["work"] / "tables" / "prs3_pilot_v1"
    table_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for workload, methods in prs3a["method_summary"].items():
        for method, values in methods.items():
            rows.append({"workload": workload, "method": method, **values})
    if rows:
        with (table_root / "prs3a_method_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    rows = []
    for pool, methods in prs3b["aggregate"].items():
        for method, blocks in methods.items():
            for token_block, values in blocks.items():
                rows.append({"pool": pool, "method": method, "token_block": token_block, **values})
    if rows:
        fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
        with (table_root / "prs3b_aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def main():
    cfg, p = config(), paths()
    if not (p["reports"] / "prs3a_complete.json").is_file() or not (p["reports"] / "prs3b_complete.json").is_file():
        raise RuntimeError("Run PRS-3A and PRS-3B first")
    prs3a = analyze_prs3a(cfg, p)
    prs3b = analyze_prs3b(cfg, p)
    anchor_ratio = prs3a["comparisons"].get(cfg["prs3a"]["anchor"], {}).get("core_over_full", {}).get("geometric_mean_ratio")
    integrity = (
        not prs3a["partial_outputs"]
        and not prs3b["partial_outputs"]
        and prs3a["all_metrics_covered"]
        and prs3b["all_metrics_covered"]
    )
    if not integrity:
        decision = "REVISE_MEASUREMENT"
    elif anchor_ratio is not None and anchor_ratio <= 1.0:
        decision = "TREND_CONFLICT"
    elif prs3a["recorded_oom"] or prs3b["recorded_oom"]:
        decision = "READY_FOR_FORMAL_PRS3_WITH_CAPACITY_LIMIT_OBSERVED"
    else:
        decision = "READY_FOR_FORMAL_PRS3"
    payload = {
        "status": "ANALYZED_PILOT_ONLY",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "decision": decision,
        "pilot_only_not_formal_evidence": True,
        "prs3a": prs3a,
        "prs3b": prs3b,
        "interpretation_limits": cfg["claims_forbidden"],
    }
    write_json(p["reports"] / "final" / "prs3_pilot_summary.json", payload)
    write_tables(p, prs3a, prs3b)
    print(json.dumps({"status": payload["status"], "decision": decision, "prs3a_metrics": prs3a["all_metrics_covered"], "prs3b_metrics": prs3b["all_metrics_covered"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
