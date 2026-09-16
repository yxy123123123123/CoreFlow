#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {
    "A0_bilateral_normed_joint": "core_q185",
    "A1_bilateral_no_norm": "A1_bilateral_no_norm",
    "A2_input_only": "A2_input_only",
    "A3_output_only": "A3_output_only",
    "A4_plain_concat_svd": "A4_plain_concat_svd",
    "A5_gate_frequency_weighted": "A5_gate_frequency_weighted",
    "A6_random_orthogonal": "A6_random_orthogonal",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def params_of(efficiency: dict):
    for key in ("coreflow_resident_adapter_parameters", "variant_resident_adapter_parameters"):
        if key in efficiency:
            return int(efficiency[key])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    summary, weights = load(args.summary), load(args.weights)
    weight_by_label = {item["label"]: item for item in weights["methods"]}
    a0_seed41 = next(item for item in summary["quality"]["s6"] if item["method"] == "core_q185")["per_seed"][0]
    if a0_seed41["seed"] != 41:
        raise ValueError("Frozen A0 seed41 record is not first in the Phase-2B summary")
    a0_correct, a0_rows = int(a0_seed41["candidate_correct"]), int(a0_seed41["problems"])
    quality = {
        "A0_bilateral_normed_joint": {"correct": a0_correct, "rows": a0_rows, "pass_at_1": a0_correct / a0_rows,
                                         "difference_vs_a0_pp": 0.0, "ci95_lower_pp": 0.0, "ci95_upper_pp": 0.0,
                                         "quality_role": "frozen_development_anchor_seed41"}
    }
    for item in summary["quality"]["s7"]:
        name = item["method"].removesuffix("_seed41")
        quality[name] = {
            "correct": int(round((a0_correct / a0_rows + item["difference_pp"] / 100) * a0_rows)),
            "rows": item["problems"],
            "pass_at_1": a0_correct / a0_rows + item["difference_pp"] / 100,
            "difference_vs_a0_pp": item["difference_pp"],
            "ci95_lower_pp": item["ci95_lower_pp"],
            "ci95_upper_pp": item["ci95_upper_pp"],
            "quality_role": "single_gate_seed41_development_ablation",
        }
    system_by_label = {item["method"]: item for item in summary["system_anchor"]["s7"]}
    system_by_label["core_q185"] = next(item for item in summary["system_anchor"]["s6"] if item["method"] == "core_q185")
    adapter_by_label = {item["method"]: item for item in summary["adapter_path"]["s7"]}
    adapter_by_label["core_q185"] = [item for item in summary["adapter_path"]["s6"] if item["method"] == "core_q185"]
    rows = []
    for variant, label in LABELS.items():
        metric = weight_by_label[label]
        q = quality.get(variant)
        system = system_by_label.get(label)
        adapter = adapter_by_label.get(label, [])
        if isinstance(adapter, dict):
            adapter = [adapter]
        efficiency = metric.get("efficiency") or {}
        rows.append({
            "variant": variant,
            "dev_correct": None if q is None else q["correct"],
            "dev_rows": None if q is None else q["rows"],
            "dev_pass_at_1": None if q is None else q["pass_at_1"],
            "difference_vs_a0_pp": None if q is None else q["difference_vs_a0_pp"],
            "ci95_lower_pp": None if q is None else q["ci95_lower_pp"],
            "ci95_upper_pp": None if q is None else q["ci95_upper_pp"],
            "quality_role": "not_measured_by_design" if q is None else q["quality_role"],
            "fused_error_mean": metric["fused_relative_output_error"]["mean"],
            "fused_error_p95": metric["fused_relative_output_error"]["p95"],
            "fused_error_worst": metric["fused_relative_output_error"]["worst"],
            "per_expert_error_mean": metric["per_expert_relative_output_error"]["mean"],
            "per_expert_error_p95": metric["per_expert_relative_output_error"]["p95"],
            "per_expert_error_worst": metric["per_expert_relative_output_error"]["worst"],
            "logical_parameters": params_of(efficiency),
            "adapter_macs_per_token": params_of(efficiency),
            "historical_system_speedup_vs_full_vectorized": None if system is None else system["median_speedup"],
            "historical_adapter_time_ratio_token1": next((x["candidate_over_full_time_ratio"] for x in adapter if x["token_block"] == 1), None),
            "historical_adapter_time_ratio_token128": next((x["candidate_over_full_time_ratio"] for x in adapter if x["token_block"] == 128), None),
            "historical_adapter_time_ratio_token512": next((x["candidate_over_full_time_ratio"] for x in adapter if x["token_block"] == 512), None),
        })
    payload = {
        "status": "LOCAL_EVIDENCE_INTEGRATED_AWAITING_WEIGHT_AUDIT",
        "protocol_id": "coreflow-prs5-weight-audit-v1",
        "source_protocol_id": summary["protocol_id"],
        "selection_forbidden": True,
        "main_method_change_forbidden": True,
        "historical_system_scope": "exploratory Phase-2B comparison against Full-vectorized; not formal original-Full paper evidence",
        "rows": rows,
    }
    write_json(args.out_json, payload)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": payload["status"], "rows": len(rows)}))


if __name__ == "__main__":
    main()
