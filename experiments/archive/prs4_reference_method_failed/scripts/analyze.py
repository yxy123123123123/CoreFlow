#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.lcb_eval import extract_code
from scripts.common import config, load_json, load_jsonl, now_utc, paths, write_json


def percentile(values: list[int], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    cfg, p = config(), paths()
    completion = p["reports"] / "pilot_b_complete.json"
    closure_path = p["reports"] / "scheme_a_legacy_qualification_closure.json"
    if not completion.is_file() or not closure_path.is_file():
        raise RuntimeError("Complete scheme A and pilot B before analyze")

    long_rows = []
    by_method = {}
    per_question = {}
    system_errors = 0
    for method in cfg["methods"]:
        method_id = method["id"]
        root = p["results"] / "pilot_b" / method_id
        if not (root / "COMPLETE.json").is_file():
            raise FileNotFoundError(root / "COMPLETE.json")
        generations = load_jsonl(root / "generations.jsonl")
        evaluations = load_jsonl(root / "executor_results.jsonl")
        if len(generations) != cfg["data"]["pilot_rows"] or len(evaluations) != cfg["data"]["pilot_rows"]:
            raise ValueError(f"Incomplete rows for {method_id}")
        generation_by_id = {str(item["question_id"]): item for item in generations}
        evaluation_by_id = {str(item["question_id"]): item for item in evaluations}
        statuses, new_tokens, output_chars = Counter(), [], []
        correct = parsed = any_test = timeout = truncated = evaluator_ok = 0
        for question_id in cfg["data"]["pilot_question_ids"]:
            generation = generation_by_id[question_id]
            item = evaluation_by_id[question_id]
            evaluation = item["evaluation"]
            raw = generation["raw_output"]
            code, mode = extract_code(raw)
            try:
                ast.parse(code)
                parse_ok = True
            except SyntaxError:
                parse_ok = False
            status = evaluation.get("status", "MISSING")
            is_system_error = status in {"EVALUATOR_ERROR", "RUNTIME_OR_EVALUATOR_ERROR"}
            passed = bool(evaluation.get("passed"))
            passed_tests = int(evaluation.get("passed_tests", 0))
            token_count = int(generation.get("timing", {}).get("new_tokens", -1))
            reached_limit = bool(generation.get("reached_max_new_tokens"))
            statuses[status] += 1
            correct += int(passed)
            parsed += int(parse_ok)
            any_test += int(passed_tests > 0)
            timeout += int(status == "TIMEOUT")
            truncated += int(reached_limit)
            evaluator_ok += int(not is_system_error)
            system_errors += int(is_system_error)
            if token_count >= 0:
                new_tokens.append(token_count)
            output_chars.append(len(raw))
            row = {
                "question_id": question_id,
                "order_index": int(generation["order_index"]),
                "method": method_id,
                "seed": cfg["pilot_B"]["gate_seed"],
                "strict_pass": int(passed),
                "status": status,
                "parse_success": int(parse_ok),
                "passed_tests": passed_tests,
                "tests": int(evaluation.get("tests", 0)),
                "at_least_one_test_pass": int(passed_tests > 0),
                "timeout": int(status == "TIMEOUT"),
                "new_tokens": token_count,
                "reached_max_new_tokens": int(reached_limit),
                "raw_output_chars": len(raw),
                "raw_output_sha256": generation["raw_output_sha256"],
                "postprocess_mode": mode,
            }
            long_rows.append(row)
            per_question.setdefault(question_id, {})[method_id] = row
        by_method[method_id] = {
            "paper_name": method["paper_name"],
            "rows": len(generations),
            "strict_correct": correct,
            "strict_pass_at_1": correct / len(generations),
            "parse_success": parsed,
            "parse_success_rate": parsed / len(generations),
            "at_least_one_test_pass": any_test,
            "at_least_one_test_pass_rate": any_test / len(generations),
            "evaluator_execution_success": evaluator_ok,
            "timeout": timeout,
            "truncation_at_max_new_tokens": truncated,
            "status_counts": dict(statuses),
            "new_tokens": {
                "min": min(new_tokens) if new_tokens else None,
                "median": statistics.median(new_tokens) if new_tokens else None,
                "p95": percentile(new_tokens, 0.95),
                "max": max(new_tokens) if new_tokens else None,
            },
            "raw_output_chars": {
                "min": min(output_chars),
                "median": statistics.median(output_chars),
                "p95": percentile(output_chars, 0.95),
                "max": max(output_chars),
            },
        }

    pairs = [
        ("coreflow_q185_frozen", "full_legacy_original"),
        ("coreflow_q185_frozen", "isvd_legacy_padded_matched_q185"),
        ("isvd_legacy_padded_matched_q185", "full_legacy_original"),
    ]
    paired = {}
    for left, right in pairs:
        counts = Counter()
        exact_output = 0
        for question_id, methods in per_question.items():
            lrow, rrow = methods[left], methods[right]
            counts[f"{lrow['strict_pass']}{rrow['strict_pass']}"] += 1
            exact_output += int(lrow["raw_output_sha256"] == rrow["raw_output_sha256"])
        paired[f"{left}_vs_{right}"] = {
            "both_pass": counts["11"],
            "left_only_pass": counts["10"],
            "right_only_pass": counts["01"],
            "both_fail": counts["00"],
            "exact_raw_output_matches": exact_output,
            "strict_pass_rate_difference_pp": 100.0 * (by_method[left]["strict_pass_at_1"] - by_method[right]["strict_pass_at_1"]),
            "role": "descriptive pilot only; no inferential or qualification claim",
        }

    final_dir = p["reports"] / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(long_rows[0])
    with (final_dir / "pilot_b_long_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    pipeline_status = "METRIC_PIPELINE_VALID" if len(long_rows) == cfg["pilot_B"]["expected_records"] and system_errors == 0 else "METRIC_PIPELINE_REQUIRES_REPAIR"
    summary = {
        "status": "ANALYZED_EXPLORATORY_ONLY",
        "protocol_id": cfg["protocol_id"],
        "timestamp_utc": now_utc(),
        "scheme_a": load_json(closure_path),
        "scheme_b": {
            "pipeline_status": pipeline_status,
            "records": len(long_rows),
            "expected_records": cfg["pilot_B"]["expected_records"],
            "methods": by_method,
            "paired_descriptive": paired,
            "system_evaluator_errors": system_errors,
        },
        "formal164_branch": "CLOSED",
        "formal164_accessed": False,
        "prs4q_original_loraflow_status": "NOT_RUN",
        "claims_allowed": [
            "The previous Stage-1 full-vectorized 0/64 result was archived as a low-floor failure.",
            "The three-method first16 pilot did or did not validate the metric/evaluator pipeline, according to pipeline_status.",
            "Pilot outcomes may be reported only as exploratory diagnostics on an already-exposed subset."
        ],
        "claims_forbidden": cfg["forbidden_claims"],
    }
    write_json(final_dir / "prs4a_prs4p_summary.json", summary)

    lines = [
        "# PRS-4A / PRS-4P pilot result summary",
        "",
        f"- Pipeline status: `{pipeline_status}`",
        "- formal164: `CLOSED` and not accessed",
        "- PRS-4Q Original LoRA-Flow qualification: `NOT_RUN`",
        "- Evidence role: exploratory metric/evaluator pilot only",
        "",
        "## Three-method first16 results",
        "",
        "| Method | Strict | Any test passed | Parse success | Timeout | Max-token truncation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id in cfg["pilot_B"]["method_order"]:
        item = by_method[method_id]
        lines.append(
            f"| {method_id} | {item['strict_correct']}/{item['rows']} | "
            f"{item['at_least_one_test_pass']}/{item['rows']} | {item['parse_success']}/{item['rows']} | "
            f"{item['timeout']}/{item['rows']} | {item['truncation_at_max_new_tokens']}/{item['rows']} |"
        )
    lines.extend([
        "",
        "These 16 already-exposed problems do not qualify the dataset and cannot open formal164.",
    ])
    (final_dir / "prs4a_prs4p_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "pipeline_status": pipeline_status, "records": len(long_rows), "formal164": "CLOSED"}))


if __name__ == "__main__":
    main()
