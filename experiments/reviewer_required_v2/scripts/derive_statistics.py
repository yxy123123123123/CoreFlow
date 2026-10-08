import json
import math
import re
import statistics
import tarfile
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[1]
NEW = BASE / "_analysis_coreflow_reviewer_required_v2_20261007" / "coreflow-reviewer-required-v2"
OLD_MBPP = BASE / "_formal_v2_primary_results_20260729" / "results" / "formal_v2_primary" / "formal_code_main"
OLD_CLASS_TAR = BASE / "coreflow_indep_confirm_v1_results.tar.gz"
GROUP1 = BASE / "_analysis_joint_vs_posthoc_20260924" / "groups" / "group1"

DATASETS = ["AddSub", "AQuA", "gsm8k", "MultiArith", "SingleEq", "SVAMP"]
PRED_FILES = {
    "AddSub": "addsub_responses.jsonl",
    "AQuA": "aqua_responses.jsonl",
    "gsm8k": "gsm8k_responses.jsonl",
    "MultiArith": "multiarith_responses.jsonl",
    "SingleEq": "singleeq_responses.jsonl",
    "SVAMP": "svamp_responses.jsonl",
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def passed_map(path):
    return {row["task_id"]: bool(row["evaluation"]["passed"]) for row in read_jsonl(path)}


def tar_passed_map(member):
    with tarfile.open(OLD_CLASS_TAR, "r:gz") as archive:
        handle = archive.extractfile(member)
        assert handle is not None
        rows = [json.loads(line) for line in handle.read().decode("utf-8").splitlines() if line.strip()]
    return {row["task_id"]: bool(row["evaluation"]["passed"]) for row in rows}


def paired_summary(candidate_by_seed, reference_by_seed, seed=20260928, reps=50000):
    sampled_numerators = np.zeros(reps, dtype=np.int64)
    discordant = {"candidate_only": 0, "reference_only": 0}
    strata = []
    total = 0
    candidate_correct = 0
    reference_correct = 0
    for name in sorted(candidate_by_seed):
        candidate = candidate_by_seed[name]
        reference = reference_by_seed[name]
        ids = sorted(set(candidate) & set(reference))
        pairs = [(candidate[item], reference[item]) for item in ids]
        strata.append(pairs)
        total += len(pairs)
        candidate_correct += sum(a for a, _ in pairs)
        reference_correct += sum(b for _, b in pairs)
        discordant["candidate_only"] += sum(a and not b for a, b in pairs)
        discordant["reference_only"] += sum(b and not a for a, b in pairs)
    observed = (candidate_correct - reference_correct) / total
    rng = np.random.default_rng(seed)
    for pairs in strata:
        differences = np.asarray([int(a) - int(b) for a, b in pairs], dtype=np.int8)
        counts = np.asarray([
            np.count_nonzero(differences == -1),
            np.count_nonzero(differences == 0),
            np.count_nonzero(differences == 1),
        ])
        probabilities = counts / counts.sum()
        samples = rng.multinomial(len(pairs), probabilities, size=reps)
        sampled_numerators += samples[:, 2] - samples[:, 0]
    diffs = np.sort(sampled_numerators / total)
    low = float(diffs[int(0.025 * reps)])
    high = float(diffs[int(0.975 * reps) - 1])
    ndisc = discordant["candidate_only"] + discordant["reference_only"]
    if ndisc:
        k = min(discordant.values())
        exact_p = min(1.0, 2 * sum(math.comb(ndisc, i) for i in range(k + 1)) / (2 ** ndisc))
    else:
        exact_p = 1.0
    return {
        "candidate_correct": candidate_correct,
        "reference_correct": reference_correct,
        "rows": total,
        "difference_pp": observed * 100,
        "bootstrap_95_ci_pp": [low * 100, high * 100],
        "candidate_only": discordant["candidate_only"],
        "reference_only": discordant["reference_only"],
        "exact_mcnemar_p": exact_p,
        "bootstrap_repetitions": reps,
        "bootstrap_seed": seed,
    }


def q224_stats():
    output = {}
    q224_mbpp = {}
    full_mbpp = {}
    q185_mbpp = {}
    for seed in [41, 42, 43]:
        q224_mbpp[str(seed)] = passed_map(
            NEW / "results" / "coreflow-reviewer-required-v2" / "q224_mbpp"
            / f"code_core_q224_k5_seed{seed}" / "executor_results.jsonl"
        )
        full_mbpp[str(seed)] = passed_map(OLD_MBPP / f"code_full_k5_seed{seed}" / "executor_results.jsonl")
        q185_mbpp[str(seed)] = passed_map(OLD_MBPP / f"code_core_q185_k5_seed{seed}" / "executor_results.jsonl")
    output["mbpp_q224_vs_full"] = paired_summary(q224_mbpp, full_mbpp)
    output["mbpp_q224_vs_q185"] = paired_summary(q224_mbpp, q185_mbpp)

    q224_class = {}
    full_class = {}
    q185_class = {}
    for seed in [41, 42, 43]:
        q224_class[str(seed)] = passed_map(
            NEW / "results" / "coreflow-reviewer-required-v2" / "q224_classeval"
            / f"code_core_q224_k5_seed{seed}" / "executor_results.jsonl"
        )
        full_class[str(seed)] = tar_passed_map(
            f"results/indep_confirm_classeval/formal_main/code_full_k5_seed{seed}/executor_results.jsonl"
        )
        q185_class[str(seed)] = tar_passed_map(
            f"results/indep_confirm_classeval/formal_main/code_core_q185_k5_seed{seed}/executor_results.jsonl"
        )
    output["classeval_q224_vs_full"] = paired_summary(q224_class, full_class)
    output["classeval_q224_vs_q185"] = paired_summary(q224_class, q185_class)
    return output


def numeric_score(answer, text):
    values = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not values:
        return False
    try:
        return abs(float(answer) - float(values[-1])) <= 0.001
    except (TypeError, ValueError):
        return False


def aqua_score(answer, text, mode):
    if mode == "official_first_any":
        values = re.findall(r"A|B|C|D|E", text.strip())
        predicted = values[0] if values else ""
    elif mode == "robust_last_standalone":
        values = re.findall(r"\b([A-E])\b", text.upper())
        predicted = values[-1] if values else ""
    else:
        raise ValueError(mode)
    return predicted == str(answer).strip().upper()


def find_comol_prediction_root(group_root):
    matches = list((group_root / "comol_outputs").glob("**/predictions/addsub_responses.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CoMoL prediction root, found {len(matches)} under {group_root}")
    return matches[0].parent


def rescore_group(group_root, methods, mode):
    prediction_root = find_comol_prediction_root(group_root)
    ground_truth = {}
    comol_text = {}
    for dataset in DATASETS:
        rows = read_jsonl(prediction_root / PRED_FILES[dataset])
        ground_truth[dataset] = [row["answer"] for row in rows]
        comol_text[dataset] = [row["response"] for row in rows]

    results = {}
    vectors = {}
    for label, quality_dir in methods.items():
        per_dataset = {}
        vector = []
        for dataset in DATASETS:
            if label == "comol":
                texts = comol_text[dataset]
            else:
                rows = read_jsonl(quality_dir / f"{dataset}.jsonl")
                rows.sort(key=lambda row: row["index"])
                texts = [row["text"] for row in rows]
            answers = ground_truth[dataset]
            assert len(texts) == len(answers)
            current = []
            for answer, text in zip(answers, texts):
                ok = aqua_score(answer, text, mode) if dataset == "AQuA" else numeric_score(answer, text)
                current.append(ok)
                vector.append((dataset, ok))
            per_dataset[dataset] = {
                "correct": sum(current), "rows": len(current), "accuracy": sum(current) / len(current)
            }
        correct = sum(item["correct"] for item in per_dataset.values())
        rows = sum(item["rows"] for item in per_dataset.values())
        results[label] = {
            "correct": correct,
            "rows": rows,
            "micro_accuracy": correct / rows,
            "macro_accuracy": statistics.fmean(item["accuracy"] for item in per_dataset.values()),
            "datasets": per_dataset,
        }
        vectors[label] = vector
    return results, vectors


def vector_paired(candidate, reference, seed=20260928, reps=50000):
    by_dataset_candidate = {}
    by_dataset_reference = {}
    for dataset in DATASETS:
        c = [ok for ds, ok in candidate if ds == dataset]
        r = [ok for ds, ok in reference if ds == dataset]
        by_dataset_candidate[dataset] = {str(i): value for i, value in enumerate(c)}
        by_dataset_reference[dataset] = {str(i): value for i, value in enumerate(r)}
    return paired_summary(by_dataset_candidate, by_dataset_reference, seed=seed, reps=reps)


def qwen_rescore():
    groups = {
        "1": (
            GROUP1,
            {
                "full": GROUP1 / "quality" / "independent_full",
                "core_q16": GROUP1 / "quality" / "core_q16",
                "core_q24": GROUP1 / "quality" / "core_q24",
                "comol": None,
            },
        ),
        "2": (
            NEW / "groups" / "group2",
            {
                "full": NEW / "groups" / "group2" / "quality" / "independent_full",
                "core_q185_effective128": NEW / "groups" / "group2" / "quality" / "core_q185",
                "comol": None,
            },
        ),
        "3": (
            NEW / "groups" / "group3",
            {
                "full": NEW / "groups" / "group3" / "quality" / "independent_full",
                "core_q185_effective128": NEW / "groups" / "group3" / "quality" / "core_q185",
                "comol": None,
            },
        ),
    }
    output = {}
    for group, (root, methods) in groups.items():
        output[group] = {}
        for mode in ["official_first_any", "robust_last_standalone"]:
            scores, vectors = rescore_group(root, methods, mode)
            comparisons = {}
            for core in [label for label in methods if label.startswith("core_")]:
                comparisons[f"{core}_vs_full"] = vector_paired(vectors[core], vectors["full"])
                comparisons[f"{core}_vs_comol"] = vector_paired(vectors[core], vectors["comol"])
            comparisons["comol_vs_full"] = vector_paired(vectors["comol"], vectors["full"])
            output[group][mode] = {"scores": scores, "comparisons": comparisons}
    return output


def service_summary():
    root = NEW / "results" / "coreflow-reviewer-required-v2" / "service" / "RTX_4080_SUPER"
    output = {}
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = path.parent.name
        method = record["method"]
        output.setdefault(config, {})[method] = {
            "mean_decode_tokens_per_second": record["mean_decode_tokens_per_second"],
            "p50_latency_seconds": record["p50_latency_seconds"],
            "p95_latency_seconds": record["p95_latency_seconds"],
            "p99_latency_seconds": record["p99_latency_seconds"],
            "peak_allocated_gib": record["peak_allocated_gib"],
            "peak_reserved_gib": record["peak_reserved_gib"],
            "startup_health_ready_seconds": record["startup_health_ready_seconds"],
            "average_power_watts": record["average_power_watts"],
            "joules_per_request": record["joules_per_request"],
            "joules_per_token": record["joules_per_token"],
            "measurement_scope": record["measurement_scope"],
        }
    for config, values in output.items():
        core = values["core_q16"]
        for baseline in ["independent_full", "comol_native"]:
            base = values[baseline]
            core[f"speedup_vs_{baseline}"] = (
                core["mean_decode_tokens_per_second"] / base["mean_decode_tokens_per_second"]
            )
            core[f"p95_reduction_vs_{baseline}_pct"] = (
                1 - core["p95_latency_seconds"] / base["p95_latency_seconds"]
            ) * 100
            core[f"energy_reduction_vs_{baseline}_pct"] = (
                1 - core["joules_per_token"] / base["joules_per_token"]
            ) * 100
    return output


def drift_summary():
    record = json.loads((NEW / "reports" / "coreflow-reviewer-required-v2" / "drift_diagnostics.json").read_text(encoding="utf-8"))
    output = {}
    for label, comparison in record["comparisons"].items():
        layers = comparison["hidden_by_layer"]
        ids = sorted(layers, key=lambda item: int(item))
        rel = [layers[item]["relative_l2_mean"] for item in ids]
        cos = [layers[item]["cosine_mean"] for item in ids]
        gate = comparison["gate_by_layer"]
        gate_ids = sorted(gate, key=lambda item: int(item))
        output[label] = {
            "rows": comparison["rows"],
            "first_hidden_relative_l2": rel[0],
            "final_hidden_relative_l2": rel[-1],
            "max_hidden_relative_l2": max(rel),
            "final_hidden_cosine": cos[-1],
            "final_logits_kl_mean": comparison["final_logits_kl_mean"],
            "final_top5_exact_agreement": comparison["final_top5_exact_agreement"],
            "mean_gate_logit_l2": statistics.fmean(gate[item]["logit_l2_mean"] for item in gate_ids),
            "mean_gate_top1_agreement": statistics.fmean(gate[item]["top1_agreement_mean"] for item in gate_ids),
        }
    return output


def storage_theory_summary():
    reports = NEW / "reports" / "coreflow-reviewer-required-v2"
    storage = json.loads((reports / "storage_audit.json").read_text(encoding="utf-8"))
    theory = json.loads((reports / "theory_error_analysis.json").read_text(encoding="utf-8"))
    storage_out = []
    for item in storage["artifacts"]:
        storage_out.append({
            "label": item["label"],
            "exists": item["exists"],
            "stored_bytes": item.get("stored_bytes"),
            "modules": item.get("modules"),
            "efficiency": item.get("efficiency"),
        })
    theory_out = []
    for bank in theory["banks"]:
        modules = bank["sampled_module_errors"]
        theory_out.append({
            "label": bank["label"],
            "q": bank["q"],
            "module_count": bank["module_count"],
            "efficiency": bank["efficiency"],
            "mean_of_module_mean_sampled_relative_error": statistics.fmean(
                item["mean_sampled_relative_error"] for item in modules
            ),
            "max_sampled_relative_error": max(item["max_sampled_relative_error"] for item in modules),
        })
    return {"storage": storage_out, "theory": theory_out}


def main():
    output = {
        "status": "LOCALLY_DERIVED_FROM_RETAINED_OUTPUTS",
        "q224": q224_stats(),
        "qwen_common_rescore": qwen_rescore(),
        "service": service_summary(),
        "drift": drift_summary(),
        "storage_and_theory": storage_theory_summary(),
    }
    target = Path(__file__).with_name("derived_statistics.json")
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
