import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "coreflow-reviewer-required-v2"
REPORTS = ROOT / "reports" / "coreflow-reviewer-required-v2"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


print("SERVICE")
service = ROOT / "results" / "coreflow-reviewer-required-v2" / "service" / "RTX_4080_SUPER"
for path in sorted(service.glob("*/*.json")):
    record = load(path)
    keys = [
        "method", "concurrency", "input_tokens", "output_tokens",
        "mean_decode_tokens_per_second", "p50_latency_seconds",
        "p95_latency_seconds", "p99_latency_seconds", "peak_allocated_gib",
        "peak_reserved_gib", "startup_seconds", "average_power_watts",
        "joules_per_request", "joules_per_output_token",
    ]
    print(path.parent.name, path.stem, {key: record.get(key) for key in keys})


print("\nDRIFT")
drift = load(REPORTS / "drift_diagnostics.json")
for name, comp in drift["comparisons"].items():
    layers = comp["hidden_by_layer"]
    layer_ids = sorted(layers, key=lambda value: int(value))
    rel = [layers[layer]["relative_l2_mean"] for layer in layer_ids]
    cos = [layers[layer]["cosine_mean"] for layer in layer_ids]
    print(name, {
        "rows": comp["rows"],
        "first_hidden_relative_l2": rel[0],
        "final_hidden_relative_l2": rel[-1],
        "max_layer_hidden_relative_l2": max(rel),
        "final_hidden_cosine": cos[-1],
        "logits": comp.get("logits"),
        "gate_summary": comp.get("gate_summary"),
    })


print("\nSTORAGE")
storage = load(REPORTS / "storage_audit.json")
print("top keys", list(storage))
for item in storage.get("artifacts", []):
    print({key: item.get(key) for key in [
        "label", "kind", "stored_bytes", "logical_parameters",
        "effective_payload_parameters", "padding_parameters", "path"
    ]})


print("\nTHEORY")
theory = load(REPORTS / "theory_error_analysis.json")
print("top keys", list(theory))
for bank in theory.get("banks", []):
    modules = bank.get("modules", [])
    max_errors = [item["max_sampled_relative_error"] for item in modules]
    mean_errors = [item["mean_sampled_relative_error"] for item in modules]
    print({
        "label": bank.get("label"),
        "q": bank.get("q"),
        "modules": len(modules),
        "mean_module_mean_error": statistics.fmean(mean_errors) if mean_errors else None,
        "max_sample_error": max(max_errors) if max_errors else None,
        "bound": bank.get("bound"),
    })
