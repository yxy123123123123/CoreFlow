from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean


def p(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_q224(root: Path):
    rows = []
    for path in sorted(root.glob("q224_*/code_core_q224_k5_seed*/COMPLETE.json")):
        payload = load(path)
        if payload is not None:
            rows.append({"path": str(path), **payload})
    return rows


def collect_service(root: Path):
    rows = []
    for path in sorted(root.glob("service/*/*/*.json")):
        if path.name == "COMPLETE.json":
            continue
        payload = load(path)
        if payload is not None:
            rows.append({"path": str(path), **payload})
    return rows


def main() -> None:
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")
    report = work / "reports" / "reviewer_minimal_phasea"
    result = work / "results" / "reviewer_minimal_phasea"
    q224 = collect_q224(result)
    service = collect_service(result)
    drift = load(report / "drift_diagnostics.json")
    by_method = {}
    for row in service:
        key = row.get("method", "unknown")
        by_method.setdefault(key, []).append(row.get("mean_decode_tokens_per_second"))
    service_summary = {
        method: {"runs": len(values), "mean_decode_tokens_per_second": mean(values) if values else None}
        for method, values in by_method.items()
    }
    payload = {
        "status": "ANALYZED",
        "protocol_id": "coreflow-reviewer-minimal-phasea-v1",
        "q224_outputs": q224,
        "service_outputs": service,
        "service_summary_by_method": service_summary,
        "drift_present": drift is not None,
        "interpretation": {
            "q224": "registered sensitivity only; q185 remains the frozen primary point",
            "service": "same-hardware method comparison; cross-hardware claim only where both hardware runs are present",
            "concurrency": "single-worker queued-load label, not continuous batching",
        },
    }
    report.mkdir(parents=True, exist_ok=True)
    (report / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CoreFlow reviewer minimal Phase-A analysis",
        "",
        f"- q224 output records: {len(q224)}",
        f"- service output records: {len(service)}",
        f"- hidden-state drift report: {'present' if drift is not None else 'pending'}",
        "",
        "The q224 runs are sensitivity evidence and do not replace the frozen q185 primary result. Service c4 is a sequential queued-load label, not a continuous-batching experiment.",
        "",
        "## Service mean decode throughput by method",
        "",
        "| Method | Runs | Mean decode tokens/s |",
        "|---|---:|---:|",
    ]
    for method, stats in sorted(service_summary.items()):
        val = "" if stats["mean_decode_tokens_per_second"] is None else f"{stats['mean_decode_tokens_per_second']:.3f}"
        lines.append(f"| {method} | {stats['runs']} | {val} |")
    (report / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "q224": len(q224), "service": len(service), "drift": drift is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
