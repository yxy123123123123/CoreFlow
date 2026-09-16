#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from common import ROOT, config, load_json, now_utc, paths, write_json


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    cfg, p = config(), paths()
    audit_path = p["results"] / "weight_audit" / "weight_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("Run audit first")
    evidence = load_json(ROOT / "evidence" / "prs5_local_evidence.json")
    audit = load_json(audit_path)
    evidence_by = {row["variant"]: row for row in evidence["rows"]}
    audit_by = {row["variant"]: row for row in audit["variants"]}
    order = list(cfg["variants"])
    rows = []
    for variant in order:
        local, weight = evidence_by[variant], audit_by[variant]
        row = dict(local)
        row.update({
            "artifact_bytes": weight["artifact_bytes"],
            "structural_zero_rank_entries": weight["structural_zero_rank_entries"],
            "expert_error_std": weight["construction_error_expert_dispersion"]["std_population"],
            "expert_error_iqr": weight["construction_error_expert_dispersion"]["iqr"],
            "expert_error_range": weight["construction_error_expert_dispersion"]["max_minus_min"],
            "left_energy_capture_mean": None if weight["left"] is None else weight["left"]["mean"],
            "right_energy_capture_mean": None if weight["right"] is None else weight["right"]["mean"],
            "joint_energy_capture_mean": None if weight["joint"] is None else weight["joint"]["mean"],
            "left_canonical_cosine_mean": None if weight["left_cos"] is None else weight["left_cos"]["mean"],
            "right_canonical_cosine_mean": None if weight["right_cos"] is None else weight["right_cos"]["mean"],
        })
        rows.append(row)
    out = p["reports"] / "final"
    out.mkdir(parents=True, exist_ok=True)
    matrix_path = out / "prs5_evidence_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {"status": "PRS5_COMPLETE_AS_STRUCTURAL_EVIDENCE_NO_METHOD_CHANGE", "protocol_id": cfg["protocol_id"],
               "timestamp_utc": now_utc(), "selection_forbidden": True, "main_method_change_forbidden": True,
               "quality_scope": "A0-A3 reuse frozen MBPP+ dev128 seed41 results; A4-A6 intentionally have no task quality",
               "system_scope": "Historical Phase-2B Full-vectorized ratios only; not formal original-Full system evidence",
               "rows": rows, "weight_audit": audit}
    write_json(out / "prs5_summary.json", payload)
    lines = [
        "# CoreFlow PRS-5 A0–A6 结构证据矩阵", "", f"状态：`{payload['status']}`", "",
        "本报告没有生成任何任务答案、没有重新选择 q，也没有改变冻结主方法。A0–A3 的质量数值全部复用 Phase-2B；A4–A6 只报告权重级与系统诊断。", "",
        "## 质量与重构", "", "| 变体 | dev128 strict pass@1 | 相对 A0 (pp, 95% CI) | fused error mean | per-expert error mean |", "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        quality = "—" if row["dev_correct"] is None else f"{row['dev_correct']}/{row['dev_rows']} ({100*row['dev_pass_at_1']:.2f}%)"
        diff = "—" if row["difference_vs_a0_pp"] is None else f"{row['difference_vs_a0_pp']:+.2f} [{row['ci95_lower_pp']:+.2f}, {row['ci95_upper_pp']:+.2f}]"
        lines.append(f"| {row['variant']} | {quality} | {diff} | {row['fused_error_mean']:.4f} | {row['per_expert_error_mean']:.4f} |")
    lines += ["", "## 专家离散度与子空间", "", "| 变体 | expert error std | IQR | range | left capture | right capture | joint capture | zero-rank |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['expert_error_std']:.4f} | {row['expert_error_iqr']:.4f} | {row['expert_error_range']:.4f} | {fmt(row['left_energy_capture_mean'])} | {fmt(row['right_energy_capture_mean'])} | {fmt(row['joint_energy_capture_mean'])} | {row['structural_zero_rank_entries']} |")
    lines += ["", "## 资源与历史系统诊断", "", "| 变体 | logical params / MAC | artifact MiB | 历史端到端 speedup vs Full-vectorized |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['logical_parameters']:,} | {row['artifact_bytes']/2**20:.2f} | {fmt(row['historical_system_speedup_vs_full_vectorized'])}× |")
    lines += ["", "## 解释边界", "", "- A0 仍是冻结正式工作点；单 seed 点估计不能触发方法替换。", "- A1–A3 的配对置信区间均覆盖 0，属于机制性开发结果，不是新的确认性质量证据。", "- A4–A6 没有 task-quality 数值是预先设计的空白，不得用重构误差替代 strict pass@1。", "- 历史系统比率以 Full-vectorized 为参照，仅用于诊断；论文正式 Original Full 系统证据来自独立补充实验。", "- `zero-rank` 是 CoreFlow/one-sided bank 的结构分配审计，不能与 Independent-SVD 的逐专家 rank allocation 混为一谈。", ""]
    (out / "prs5_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "report": str(out / "prs5_report.md")}))


if __name__ == "__main__":
    main()
