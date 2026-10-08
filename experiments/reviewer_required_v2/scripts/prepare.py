from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def p(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")
    formal_pkg = p("FORMAL_V2_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    classeval_pkg = p("CLASSEVAL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_upload")
    qwen_work = p("QWEN_MATCH_WORK_ROOT", "/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")

    preflight = work / "reports" / "reviewer_minimal_phasea" / "preflight.json"
    if not preflight.exists():
        raise RuntimeError("Run preflight first")
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    mbpp_src = formal_pkg / "data" / "mbppplus_formal_candidate250.jsonl"
    class_src = classeval_pkg / "data" / "classeval_formal.jsonl"
    mbpp_dst = data / "llama_mbppplus_formal_q224.jsonl"
    class_dst = data / "llama_classeval_formal_q224.jsonl"
    if not mbpp_dst.exists():
        shutil.copy2(mbpp_src, mbpp_dst)
    if not class_dst.exists():
        shutil.copy2(class_src, class_dst)

    rows = [json.loads(line) for line in mbpp_src.read_text(encoding="utf-8").splitlines() if line.strip()]
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"coreflow-reviewer-minimal-phasea-v1|drift|{row['task_id']}".encode("utf-8")
        ).hexdigest(),
    )
    drift_rows = ranked[:32]
    drift_path = data / "llama_drift32.jsonl"
    drift_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in drift_rows), encoding="utf-8")

    qgroup_src = qwen_work / "groups" / "group1"
    qgroup_dst = work / "qwen_group1" / "groups" / "group1"
    for name in ("independent_loras", "gate", "banks", "comol_outputs"):
        link_or_copy(qgroup_src / name, qgroup_dst / name)

    manifest = {
        "status": "PASS",
        "protocol_id": "coreflow-reviewer-minimal-phasea-v1",
        "prepared_before_gpu_outputs": True,
        "data": {
            "mbpp_formal": {"path": str(mbpp_dst), "rows": len(rows), "sha256": sha256(mbpp_dst)},
            "classeval_formal": {
                "path": str(class_dst),
                "rows": sum(1 for _ in class_dst.open("r", encoding="utf-8")),
                "sha256": sha256(class_dst),
            },
            "drift32": {"path": str(drift_path), "rows": len(drift_rows), "sha256": sha256(drift_path)},
        },
        "qwen_group1": {"source": str(qgroup_src), "linked": str(qgroup_dst)},
    }
    write_json(work / "reports" / "reviewer_minimal_phasea" / "prepare.json", manifest)
    print(json.dumps({"status": "PASS", "mbpp_rows": len(rows), "classeval_rows": manifest["data"]["classeval_formal"]["rows"], "drift_rows": len(drift_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
