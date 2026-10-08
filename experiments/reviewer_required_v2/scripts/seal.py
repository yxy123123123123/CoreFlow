from __future__ import annotations

import hashlib
import json
import os
import platform
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


def main() -> None:
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_minimal_phaseA_v1_workspace")
    report = work / "reports" / "reviewer_minimal_phasea"
    prepared = report / "prepare.json"
    if not prepared.exists():
        raise RuntimeError("Run prepare first")
    seal = report / "PROTOCOL_SEAL.json"
    if seal.exists():
        print(json.dumps(json.loads(seal.read_text(encoding="utf-8")), ensure_ascii=False))
        return
    result_root = work / "results" / "reviewer_minimal_phasea"
    existing = [str(path) for path in result_root.rglob("*")] if result_root.exists() else []
    if existing:
        raise RuntimeError("Cannot seal after runtime outputs exist: " + ", ".join(existing[:5]))
    payload = {
        "status": "SEALED_BEFORE_PHASEA_GPU_OUTPUTS",
        "protocol_id": "coreflow-reviewer-minimal-phasea-v1",
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "config_sha256": sha256(ROOT / "config" / "protocol.json"),
        "prepare_sha256": sha256(prepared),
        "sealed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "q224_is_sensitivity_only": True,
        "continuous_batching_claim": False,
    }
    seal.parent.mkdir(parents=True, exist_ok=True)
    seal.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
