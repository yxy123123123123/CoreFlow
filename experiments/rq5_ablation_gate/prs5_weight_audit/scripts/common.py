from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: str | Path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config() -> dict:
    return load_json(ROOT / "config" / "prs5_protocol.json")


def paths(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or config()
    d = cfg["defaults"]
    source = Path(os.environ.get("SOURCE_WORK_ROOT", d["source_work_root"]))
    m2 = Path(os.environ.get("M2_ASSET_ROOT", d["m2_asset_root"]))
    phase2b = Path(os.environ.get("PHASE2B_WORK_ROOT", d["phase2b_work_root"]))
    work = Path(os.environ.get("WORK_ROOT", d["work_root"]))
    return {
        "source": source,
        "asset_root": source / "official_assets",
        "m2": m2,
        "phase2b": phase2b,
        "work": work,
        "reports": work / "reports" / "prs5",
        "results": work / "results" / "prs5",
        "logs": work / "logs",
    }


def variant_dir(item: dict, p: dict[str, Path]) -> Path:
    base = p["m2"] if item["root"] == "m2" else p["phase2b"]
    return base / item["relative"]
