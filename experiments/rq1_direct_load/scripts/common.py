from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
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
    return load_json(ROOT / "config" / "stage1_protocol.json")


def paths(cfg: dict | None = None) -> dict[str, Path]:
    cfg = cfg or config(); d = cfg["defaults"]
    model = Path(os.environ.get("MODEL", d["model"]))
    source = Path(os.environ.get("SOURCE_WORK_ROOT", d["source_work_root"]))
    m2 = Path(os.environ.get("M2_ASSET_ROOT", d["m2_asset_root"]))
    formal = Path(os.environ.get("FORMAL_V2_WORK_ROOT", d["formal_v2_work_root"]))
    work = Path(os.environ.get("WORK_ROOT", d["work_root"]))
    return {
        "model": model, "source": source, "asset_root": source / "official_assets",
        "vendor_root": source / "vendor" / "LoRAFlow", "m2_root": m2,
        "m2_lock": Path(os.environ.get("M2_ASSET_LOCK", d["m2_asset_lock"])),
        "formal_v2": formal, "work": work, "data": work / "data" / "stage1",
        "reports": work / "reports" / "stage1", "results": work / "results" / "stage1",
        "banks": work / "banks" / "stage1", "logs": work / "logs",
    }


def request_json(url: str, payload: dict | None = None, timeout: int = 1200):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if data is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(url: str, process, timeout_seconds: int = 900):
    deadline, last = time.time() + timeout_seconds, None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            return request_json(url, timeout=5)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = str(exc); time.sleep(2)
    raise TimeoutError(f"Server health timeout: {last}")
