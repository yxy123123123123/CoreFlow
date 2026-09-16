from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def config():
    return load_json(ROOT / "config" / "original_baselines_protocol.json")


def paths(cfg=None):
    cfg = cfg or config()
    defaults = cfg["defaults"]
    source = Path(os.environ.get("SOURCE_WORK_ROOT", defaults["source_work_root"])).resolve()
    formal1 = Path(os.environ.get("FORMAL_V1_WORK_ROOT", defaults["formal_v1_work_root"])).resolve()
    formal2 = Path(os.environ.get("FORMAL_V2_WORK_ROOT", defaults["formal_v2_work_root"])).resolve()
    work = Path(os.environ.get("WORK_ROOT", defaults["work_root"])).resolve()
    return {
        "root": ROOT,
        "model": Path(os.environ.get("MODEL", defaults["model"])).resolve(),
        "source": source,
        "formal1": formal1,
        "formal2": formal2,
        "work": work,
        "reports": work / "reports" / "original_baselines_v1",
        "results": work / "results" / "original_baselines_v1",
        "data": work / "data" / "original_baselines_v1",
        "asset_root": source / "official_assets",
        "official_root": source / "vendor" / "LoRAFlow",
        "runtime_python": Path(os.environ.get("PYTHON", source / "runtime_env" / "bin" / "python")),
        "cuda_device": os.environ.get("CUDA_DEVICE", defaults["cuda_device"]),
    }


def gate_path(cfg, p, seed):
    return p["formal1"] / cfg["assets"]["gate_pattern"].format(seed=seed)


def core_bank(cfg, p):
    return p["formal1"] / cfg["assets"]["core_bank"]


def isvd_bank(cfg, p):
    return p["formal2"] / cfg["assets"]["isvd_bank"]


def request_json(url, payload=None, timeout=1800):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_health(url, process, timeout=1200):
    deadline, last = time.time() + timeout, None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}")
        try:
            return request_json(url, timeout=3)
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise TimeoutError(f"Server did not become healthy: {last}")
