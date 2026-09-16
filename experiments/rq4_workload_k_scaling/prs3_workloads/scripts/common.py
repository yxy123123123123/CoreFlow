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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def config():
    return load_json(ROOT / "config" / "prs3_pilot_protocol.json")


def _find_k8_asset_root():
    explicit = os.environ.get("K8_ASSET_ROOT")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path("/root/autodl-tmp/coreflow_m3_k_scaling_mini_upload/lora_assets/m3_k8_strict"),
            Path("/root/autodl-tmp/coreflow_applsci_supplement_v2_system_r1_upload/lora_assets/m3_k8_strict"),
        ]
    )
    for root in candidates:
        if all((root / name / "adapter_config.json").is_file() for name in ("magicoder", "openwebmath", "gsm8k_loftq")):
            return root.resolve()
    return candidates[0].resolve()


def paths(cfg=None):
    cfg = cfg or config()
    defaults = cfg["defaults"]
    source = Path(os.environ.get("SOURCE_WORK_ROOT", defaults["source_work_root"])).resolve()
    formal1 = Path(os.environ.get("FORMAL_V1_WORK_ROOT", defaults["formal_v1_work_root"])).resolve()
    formal2 = Path(os.environ.get("FORMAL_V2_WORK_ROOT", defaults["formal_v2_work_root"])).resolve()
    m3 = Path(os.environ.get("M3_WORK_ROOT", defaults["m3_work_root"])).resolve()
    work = Path(os.environ.get("WORK_ROOT", defaults["work_root"])).resolve()
    return {
        "root": ROOT,
        "model": Path(os.environ.get("MODEL", defaults["model"])).resolve(),
        "source": source,
        "formal1": formal1,
        "formal2": formal2,
        "m3": m3,
        "k8_assets": _find_k8_asset_root(),
        "work": work,
        "reports": work / "reports" / "prs3_pilot_v1",
        "results": work / "results" / "prs3_pilot_v1",
        "data": work / "data" / "prs3_pilot_v1",
        "asset_root": source / "official_assets",
        "official_root": source / "vendor" / "LoRAFlow",
        "runtime_python": Path(os.environ.get("PYTHON", source / "runtime_env" / "bin" / "python")),
        "cuda_device": os.environ.get("CUDA_DEVICE", defaults["cuda_device"]),
    }


def gate_path(cfg, p):
    return p["formal1"] / cfg["assets"]["gate_pattern"].format(seed=cfg["assets"]["gate_seed"])


def bank_path(cfg, p, pool, kind):
    item = cfg["assets"]["banks"][pool]
    relative = item[f"{kind}_relative"]
    root_kind = item.get(f"{kind}_absolute_kind", "m3")
    base = {"m3": p["m3"], "formal_v1": p["formal1"], "formal_v2": p["formal2"]}[root_kind]
    return base / relative


def expert_path(cfg, p, name):
    item = cfg["assets"]["experts"][name]
    if item["source"] == "official":
        return p["asset_root"] / "LoRAs" / item["relative"]
    if item["source"] == "k8_external":
        return p["k8_assets"] / item["relative"]
    raise ValueError(item["source"])


def adapter_checkpoint(path):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = Path(path) / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No adapter checkpoint under {path}")


def request_json(url, payload=None, timeout=1800):
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request, timeout=timeout) as response:
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
