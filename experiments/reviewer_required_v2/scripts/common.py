from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
WORK = Path(os.environ.get("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_required_v2_workspace")).expanduser().resolve()
REPORT = WORK / "reports" / CFG["protocol_id"]
RESULT = WORK / "results" / CFG["protocol_id"]
MODEL = Path(os.environ.get("MODEL_PATH", "/root/autodl-tmp/coreflow_qwen3_load_smoke_v1_upload/assets/base_model")).expanduser().resolve()
SOURCE_WORK_ROOT = Path(os.environ.get("SOURCE_WORK_ROOT", "/root/autodl-tmp/coreflow_m0")).expanduser().resolve()
COMOL_ROOT = Path(os.environ.get("COMOL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload")).expanduser().resolve()
DATA_ROOT = Path(os.environ.get("MATH_DATA_ROOT", str(COMOL_ROOT / "datasets"))).expanduser().resolve()

def active_groups():
    raw = os.environ.get("ACTIVE_GROUPS", "2,3")
    values = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in values if x not in CFG["expert_groups"]]
    if bad: raise ValueError(f"Invalid ACTIVE_GROUPS: {bad}")
    return values

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def group_root(group): return WORK / "groups" / f"group{group}"
def manifest_path(group): return group_root(group) / "independent_loras" / "manifest.json"
