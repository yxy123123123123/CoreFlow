from __future__ import annotations
import json, os, platform
from pathlib import Path
from common import CFG, MODEL, SOURCE_WORK_ROOT, COMOL_ROOT, DATA_ROOT, WORK, REPORT, sha256

def ep(name, default): return Path(os.environ.get(name, default)).expanduser().resolve()

def main():
    formal_pkg = ep("FORMAL_V2_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    formal_work = ep("FORMAL_V2_WORK_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    classeval_pkg = ep("CLASSEVAL_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_upload")
    classeval_work = ep("CLASSEVAL_WORK_ROOT", "/root/autodl-tmp/coreflow_indep_confirm_v1_workspace")
    qwen_work = ep("QWEN_MATCH_WORK_ROOT", "/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")
    required = {
        "model_config": MODEL / "config.json",
        "source_assets": SOURCE_WORK_ROOT / "official_assets",
        "source_vendor": SOURCE_WORK_ROOT / "vendor" / "LoRAFlow",
        "comol_train": COMOL_ROOT / "train.py",
        "comol_eval": COMOL_ROOT / "test_math.py",
        "math14k_train": DATA_ROOT / "math_14k" / "train.json",
        "math14k_validation": DATA_ROOT / "math_14k" / "validation.json",
        "math_commonsense": DATA_ROOT / "math_commonsense",
        "formal_runner": formal_pkg / "scripts" / "run_formal_method.py",
        "formal_mbpp": formal_pkg / "data" / "mbppplus_formal_candidate250.jsonl",
        "formal_m2_assets": formal_work / "m2_assets",
        "classeval_runner": classeval_pkg / "scripts" / "run_formal_method.py",
        "classeval_data": classeval_pkg / "data" / "classeval_formal.jsonl",
        "qwen_group1": qwen_work / "groups" / "group1",
        "qwen_group1_manifest": qwen_work / "groups" / "group1" / "bank_manifest.json",
        "qwen_group1_comol": qwen_work / "groups" / "group1" / "comol_manifest.json",
        "qwen_group1_q16_bank": qwen_work / "groups" / "group1" / "banks" / "q16_bf16" / "core_bank.safetensors",
    }
    missing = [k for k,v in required.items() if not v.exists()]
    if missing: raise FileNotFoundError("Missing prerequisites: " + ", ".join(f"{k}={required[k]}" for k in missing))
    q224_bank = formal_work / "m2_assets" / "core_banks" / "k5_code" / "q224_bf16" / "core_bank.safetensors"
    q224_gate = formal_work / "m2_assets" / "gates" / "code_k5_code_seed41.pt"
    if not q224_bank.exists() or not q224_gate.exists(): raise FileNotFoundError("q224 bank/gate missing")
    payload = {"status":"PASS","protocol_id":CFG["protocol_id"],"python":platform.python_version(),"platform":platform.platform(),"active_groups":os.environ.get("ACTIVE_GROUPS","2,3"),"paths":{k:str(v) for k,v in required.items()},"hashes":{"model_config":sha256(required["model_config"]),"formal_mbpp":sha256(required["formal_mbpp"]),"classeval_data":sha256(required["classeval_data"]),"q224_bank":sha256(q224_bank),"q224_gate":sha256(q224_gate)},"work_root":str(WORK)}
    REPORT.mkdir(parents=True, exist_ok=True); (REPORT/"preflight.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","protocol_id":CFG["protocol_id"],"groups":os.environ.get("ACTIVE_GROUPS","2,3")},ensure_ascii=False))

if __name__ == "__main__": main()
