from __future__ import annotations
import json, os, re
from pathlib import Path
from common import CFG, REPORT, WORK, write_json

def ep(name,default): return Path(os.environ.get(name,default)).expanduser().resolve()
def main():
    qwen=ep("QWEN_MATCH_WORK_ROOT","/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")
    record=[]; manifest=qwen/"groups"/"group1"/"independent_loras"/"manifest.json"
    if manifest.exists():
        payload=json.loads(manifest.read_text(encoding="utf-8")); record.append({"stage":"independent_lora_group1","elapsed_seconds":payload.get("elapsed_seconds"),"training":payload.get("training"),"source":"manifest"})
    train_log=qwen.parent/"logs"/"train_independent_group1.log"
    if train_log.exists(): record.append({"stage":"independent_lora_group1_log","path":str(train_log),"source":"log","sha256_size":train_log.stat().st_size})
    comol=qwen/"groups"/"group1"/"comol_manifest.json"
    if comol.exists(): record.append({"stage":"comol_group1","manifest":json.loads(comol.read_text(encoding="utf-8"))})
    payload={"status":"ANALYZED","protocol_id":CFG["protocol_id"],"lifecycle_units":{"independent_lora_training":"includes all eight independent runs","comol_training":"joint MoE-LoRA training","coreflow_compile":"offline bank construction; report separately","inference":"excluded from lifecycle training cost"},"existing_group1":record,"new_group2_3":"will be recorded by stage manifests; no claim is made until both groups complete"}
    write_json(REPORT/"lifecycle_audit.json",payload); print(json.dumps({"status":payload["status"],"existing_records":len(record)},ensure_ascii=False))
if __name__=="__main__": main()
