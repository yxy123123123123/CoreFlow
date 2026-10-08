from __future__ import annotations
import datetime, json, os
from pathlib import Path
from common import CFG, ROOT, WORK, REPORT, sha256

def main():
    if not (REPORT/"prepare.json").exists(): raise RuntimeError("Run prepare-required first")
    seal=REPORT/"PROTOCOL_SEAL.json"
    if seal.exists(): print(seal.read_text(encoding="utf-8")); return
    result=WORK/"results"/CFG["protocol_id"]
    if result.exists() and any(result.rglob("*")): raise RuntimeError("Cannot seal after required runtime outputs exist")
    payload={"status":"SEALED_BEFORE_REQUIRED_GPU_OUTPUTS","protocol_id":CFG["protocol_id"],"config_sha256":sha256(ROOT/"config"/"protocol.json"),"prepare_sha256":sha256(REPORT/"prepare.json"),"sealed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"full_six_dataset_comol_repeats":True,"continuous_batching_claim":False}
    REPORT.mkdir(parents=True,exist_ok=True); seal.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False))

if __name__=="__main__": main()
