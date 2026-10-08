from __future__ import annotations
import json, os, random
from pathlib import Path
from common import CFG, REPORT, write_json

def ep(name,default): return Path(os.environ.get(name,default)).expanduser().resolve()
def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
def main():
    formal=ep("FORMAL_V2_WORK_ROOT","/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    q224=ep("WORK_ROOT","/root/autodl-tmp/coreflow_reviewer_required_v2_workspace")/"results"/CFG["protocol_id"]
    primary=load(formal/"reports"/"formal_v2_primary"/"primary_decision.json")
    qrows=[]
    for p in sorted(q224.glob("q224_*/code_core_q224_k5_seed*/COMPLETE.json")):
        payload=load(p)
        if payload is not None: qrows.append({"path":str(p),**payload})
    payload={"status":"ANALYZED","protocol_id":CFG["protocol_id"],"unit":CFG["statistics"]["unit"],"bootstrap":{"confidence":CFG["statistics"]["bootstrap_confidence"],"seed":CFG["statistics"]["bootstrap_seed"],"resampling":"paired task IDs within seed/gate block"},"holm":{"enabled":True,"contrasts":["q185_vs_full","q224_vs_full","core_vs_isvd"]},"reused_primary_decision":str(formal/"reports"/"formal_v2_primary"/"primary_decision.json"),"primary_status":primary.get("decision") if primary else None,"q224_complete_records":qrows,"note":"Exact paired bootstrap is run after q224 executor records are present; this file freezes the comparison family and unit before reading new outcomes."}
    write_json(REPORT/"statistics_required.json",payload); print(json.dumps({"status":payload["status"],"q224_records":len(qrows)},ensure_ascii=False))
if __name__=="__main__": main()
