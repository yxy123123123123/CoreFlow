from __future__ import annotations
import json, os
from pathlib import Path
from common import CFG, REPORT, WORK, write_json

def ep(name,default): return Path(os.environ.get(name,default)).expanduser().resolve()
def bytes_tree(path):
    if not path.exists(): return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
def load(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
def main():
    formal=ep("FORMAL_V2_WORK_ROOT","/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    qwen=ep("QWEN_MATCH_WORK_ROOT","/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")
    items=[]
    targets=[("llama_core_q185",formal/"m2_assets"/"core_banks"/"k5_code"/"q185_bf16"),("llama_core_q224",formal/"m2_assets"/"core_banks"/"k5_code"/"q224_bf16"),("llama_isvd_q185",formal/"isvd_banks"/"k5_code"/"matched_q185_bf16"),("qwen_group1_banks",qwen/"groups"/"group1"/"banks"),("qwen_group1_loras",qwen/"groups"/"group1"/"independent_loras")]
    for label,path in targets:
        record={"label":label,"path":str(path),"exists":path.exists(),"stored_bytes":bytes_tree(path)}
        config=path/"core_config.json" if path.is_dir() else path
        if config.exists():
            cfg=load(config); record["config_keys"]=sorted(cfg) if isinstance(cfg,dict) else None; record["modules"]=len(cfg.get("modules",[])) if isinstance(cfg,dict) else None
            record["efficiency"]=cfg.get("efficiency") if isinstance(cfg,dict) else None
        items.append(record)
    payload={"status":"ANALYZED","protocol_id":CFG["protocol_id"],"definitions":{"logical_parameters":"mathematical parameter count","effective_payload":"non-padding tensor payload","stored_bytes":"recursive bytes on disk","runtime_memory":"allocated/reserved after loading; must be measured separately"},"artifacts":items,"padding_is_not_silently_counted_as_effective_payload":True}
    write_json(REPORT/"storage_audit.json",payload); print(json.dumps({"status":payload["status"],"artifacts":len(items)},ensure_ascii=False))
if __name__=="__main__": main()
