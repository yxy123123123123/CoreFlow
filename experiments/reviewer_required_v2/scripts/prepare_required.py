from __future__ import annotations
import hashlib, json, os, shutil
from pathlib import Path
from common import CFG, WORK, REPORT, active_groups, group_root, write_json, sha256

def ep(name, default): return Path(os.environ.get(name, default)).expanduser().resolve()
def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink(): return
    try: os.symlink(src,dst,target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copy2(src,dst)

def main():
    if not (REPORT/"preflight.json").exists(): raise RuntimeError("Run preflight-required first")
    formal_pkg=ep("FORMAL_V2_PACKAGE_ROOT","/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    class_pkg=ep("CLASSEVAL_PACKAGE_ROOT","/root/autodl-tmp/coreflow_indep_confirm_v1_upload")
    qwen_work=ep("QWEN_MATCH_WORK_ROOT","/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace")
    data=WORK/"data"; data.mkdir(parents=True,exist_ok=True)
    sources={"mbpp":formal_pkg/"data"/"mbppplus_formal_candidate250.jsonl","classeval":class_pkg/"data"/"classeval_formal.jsonl"}
    destinations={"mbpp":data/"llama_mbppplus_formal_q224.jsonl","classeval":data/"llama_classeval_formal_q224.jsonl"}
    for k,src in sources.items():
        if not destinations[k].exists(): shutil.copy2(src,destinations[k])
    existing_group1=qwen_work/"groups"/"group1"
    if not existing_group1.exists(): raise FileNotFoundError(existing_group1)
    link_or_copy(existing_group1,WORK/"existing_group1")
    link_or_copy(existing_group1,WORK/"qwen_group1"/"groups"/"group1")
    for group in active_groups(): group_root(group).mkdir(parents=True,exist_ok=True)
    payload={"status":"PASS","protocol_id":CFG["protocol_id"],"prepared_before_required_gpu_outputs":True,"data":{"mbpp":{"path":str(destinations["mbpp"]),"rows":sum(1 for _ in destinations["mbpp"].open(encoding="utf-8")),"sha256":sha256(destinations["mbpp"])},"classeval":{"path":str(destinations["classeval"]),"rows":sum(1 for _ in destinations["classeval"].open(encoding="utf-8")),"sha256":sha256(destinations["classeval"])}},"reuse_group1":str(existing_group1),"active_groups":active_groups()}
    write_json(REPORT/"prepare.json",payload); print(json.dumps(payload,ensure_ascii=False))

if __name__=="__main__": main()
