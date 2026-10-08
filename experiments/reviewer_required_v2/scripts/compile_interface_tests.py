from __future__ import annotations
import json, os
from pathlib import Path
from common import CFG, REPORT, write_json

def validate(meta):
    if not isinstance(meta,dict): raise ValueError("metadata_not_object")
    experts=meta.get("expert_order")
    if not experts: raise ValueError("missing_expert_order")
    if len(experts)!=len(set(experts)): raise ValueError("duplicate_expert_name")
    modules=meta.get("modules")
    if not isinstance(modules,list) or not modules: raise ValueError("missing_modules")
    for item in modules:
        if not isinstance(item,dict): raise ValueError("module_not_object")
        if not item.get("name"): raise ValueError("missing_module_name")
        if int(item.get("effective_left_rank",-1))<=0 or int(item.get("effective_right_rank",-1))<=0: raise ValueError("invalid_rank")
        if item.get("dtype") not in (None,"bf16","bfloat16","fp16","float16","fp32","float32"): raise ValueError("invalid_dtype")
    if meta.get("partial_modules") is True: raise ValueError("partial_module_coverage_rejected")
    return True

def main():
    cases=[("missing_expert_order",{}),("duplicate_expert_name",{"expert_order":["a","a"],"modules":[{"name":"x","effective_left_rank":1,"effective_right_rank":1}]}),("invalid_rank",{"expert_order":["a"],"modules":[{"name":"x","effective_left_rank":0,"effective_right_rank":1}]}),("invalid_dtype",{"expert_order":["a"],"modules":[{"name":"x","effective_left_rank":1,"effective_right_rank":1,"dtype":"int8"}]}),("partial_module_coverage",{"expert_order":["a"],"partial_modules":True,"modules":[{"name":"x","effective_left_rank":1,"effective_right_rank":1}]}),("valid_minimal",{"expert_order":["a"],"modules":[{"name":"x","effective_left_rank":1,"effective_right_rank":1,"dtype":"bf16"}]})]
    out=[]
    for name,meta in cases:
        try: validate(meta); passed=name=="valid_minimal"; observed="accepted"
        except Exception as exc: passed=name!="valid_minimal"; observed=str(exc)
        out.append({"case":name,"expected":"accept" if name=="valid_minimal" else "reject","observed":observed,"pass":passed})
    payload={"status":"PASS" if all(x["pass"] for x in out) else "FAIL","protocol_id":CFG["protocol_id"],"cases":out,"scope":"CPU contract tests; no model output"}
    write_json(REPORT/"compile_interface_tests.json",payload); print(json.dumps(payload,ensure_ascii=False))
    if payload["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
