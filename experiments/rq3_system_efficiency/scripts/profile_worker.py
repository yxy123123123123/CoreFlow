#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time, types
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from coreflow.loraflow import load_loraflow_model
import coreflow.runtime as runtime
from scripts.common import config, core_bank, gate_path, isvd_bank, paths, sha256_file, write_json

def scope_full_projections(model):
    patched=0
    for module in model.modules():
        for attr in ("lora_A","lora_B"):
            container=getattr(module,attr,None)
            if not container: continue
            for child in container.values():
                if getattr(child,"_profile_wrapped",False): continue
                original=child.forward
                def wrapped(self,*a,_original=original,**kw):
                    with torch.autograd.profiler.record_function("adapter_scope::full_lora_projection"):
                        return _original(*a,**kw)
                child.forward=types.MethodType(wrapped,child); child._profile_wrapped=True; patched+=1
    return patched

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    cfg,p=config(),paths(); final=Path(args.output)
    if final.is_file(): print(f"[SKIP] {final}"); return
    if final.exists(): raise RuntimeError(f"Retained output exists: {final}")
    order=cfg["expert_order"]; t0=time.perf_counter()
    tokenizer,model,gate_report=load_loraflow_model(p["model"],p["asset_root"],p["official_root"],task=None,adapter_order=order,gate_path=gate_path(cfg,p,cfg["r2_systems"]["gate_seed"]),dtype=torch.bfloat16,device="cuda")
    wrapped=0
    if args.method=="coreflow_q185_frozen":
        original=runtime.coreflow_delta
        def scoped(*a,**kw):
            with torch.autograd.profiler.record_function("adapter_scope::coreflow_delta"): return original(*a,**kw)
        runtime.coreflow_delta=scoped; install=runtime.install_coreflow(model,core_bank(cfg,p),False,True,order)
    elif args.method=="isvd_legacy_padded_matched_q185":
        original=runtime.isvd_delta
        def scoped(*a,**kw):
            with torch.autograd.profiler.record_function("adapter_scope::isvd_padded_delta"): return original(*a,**kw)
        runtime.isvd_delta=scoped; install=runtime.install_isvd(model,isvd_bank(cfg,p),True,order)
    else:
        install=None; wrapped=scope_full_projections(model)
    ids=tokenizer("CoreFlow profiler input. ",add_special_tokens=True,return_tensors="pt")["input_ids"][0]; length=cfg["r3_profiler"]["input_length"]; ids=ids.repeat((length+len(ids)-1)//len(ids))[:length].unsqueeze(0).to("cuda"); mask=torch.ones_like(ids)
    kwargs=dict(input_ids=ids,attention_mask=mask,do_sample=False,min_new_tokens=cfg["r3_profiler"]["output_length"],max_new_tokens=cfg["r3_profiler"]["output_length"],use_cache=True)
    with torch.inference_mode():
        for _ in range(cfg["r3_profiler"]["warmup"]): model.generate(**kwargs)
    torch.cuda.synchronize()
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
        with torch.inference_mode():
            for _ in range(cfg["r3_profiler"]["active_steps"]):
                with torch.autograd.profiler.record_function("whole_model_generation"):
                    model.generate(**kwargs)
                prof.step()
    torch.cuda.synchronize(); events=list(prof.events()); cuda_events=[e for e in events if str(e.device_type).endswith("CUDA")]; cpu_ops=[e for e in events if str(e.device_type).endswith("CPU") and not e.name.startswith("ProfilerStep")]; scopes=[e for e in events if e.name.startswith("adapter_scope::")]
    by_name={}
    for e in scopes:
        item=by_name.setdefault(e.name,{"calls":0,"cpu_time_total_us":0.0,"cuda_time_total_us":0.0}); item["calls"]+=1; item["cpu_time_total_us"]+=float(e.cpu_time_total); item["cuda_time_total_us"]+=float(getattr(e,"cuda_time_total",0.0))
    payload={"status":"PASS","protocol_id":cfg["protocol_id"],"method":args.method,"load_install_seconds":time.perf_counter()-t0,"gate":gate_report,"install":install,"full_projection_modules_wrapped":wrapped,"active_generation_steps":cfg["r3_profiler"]["active_steps"],"operator_calls_cpu":len(cpu_ops),"kernel_events_cuda":len(cuda_events),"adapter_scopes":by_name,"definitions":{"operator_calls_cpu":"individual profiler CPU operator events","kernel_events_cuda":"individual profiler CUDA device events; not CPU operator calls","adapter_scope":"Core/ISVD fused delta; Full covers LoRA A/B projections and excludes base linear/addition"},"trace_exported":False}
    write_json(final,payload); print(json.dumps({"status":"PASS","method":args.method,"kernel_events_cuda":len(cuda_events)}))

if __name__=="__main__": main()
