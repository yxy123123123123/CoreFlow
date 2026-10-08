from __future__ import annotations
import argparse, gc, json, re, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import CFG, DATA_ROOT, MODEL, group_root, load_json, manifest_path, write_json
from coreflow.io import canonical_module_name

def fmt(r):
    if r.get("input"):
        return ("Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n"+r.get("instruction","")+"\n\n### Input:\n"+r.get("input","")+"\n\n### Response:")
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"+r.get("instruction","")+"\n\n### Response:"
def load_model(group):
    tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True,padding_side="left",truncation_side="left")
    base=AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,device_map={"":"cuda:0"},low_cpu_mem_usage=True,local_files_only=True,attn_implementation="sdpa").eval()
    recs=load_json(manifest_path(group))["records"]; first=Path(recs[0]["path"])
    model=PeftModel.from_pretrained(base,str(first),adapter_name="expert_01",is_trainable=False,local_files_only=True,low_cpu_mem_usage=True,autocast_adapter_dtype=False).eval()
    for i,rec in enumerate(recs[1:],2): model.load_adapter(str(rec["path"]),adapter_name=f"expert_{i:02d}",is_trainable=False,low_cpu_mem_usage=True,autocast_adapter_dtype=False)
    model.set_adapter("expert_01"); return tok,model
def targets(model):
    order=[f"expert_{i:02d}" for i in range(1,9)]; out=[]
    for raw,m in model.named_modules():
        if hasattr(m,"base_layer") and hasattr(m,"lora_A") and all(n in m.lora_A and n in m.lora_B for n in order): out.append((raw,m))
    if len(out)!=36*5: raise RuntimeError(f"target modules={len(out)}")
    return out
def install(model,group,method,q=None):
    order=[f"expert_{i:02d}" for i in range(1,9)]; layers=next(x for x in model.modules() if isinstance(x,torch.nn.ModuleList) and len(x)==36); mapping=defaultdict(list)
    modules=targets(model)
    for raw,m in modules: mapping[int(raw.split(".")[raw.split(".").index("layers")+1])].append(m)
    gate=load_file(str(group_root(group)/"gate"/"gate.safetensors"),device="cuda:0"); hooks=[]
    for idx,layer in enumerate(layers):
        def hook(_m,args,i=idx):
            h=args[0]; z=h.float(); z=z*torch.rsqrt(z.square().mean(dim=-1,keepdim=True).clamp_min(1e-6)); w=torch.softmax(torch.einsum("...d,cd->...c",z,gate["gate.weight"][i].float())+gate["gate.bias"][i].float(),dim=-1).to(h.dtype)
            for mod in mapping[i]: mod._route_weights=w
        hooks.append(layer.register_forward_pre_hook(hook))
    if method=="full":
        for _raw,m in modules:
            _module=m
            def forward(self,x,*args,_base=_module.base_layer,_module=_module,**kwargs):
                kwargs.pop("adapter_names",None); result=_base(x,*args,**kwargs); w=_module._route_weights
                for j,name in enumerate(order):
                    cast=_module._cast_input_dtype(x,_module.lora_A[name].weight.dtype); result=result+_module.lora_B[name](_module.lora_A[name](_module.lora_dropout[name](cast)))*_module.scaling[name]*w[...,j:j+1].to(result.dtype)
                return result
            m.forward=forward.__get__(m,type(m))
    else:
        cfg=load_json(group_root(group)/"banks"/f"q{q}_bf16"/"core_config.json"); tensors=load_file(str(group_root(group)/"banks"/f"q{q}_bf16"/"core_bank.safetensors"),device="cpu"); items={x["name"]:x for x in cfg["modules"]}
        for raw,m in modules:
            name=canonical_module_name(raw); p=items[name]["tensor_prefix"]; dev,dtype=m.base_layer.weight.device,m.base_layer.weight.dtype
            m.register_buffer("_u",tensors[f"{p}.U"].to(dev,dtype=dtype),persistent=False); m.register_buffer("_v",tensors[f"{p}.V"].to(dev,dtype=dtype),persistent=False); m.register_buffer("_c",tensors[f"{p}.C"].to(dev,dtype=dtype),persistent=False)
            _module=m
            def forward(self,x,*args,_base=_module.base_layer,_module=_module,**kwargs):
                kwargs.pop("adapter_names",None); result=_base(x,*args,**kwargs); z=torch.matmul(x.to(_module._v.dtype),_module._v); mix=torch.einsum("...r,tlr,...t->...l",z,_module._c,_module._route_weights.to(z.dtype)); return result+torch.matmul(mix,_module._u.T).to(result.dtype)
            m.forward=forward.__get__(m,type(m)); m.lora_A.clear();m.lora_B.clear();m.lora_dropout.clear()
    return hooks
def numeric(text):
    vals=re.findall(r"[-+]?\d+(?:\.\d+)?",text.replace(",","")); return float(vals[-1]) if vals else None
def score(name,row,text):
    if name.lower()=="aqua":
        found=re.findall(r"\b([A-E])\b",text.upper()); return bool(found and found[-1]==str(row.get("answer","" )).strip().upper())
    p=numeric(text); 
    try: return p is not None and abs(p-float(row.get("answer")))<=0.001
    except Exception: return False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); ap.add_argument("--method",choices=["full","core"],required=True); ap.add_argument("--q",type=int,default=16); ap.add_argument("--limit",type=int,default=0); args=ap.parse_args()
    out=group_root(args.group)/"quality"/("independent_full" if args.method=="full" else f"core_q{args.q}"); out.mkdir(parents=True,exist_ok=True)
    tok,model=load_model(args.group); hooks=install(model,args.group,args.method,args.q); all_metrics={}; started=time.time()
    try:
        for name in CFG["quality"]["datasets"]:
            rows=json.loads((DATA_ROOT/"math_commonsense"/name/"test.json").read_text(encoding="utf-8")); rows=rows[:args.limit] if args.limit else rows; correct=0; records=[]; path=out/f"{name}.jsonl"
            with path.open("w",encoding="utf-8") as h:
                for i,row in enumerate(rows):
                    enc=tok(fmt(row),return_tensors="pt",truncation=True,max_length=512,add_special_tokens=True); enc={k:v.to("cuda:0") for k,v in enc.items()}; input_len=enc["input_ids"].shape[-1]
                    with torch.inference_mode(): ids=model.generate(**enc,max_new_tokens=int(CFG["quality"]["max_new_tokens"]),do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
                    text=tok.decode(ids[0,input_len:].detach().cpu().tolist(),skip_special_tokens=True); ok=score(name,row,text); correct+=int(ok); h.write(json.dumps({"index":i,"correct":ok,"text":text},ensure_ascii=False)+"\n")
                    if (i+1)%25==0: print(f"[QUALITY] method={args.method} dataset={name} rows={i+1}/{len(rows)} correct={correct}",flush=True)
            all_metrics[name]={"correct":correct,"rows":len(rows),"accuracy":correct/len(rows) if rows else 0}
        write_json(out/"metrics.json",{"status":"PASS","group":args.group,"method":args.method,"metrics":all_metrics,"elapsed_seconds":time.time()-started})
    finally:
        for h in hooks:h.remove()
        del model;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__": main()
