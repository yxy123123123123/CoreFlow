from __future__ import annotations
import argparse, json, gc, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import CFG, MODEL, RESULT, group_root, load_json, manifest_path, write_json

def rows(root,n):
    data=json.loads((root/"math_14k"/"validation.json").read_text(encoding="utf-8"))[:n]
    return data
def text_pair(row):
    if row.get("input"):
        prompt=("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n### Instruction:\n"+row["instruction"]+
                "\n\n### Input:\n"+row["input"]+"\n\n### Response:")
    else:
        prompt=("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"+row["instruction"]+"\n\n### Response:")
    return prompt, prompt+row.get("output","")
def load_base():
    return AutoModelForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,device_map={"":"cuda:0"},low_cpu_mem_usage=True,local_files_only=True,attn_implementation="sdpa").eval()
def loss_for_adapter(adapter, rows, tok):
    base=load_base(); model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True,low_cpu_mem_usage=True,autocast_adapter_dtype=False).eval(); vals=[]
    with torch.inference_mode():
        for idx,row in enumerate(rows):
            _,full=text_pair(row); enc=tok(full,return_tensors="pt",truncation=True,max_length=CFG["training"]["max_length"],add_special_tokens=True); ids=enc["input_ids"].to("cuda:0")
            vals.append(float(model(input_ids=ids,labels=ids).loss.detach().cpu()))
            if (idx+1)%32==0: print(f"[GATE] adapter={adapter.name} rows={idx+1}/{len(rows)}",flush=True)
    del model,base; gc.collect(); torch.cuda.empty_cache(); return torch.tensor(vals,dtype=torch.float32)
def features(rows,tok):
    base=load_base(); out=[]
    with torch.inference_mode():
        for idx,row in enumerate(rows):
            prompt,_=text_pair(row); enc=tok(prompt,return_tensors="pt",truncation=True,max_length=CFG["training"]["max_length"],add_special_tokens=True); ids=enc["input_ids"].to("cuda:0"); mask=enc["attention_mask"].to("cuda:0")
            hs=base(input_ids=ids,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states[1:]
            pos=int(mask.sum().item())-1; out.append(torch.stack([h[0,pos].float().cpu() for h in hs]))
            if (idx+1)%32==0: print(f"[GATE] features rows={idx+1}/{len(rows)}",flush=True)
    del base; gc.collect(); torch.cuda.empty_cache(); return torch.stack(out)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); args=ap.parse_args(); group=args.group
    target=group_root(group)/"gate"; target.mkdir(parents=True,exist_ok=True); final=target/"gate.safetensors"
    if final.is_file(): print(json.dumps({"status":"SKIP_PASS","path":str(final)})); return
    n=int(CFG["gate_calibration"]["rows"]); data_root=Path(__import__('common').DATA_ROOT); data=rows(data_root,n); tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True,padding_side="left")
    manifest=load_json(manifest_path(group)); losses=[]; adapters=[]
    for rec in manifest["records"]:
        path=Path(rec["path"]); adapters.append(path); losses.append(loss_for_adapter(path,data,tok))
    loss=torch.stack(losses,dim=1); tau=float(CFG["gate_calibration"]["temperature"]); target_probs=F.softmax(-(loss-loss.mean(dim=1,keepdim=True))/tau,dim=1)
    x=features(data,tok).to("cuda:0"); y=target_probs.to("cuda:0"); w=torch.zeros((36,8,4096),device="cuda:0",requires_grad=True); b=torch.zeros((36,8),device="cuda:0",requires_grad=True); opt=torch.optim.Adam([w,b],lr=float(CFG["gate_calibration"]["learning_rate"]),weight_decay=float(CFG["gate_calibration"]["weight_decay"]))
    target_layers = y[:,None,:].expand(-1, x.shape[1], -1)
    for epoch in range(int(CFG["gate_calibration"]["epochs"])):
        opt.zero_grad(); logits=torch.einsum("nld,led->nle",x,w)+b; logp=F.log_softmax(logits,dim=-1); loss_train=F.kl_div(logp,target_layers,log_target=False,reduction="none").sum(dim=-1).mean(); loss_train.backward(); opt.step()
    with torch.no_grad(): pred=(torch.einsum("nld,led->nle",x,w)+b).mean(dim=1).argmax(dim=1); labels=target_probs.to(pred.device).argmax(dim=1); acc=float((pred==labels).float().mean())
    save_file({"gate.weight":w.detach().cpu().contiguous(),"gate.bias":b.detach().cpu().contiguous()},str(final),metadata={"protocol_id":CFG["protocol_id"],"group":group,"rows":str(n)})
    write_json(target/"gate_calibration.json",{"status":"PASS","group":group,"rows":n,"loss_matrix_shape":list(loss.shape),"pseudo_label_accuracy":acc,"label_rule":CFG["gate_calibration"]["label_rule"],"target_rule":CFG["gate_calibration"]["target_rule"],"adapter_paths":[str(x) for x in adapters]})
    print(json.dumps({"status":"PASS","group":group,"gate":str(final),"pseudo_label_accuracy":acc}))
if __name__=="__main__": main()
