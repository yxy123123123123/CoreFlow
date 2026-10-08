from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts")); sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import CFG, REPORT, group_root, load_json, manifest_path, sha256, write_json
from coreflow.decompose import build_banks
from coreflow.io import adapter_checkpoint_path, load_adapter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); args=ap.parse_args(); group=args.group
    manifest=load_json(manifest_path(group)); records=manifest["records"]
    groups=[]; metadata=[]
    for rec in records:
        path=Path(rec["path"]); cfg,mods=load_adapter(path); groups.append(mods); metadata.append({"name":f"expert_{rec['expert']:02d}","seed":rec["seed"],"checkpoint_sha256":sha256(adapter_checkpoint_path(path)),"adapter_config_sha256":sha256(path/"adapter_config.json")})
    out=group_root(group)/"banks"; out.mkdir(parents=True,exist_ok=True)
    build_banks(groups,[f"expert_{i:02d}" for i in range(1,9)],CFG["coreflow"]["candidate_q"],out,save_dtype=torch.bfloat16,balanced=True,device="cuda:0",source_metadata=metadata,bank_label_suffix="_bf16")
    banks={}
    for q in CFG["coreflow"]["candidate_q"]:
        p=out/f"q{q}_bf16"; banks[str(q)]={"path":str(p),"sha256":sha256(p/"core_bank.safetensors"),"config":load_json(p/"core_config.json")}
    write_json(group_root(group)/"bank_manifest.json",{"status":"PASS","group":group,"banks":banks,"candidate_q":CFG["coreflow"]["candidate_q"]})
    print(json.dumps({"status":"PASS","group":group,"q":list(banks)}))
if __name__=="__main__": main()
