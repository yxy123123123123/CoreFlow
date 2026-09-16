#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.direct_load import install_core_direct,install_isvd_compact_direct,load_base_with_gate
from coreflow.loraflow import load_loraflow_model,move_unique_tensors_
from coreflow.runtime import install_coreflow,install_full_vectorized,install_isvd_compact
from scripts.common import config,paths,write_json

def signature(tok,model,prompt,n):
    inputs=tok(prompt,return_tensors='pt').to('cuda');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    with torch.inference_mode():
        logits=model(**inputs).logits[:,-1,:].float();values,indices=torch.topk(logits,64,dim=-1)
        out=model.generate(**inputs,do_sample=False,temperature=1.0,top_p=1.0,max_new_tokens=n,use_cache=True)
    torch.cuda.synchronize();return {'tokens':out[0,inputs['input_ids'].shape[-1]:].cpu().tolist(),'top64_indices':indices[0].cpu().tolist(),
        'top64_values':[float(x) for x in values[0].cpu()],'finite':bool(torch.isfinite(logits).all()),'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
        'peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),'elapsed_seconds':time.perf_counter()-started}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=('full_direct','core_direct','core_legacy','isvd_direct','isvd_legacy'),required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    cfg,p=config(),paths();order=cfg['expert_order'];gate=p['m2_root']/cfg['gate']['relative_pattern'].format(seed=41);method=a.method
    if method in {'full_direct','core_legacy','isvd_legacy'}:
        tok,model,load=load_loraflow_model(p['model'],p['asset_root'],p['vendor_root'],adapter_order=order,gate_path=gate,dtype=torch.bfloat16,device='cpu')
        if method=='full_direct': install=install_full_vectorized(model,expected_expert_order=order,release_sources=True)
        elif method=='core_legacy': install=install_coreflow(model,p['m2_root']/cfg['banks']['core_relative'],expected_expert_order=order,release_sources=True)
        else: install=install_isvd_compact(model,p['formal_v2']/cfg['banks']['isvd_relative_from_formal_v2'],expected_expert_order=order,release_sources=True)
    else:
        tok,model,load=load_base_with_gate(p['model'],p['vendor_root'],gate,order,device='cpu')
        if method=='core_direct': install=install_core_direct(model,p['m2_root']/cfg['banks']['core_relative'],order)
        else: install=install_isvd_compact_direct(model,p['banks']/'isvd_compact',order)
    material=move_unique_tensors_(model,'cuda',torch.bfloat16);model.eval();torch.cuda.empty_cache();torch.cuda.synchronize()
    payload={'status':'PASS','method':method,'load':load,'install':install,'materialization':material,
             'signature':signature(tok,model,cfg['e1d']['prompt'],cfg['e1d']['short_generation_tokens'])}
    write_json(a.output,payload);print(json.dumps({'status':'PASS','method':method,'tokens':payload['signature']['tokens'],'peak':payload['signature']['peak_allocated_bytes']}))
if __name__=='__main__':main()
