#!/usr/bin/env python3
from __future__ import annotations
import argparse,random,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.loraflow import load_loraflow_model,move_unique_tensors_
from coreflow.runtime import install_full_vectorized

def main():
    ap=argparse.ArgumentParser()
    for name in ('model','asset_root','vendor_root','gate'): ap.add_argument('--'+name.replace('_','-'),required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--max-input-tokens',type=int,required=True);ap.add_argument('--max-new-tokens',type=int,required=True);a=ap.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    order=['zh','ru','es','math','code']
    tok,model,load=load_loraflow_model(a.model,a.asset_root,a.vendor_root,adapter_order=order,gate_path=a.gate,dtype=torch.bfloat16,device='cpu')
    install=install_full_vectorized(model,expected_expert_order=order,release_sources=True)
    material=move_unique_tensors_(model,device='cuda',dtype=torch.bfloat16);model.eval();torch.cuda.empty_cache();torch.cuda.synchronize()
    from flask import Flask,jsonify,request
    app=Flask(__name__)
    @app.get('/health')
    def health(): return jsonify({'status':'ok','kind':'full_vectorized','load':load,'install':install,'materialization':material,'gpu':torch.cuda.get_device_name(0)})
    @app.post('/infer')
    def infer():
        payload=request.get_json(force=True);seed=int(payload['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        inputs=tok(str(payload['prompt']),max_length=a.max_input_tokens,truncation=True,return_tensors='pt').to('cuda');n=int(inputs['input_ids'].shape[-1])
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(): out=model.generate(**inputs,do_sample=False,temperature=1.0,top_p=1.0,max_new_tokens=a.max_new_tokens,use_cache=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started;new=int(out.shape[-1]-n)
        return jsonify({'answer':tok.decode(out[0,n:],skip_special_tokens=True),'timing':{'input_tokens':n,'new_tokens':new,'total_latency_seconds':elapsed,
            'tokens_per_second_total':new/max(elapsed,1e-9),'peak_memory_allocated_bytes':int(torch.cuda.max_memory_allocated()),'peak_memory_reserved_bytes':int(torch.cuda.max_memory_reserved())}})
    app.run(host='127.0.0.1',port=a.port,debug=False,threaded=False)
if __name__=='__main__':main()
