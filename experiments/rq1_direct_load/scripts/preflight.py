#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.io import adapter_checkpoint_path
from scripts.common import config,now_utc,paths,sha256_file,write_json

def main():
    cfg,p=config(),paths(); report=p['reports']; report.mkdir(parents=True,exist_ok=True)
    if (report/'E0_SEAL.json').exists(): raise RuntimeError('Protocol already sealed')
    for path in [p['model'],p['asset_root']/'LoRAs',p['vendor_root'],p['m2_root'],p['formal_v2']]:
        if not path.exists(): raise FileNotFoundError(path)
    embedded={}
    for rel,expected in cfg['embedded'].items():
        observed=sha256_file(ROOT/rel)
        if observed!=expected: raise ValueError(f'Embedded SHA-256 mismatch: {rel}: {observed}')
        embedded[rel]=observed
    experts={}
    for name in cfg['expert_order']:
        spec=cfg['experts'][name]; root=p['asset_root']/'LoRAs'/spec['relative']; ckpt=adapter_checkpoint_path(root)
        got={'config':sha256_file(root/'adapter_config.json'),'checkpoint':sha256_file(ckpt)}
        if got!={'config':spec['adapter_config_sha256'],'checkpoint':spec['checkpoint_sha256']}: raise ValueError(f'Expert hash mismatch {name}: {got}')
        experts[name]=got
    gates={}
    for seed in cfg['gate']['formal_seeds']:
        path=p['m2_root']/cfg['gate']['relative_pattern'].format(seed=seed); got=sha256_file(path)
        if got!=cfg['gate']['sha256'][str(seed)]: raise ValueError(f'Gate hash mismatch seed={seed}: {got}')
        gates[str(seed)]=got
    core=p['m2_root']/cfg['banks']['core_relative']/'core_bank.safetensors'
    isvd=p['formal_v2']/cfg['banks']['isvd_relative_from_formal_v2']/'isvd_bank.safetensors'
    if sha256_file(core)!=cfg['banks']['core_sha256']: raise ValueError('Core bank hash mismatch')
    if sha256_file(isvd)!=cfg['banks']['isvd_sha256']: raise ValueError('ISVD bank hash mismatch')
    dep=subprocess.run([sys.executable,'-c','import torch,transformers,peft,flask,safetensors,numpy'],capture_output=True,text=True)
    if dep.returncode: raise RuntimeError(dep.stderr[-2000:])
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Exactly one visible CUDA GPU is required')
    gpu=torch.cuda.get_device_name(0)
    if cfg['hardware']['required_gpu_substring'] not in gpu: raise RuntimeError(gpu)
    payload={'status':'PASS','protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),'gpu':gpu,'embedded':embedded,'experts':experts,'gates':gates,
             'core_bank':{'path':str(core),'sha256':sha256_file(core)},'isvd_bank':{'path':str(isvd),'sha256':sha256_file(isvd)},'python':sys.version}
    write_json(report/'preflight.json',payload);print(json.dumps({'status':'PASS','gpu':gpu,'experts':5,'gates':3}))
if __name__=='__main__':main()
