#!/usr/bin/env python3
from __future__ import annotations
import json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.runtime import isvd_delta,packed_delta
from scripts.common import config,now_utc,paths,sha256_file,write_json

def metrics(candidate,reference):
    c,r=candidate.float(),reference.float();rn=float(torch.linalg.vector_norm(r));cn=float(torch.linalg.vector_norm(c-r))
    if rn<1e-12: nrmse=0.0 if cn<1e-12 else float('inf');cos=1.0 if cn<1e-12 else 0.0
    else:
        nrmse=cn/rn; cos=float(torch.nn.functional.cosine_similarity(c.reshape(1,-1),r.reshape(1,-1)).item())
    return {'nrmse':nrmse,'cosine':cos,'max_abs':float((c-r).abs().max())}

def main():
    cfg,p=config(),paths();decision=p['reports']/'qualification_decision.json'
    if not decision.is_file() or json.loads(decision.read_text(encoding='utf-8')).get('status')!='QUALIFICATION_PASS': raise RuntimeError('Qualification did not pass')
    final=p['banks']/'isvd_compact';partial=final.with_name(final.name+'.partial');report=p['reports']/'e1_isvd_correctness.json'
    if report.is_file() and json.loads(report.read_text(encoding='utf-8')).get('status')=='E1_ISVD_PASS': print(json.dumps({'status':'PASS','reused':True}));return
    if final.exists() or partial.exists(): raise RuntimeError(f'Retained compact bank exists: {final} or {partial}')
    partial.mkdir(parents=True);source=p['formal_v2']/cfg['banks']['isvd_relative_from_formal_v2'];source_cfg=json.loads((source/'isvd_config.json').read_text(encoding='utf-8'))
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors={};modules=[];compact_params=0;padded_params=0
    with safe_open(str(source/'isvd_bank.safetensors'),framework='pt',device='cpu') as h:
        for idx,item in enumerate(source_cfg['modules']):
            prefix=item['tensor_prefix'];b=h.get_tensor(prefix+'.B');v=h.get_tensor(prefix+'.V');ranks=[int(x) for x in item['expert_ranks']]
            bp=[];vp=[];owner=[]
            for expert,rank in enumerate(ranks):
                if rank>0: bp.append(b[expert,:,:rank]);vp.append(v[expert,:,:rank]);owner.extend([expert]*rank)
            bcat=torch.cat(bp,dim=1).contiguous() if bp else torch.empty((item['out_features'],0),dtype=b.dtype)
            vcat=torch.cat(vp,dim=1).contiguous() if vp else torch.empty((item['in_features'],0),dtype=v.dtype)
            cp=f'm{idx:03d}';tensors[cp+'.B']=bcat;tensors[cp+'.V']=vcat;tensors[cp+'.owner']=torch.tensor(owner,dtype=torch.int64)
            compact_params+=sum(ranks)*(item['in_features']+item['out_features']);padded_params+=len(ranks)*item['max_rank']*(item['in_features']+item['out_features'])
            modules.append({**item,'compact_tensor_prefix':cp,'packed_rank':len(owner)})
    save_file(tensors,str(partial/'isvd_compact.safetensors'),metadata={'format':'coreflow-isvd-compact-v1','source_sha256':cfg['banks']['isvd_sha256']})
    compact_cfg={'format':'coreflow-isvd-compact-v1','schema_version':1,'expert_order':cfg['expert_order'],'source_isvd_sha256':cfg['banks']['isvd_sha256'],
        'modules':modules,'compact_parameters':compact_params,'legacy_padded_parameters':padded_params,'padding_parameters_eliminated':padded_params-compact_params}
    write_json(partial/'isvd_compact_config.json',compact_cfg)
    # Frozen representatives cover all seven target-module types and model depth.
    target_names=[
        'model.layers.0.self_attn.q_proj','model.layers.5.self_attn.k_proj','model.layers.10.self_attn.v_proj',
        'model.layers.15.self_attn.o_proj','model.layers.20.mlp.gate_proj','model.layers.25.mlp.up_proj','model.layers.31.mlp.down_proj',
    ]
    by_name={item['name']:idx for idx,item in enumerate(modules)}
    missing=[name for name in target_names if name not in by_name]
    if missing: raise ValueError(f'Frozen representative modules missing: {missing}')
    indices=[by_name[name] for name in target_names];cases=[];fp32=[];deterministic=True;started=time.time()
    with safe_open(str(source/'isvd_bank.safetensors'),framework='pt',device='cpu') as ph, safe_open(str(partial/'isvd_compact.safetensors'),framework='pt',device='cpu') as ch:
        for pos,idx in enumerate(indices):
            item=modules[idx];prefix=item['tensor_prefix'];cp=item['compact_tensor_prefix'];ranks=torch.tensor(item['expert_ranks'],device='cuda')
            bpad=ph.get_tensor(prefix+'.B').cuda();vpad=ph.get_tensor(prefix+'.V').cuda();bcat=ch.get_tensor(cp+'.B').cuda();vcat=ch.get_tensor(cp+'.V').cuda();owner=ch.get_tensor(cp+'.owner').cuda()
            for seed in cfg['e1_correctness']['gate_seeds']:
                gen=torch.Generator(device='cuda').manual_seed(seed*1000+idx)
                for block in cfg['e1_correctness']['token_blocks']:
                    x=torch.randn((block,item['in_features']),generator=gen,device='cuda',dtype=torch.bfloat16);g=torch.randn((block,5),generator=gen,device='cuda',dtype=torch.bfloat16)
                    ref=isvd_delta(x,g,bpad,vpad,ranks);cand=packed_delta(x,g,bcat,vcat,owner);m=metrics(cand,ref);cases.append({'module':item['name'],'block':block,'seed':seed,**m})
                    deterministic=deterministic and torch.equal(cand,packed_delta(x,g,bcat,vcat,owner))
            del bpad,vpad,bcat,vcat,owner,ranks;torch.cuda.empty_cache()
        # 100 deterministic FP32 draws over representative modules, small blocks.
        for n in range(cfg['e1_correctness']['fp32_samples']):
            idx=indices[n%len(indices)];item=modules[idx];prefix=item['tensor_prefix'];cp=item['compact_tensor_prefix']
            bpad=ph.get_tensor(prefix+'.B').cuda().float();vpad=ph.get_tensor(prefix+'.V').cuda().float();bcat=ch.get_tensor(cp+'.B').cuda().float();vcat=ch.get_tensor(cp+'.V').cuda().float();owner=ch.get_tensor(cp+'.owner').cuda();ranks=torch.tensor(item['expert_ranks'],device='cuda')
            gen=torch.Generator(device='cuda').manual_seed(20260802+n);x=torch.randn((1,item['in_features']),generator=gen,device='cuda');g=torch.randn((1,5),generator=gen,device='cuda')
            fp32.append(metrics(packed_delta(x,g,bcat,vcat,owner),isvd_delta(x,g,bpad,vpad,ranks)));del bpad,vpad,bcat,vcat,owner,ranks
    nrmse=sorted(x['nrmse'] for x in cases);median=nrmse[len(nrmse)//2];max_n=max(nrmse);min_cos=min(x['cosine'] for x in cases)
    fp32_max=max(x['max_abs'] for x in fp32);threshold=cfg['e1_correctness'];passed=(median<=threshold['bf16_nrmse_median_max'] and max_n<=threshold['bf16_nrmse_max'] and min_cos>=threshold['bf16_cosine_min'] and fp32_max<=threshold['fp32_atol']+threshold['fp32_rtol'] and deterministic)
    payload={'status':'E1_ISVD_PASS' if passed else 'E1_ISVD_FAIL','protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),'source_bank_sha256':sha256_file(source/'isvd_bank.safetensors'),
        'compact_bank_sha256':sha256_file(partial/'isvd_compact.safetensors'),'compact_config_sha256':sha256_file(partial/'isvd_compact_config.json'),
        'compact_parameters':compact_params,'legacy_padded_parameters':padded_params,'padding_parameters_eliminated':padded_params-compact_params,
        'representative_indices':indices,'bf16_cases':len(cases),'bf16_nrmse_median':median,'bf16_nrmse_max':max_n,'bf16_cosine_min':min_cos,
        'fp32_samples':len(fp32),'fp32_max_abs':fp32_max,'self_deterministic':deterministic,'cases':cases,'elapsed_seconds':time.time()-started,
        'compact_is_independent_method':True,'bitwise_equivalent_to_padded':False}
    write_json(report,payload)
    if not passed: print(json.dumps({'status':payload['status'],'median':median,'max':max_n,'cosine':min_cos,'fp32':fp32_max}));raise SystemExit(1)
    partial.rename(final);print(json.dumps({'status':payload['status'],'compact_bank':str(final),'bf16_cases':len(cases)}))
if __name__=='__main__':main()
