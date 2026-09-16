#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.common import config,now_utc,paths,write_json
def main():
    cfg,p=config(),paths();q=json.loads((p['reports']/'qualification_decision.json').read_text(encoding='utf-8'))
    if q.get('status')!='QUALIFICATION_PASS': raise RuntimeError('Qualification did not pass')
    e1=json.loads((p['reports']/'e1_isvd_correctness.json').read_text(encoding='utf-8')) if (p['reports']/'e1_isvd_correctness.json').is_file() else {'status':'MISSING'}
    final=p['reports']/'e1d_direct_load.json'
    if final.is_file(): print(json.dumps({'status':'PASS','reused':True}));return
    outroot=p['results']/'e1d';outroot.mkdir(parents=True,exist_ok=True)
    methods=['core_direct','core_legacy','full_direct']+(['isvd_direct','isvd_legacy'] if e1.get('status')=='E1_ISVD_PASS' else [])
    runs={}
    for method in methods:
        output=outroot/f'{method}.json';log=outroot/f'{method}.log'
        if not output.is_file():
            cmd=[sys.executable,str(ROOT/'scripts'/'run_e1d_method.py'),'--method',method,'--output',str(output)]
            with log.open('w',encoding='utf-8',newline='\n') as handle:
                done=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,text=True)
            if done.returncode: raise RuntimeError(f'E1D {method} failed; retained log: {log}')
        runs[method]=json.loads(output.read_text(encoding='utf-8'))
    errors=[];max_peak=int(cfg['hardware']['max_direct_load_peak_gib']*1024**3)
    for method,run in runs.items():
        if not run['signature']['finite']: errors.append(f'{method}:nonfinite')
        if run['signature']['peak_allocated_bytes']>max_peak: errors.append(f'{method}:peak')
        if method.endswith('_direct') and method!='full_direct':
            if run['load'].get('source_lora_loaded') is not False or run['install'].get('source_lora_loaded') is not False: errors.append(f'{method}:source_lora')
    comparisons={}
    for direct,legacy in [('core_direct','core_legacy'),('isvd_direct','isvd_legacy')]:
        if direct not in runs: continue
        ds,ls=runs[direct]['signature'],runs[legacy]['signature'];ids=ds['top64_indices']==ls['top64_indices'];maxdiff=max(abs(a-b) for a,b in zip(ds['top64_values'],ls['top64_values'])) if ids else float('inf')
        comparisons[direct]={'reference':legacy,'tokens_identical':ds['tokens']==ls['tokens'],'top64_indices_identical':ids,'top64_value_max_abs':maxdiff}
        if not comparisons[direct]['tokens_identical'] or not ids or maxdiff>0.02: errors.append(f'{direct}:semantic_mismatch')
    payload={'status':'E1D_PASS' if not errors else 'E1D_FAIL','protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),'independent_process_per_path':True,
        'runs':runs,'comparisons':comparisons,'isvd_e1_status':e1.get('status'),'errors':errors}
    write_json(final,payload);print(json.dumps({'status':payload['status'],'methods':methods,'comparisons':comparisons,'errors':errors}))
    if errors: raise SystemExit(1)
if __name__=='__main__':main()
