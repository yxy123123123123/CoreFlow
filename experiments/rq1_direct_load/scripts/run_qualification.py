#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.lcb_eval import evaluate,prompt_for
from scripts.common import config,paths,request_json,wait_health,write_json

def main():
    cfg,p=config(),paths();seal=p['reports']/'E0_SEAL.json'
    if not seal.is_file(): raise RuntimeError('Run seal first')
    final=p['results']/'qualification'/'full_seed41';partial=final.with_name(final.name+'.partial')
    if (final/'COMPLETE.json').is_file(): print(f'[SKIP] {final}');return
    if final.exists() or partial.exists(): raise RuntimeError(f'Retained partial/final output exists: {partial} or {final}')
    partial.mkdir(parents=True); seed=cfg['qualification']['gate_seed'];gate=p['m2_root']/cfg['gate']['relative_pattern'].format(seed=seed)
    cmd=[sys.executable,str(ROOT/'scripts'/'serve_full.py'),'--model',str(p['model']),'--asset-root',str(p['asset_root']),'--vendor-root',str(p['vendor_root']),
         '--gate',str(gate),'--port','6800','--max-input-tokens',str(cfg['generation']['max_input_tokens']),'--max-new-tokens',str(cfg['generation']['max_new_tokens'])]
    (partial/'command.txt').write_text(' '.join(cmd)+'\n',encoding='utf-8',newline='\n');log=(partial/'server.log').open('w',encoding='utf-8',newline='\n')
    server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,text=True)
    rows=[json.loads(x) for x in (p['data']/cfg['dataset']['qualification_file']).read_text(encoding='utf-8').splitlines() if x.strip()]
    evaluated=[];started=time.time()
    try:
        write_json(partial/'health.json',wait_health('http://127.0.0.1:6800/health',server))
        with (partial/'generations.jsonl').open('w',encoding='utf-8',newline='\n') as gf,(partial/'executor_results.jsonl').open('w',encoding='utf-8',newline='\n') as ef:
            for i,row in enumerate(rows):
                prompt=prompt_for(row);resp=request_json('http://127.0.0.1:6800/infer',{'prompt':prompt,'seed':seed*100000+i},timeout=1800);raw=resp['answer']
                gen={'question_id':row['question_id'],'order_index':i,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'raw_output':raw,'timing':resp['timing']}
                gf.write(json.dumps(gen,ensure_ascii=False,sort_keys=True)+'\n');gf.flush()
                ev=evaluate(row,raw,cfg['generation']['per_test_timeout_seconds'],cfg['generation']['per_problem_global_timeout_seconds'])
                item={'question_id':row['question_id'],'order_index':i,'raw_output_sha256':hashlib.sha256(raw.encode()).hexdigest(),'evaluation':ev}
                ef.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n');ef.flush();evaluated.append(item)
                if i==0 or (i+1)%5==0 or i+1==len(rows):
                    correct=sum(x['evaluation']['passed'] for x in evaluated);print(f'[PROGRESS] qualification rows={i+1}/{len(rows)} correct={correct} elapsed_min={(time.time()-started)/60:.1f}',flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(30)
            except subprocess.TimeoutExpired:server.kill();server.wait(10)
        log.close()
    correct=sum(x['evaluation']['passed'] for x in evaluated);counts=Counter(x['evaluation']['status'] for x in evaluated)
    summary={'status':'PASS','method':'full_vectorized','seed':seed,'rows':len(evaluated),'correct':correct,'pass_at_1':correct/len(evaluated),
             'status_counts':dict(counts),'elapsed_seconds':time.time()-started}
    write_json(partial/'summary.json',summary);write_json(partial/'COMPLETE.json',summary);partial.rename(final);print(json.dumps(summary,sort_keys=True))
if __name__=='__main__':main()
