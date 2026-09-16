#!/usr/bin/env python3
from __future__ import annotations
import json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coreflow.lcb_eval import evaluate,extract_code,prompt_for
from scripts.common import config,now_utc,paths,sha256_file,write_json

def load_rows(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
    cfg,p=config(),paths()
    if not (p['reports']/'preflight.json').is_file(): raise RuntimeError('Run preflight first')
    if (p['reports']/'E0_SEAL.json').exists(): raise RuntimeError('Protocol already sealed')
    p['data'].mkdir(parents=True,exist_ok=True); files={}
    for key in ('qualification_file','formal_file','reserve_file','split_manifest'):
        name=cfg['dataset'][key]; src=ROOT/'data'/name; dst=p['data']/name
        if dst.exists() and sha256_file(dst)!=sha256_file(src): raise RuntimeError(f'Refusing to overwrite mismatched data: {dst}')
        if not dst.exists(): shutil.copy2(src,dst)
        files[name]={'sha256':sha256_file(dst),'bytes':dst.stat().st_size}
    splits={name:load_rows(p['data']/cfg['dataset'][name]) for name in ('qualification_file','formal_file','reserve_file')}
    expected={'qualification_file':64,'formal_file':164,'reserve_file':16}
    for name,rows in splits.items():
        if len(rows)!=expected[name] or len({r['question_id'] for r in rows})!=len(rows): raise ValueError(f'Bad split: {name}')
        if any(r['difficulty'] not in {'easy','medium'} for r in rows): raise ValueError(f'Hard row in {name}')
    ids=[{r['question_id'] for r in splits[name]} for name in splits]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i+1,3)): raise ValueError('Split overlap')
    # CPU-only evaluator/postprocessor smoke; no benchmark model output.
    code='```python\na,b=map(int,input().split())\nprint(a+b)\n```'
    sample={'question_id':'SMOKE','question_content':'sum','starter_code':'','public_test_cases':json.dumps([{'input':'2 3\n','output':'5\n','testtype':'stdin'}]),
            'private_test_cases':json.dumps([{'input':'-1 4\n','output':'3\n','testtype':'stdin'}]),'metadata':'{}'}
    smoke=evaluate(sample,code,2,20)
    if not smoke['passed'] or extract_code('x\n```python\nprint(1)\n```')[0]!='print(1)' or not prompt_for(sample).startswith('### Question:'): raise RuntimeError(smoke)
    payload={'status':'PASS','protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),'files':files,
             'splits':{k:len(v) for k,v in splits.items()},'disjoint':True,'formal_model_accessed':False,'evaluator_smoke':smoke}
    write_json(p['reports']/'data_audit.json',payload);print(json.dumps({'status':'PASS','splits':payload['splits']}))
if __name__=='__main__':main()
