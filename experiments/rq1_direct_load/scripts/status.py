#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.common import paths
def value(path,key='status'):
    if not path.is_file(): return 'MISSING'
    try:return json.loads(path.read_text(encoding='utf-8')).get(key,'PRESENT')
    except:return 'INVALID_JSON'
def main():
    p=paths();r=p['reports'];print('=== CoreFlow submission-final Stage-1 ===')
    for name in ['preflight.json','data_audit.json','E0_SEAL.json','qualification_decision.json','e1_isvd_correctness.json','e1d_direct_load.json','stage1_decision.json']:
        print(f'{name}: {value(r/name)}')
    q=p['results']/'qualification'/'full_seed41';print('\n=== qualification ===')
    if (q/'COMPLETE.json').is_file():
        x=json.loads((q/'COMPLETE.json').read_text(encoding='utf-8'));print(f"COMPLETE correct={x['correct']}/{x['rows']} elapsed_min={x['elapsed_seconds']/60:.1f}")
    elif q.with_name(q.name+'.partial').exists():
        f=q.with_name(q.name+'.partial')/'executor_results.jsonl';rows=sum(1 for _ in f.open(encoding='utf-8')) if f.is_file() else 0;print(f'PARTIAL rows={rows}/64')
    else: print('PENDING')
    print('\n=== E1D method files ===')
    for method in ['core_direct','core_legacy','full_direct','isvd_direct','isvd_legacy']:
        print(f'{method}: {"COMPLETE" if (p["results"] / "e1d" / (method+".json")).is_file() else "PENDING"}')
if __name__=='__main__':main()
