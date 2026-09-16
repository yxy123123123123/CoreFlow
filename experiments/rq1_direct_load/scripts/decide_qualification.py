#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.common import config,now_utc,paths,sha256_file,write_json
def main():
    cfg,p=config(),paths();src=p['results']/'qualification'/'full_seed41'/'COMPLETE.json'
    if not src.is_file(): raise RuntimeError('Qualification is incomplete')
    run=json.loads(src.read_text(encoding='utf-8'));q=cfg['qualification'];passed=run['rows']==q['rows'] and run['correct']>=q['minimum_correct']
    decision=q['decision_pass'] if passed else q['decision_fail'];payload={'status':decision,'protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),
        'correct':run['correct'],'rows':run['rows'],'pass_at_1':run['pass_at_1'],'minimum_correct':q['minimum_correct'],'qualification_complete_sha256':sha256_file(src),
        'next_stage_allowed':passed,'alternate_dataset_search_forbidden':True}
    write_json(p['reports']/'qualification_decision.json',payload);print(json.dumps(payload,sort_keys=True))
    if not passed: raise SystemExit(2)
if __name__=='__main__':main()
