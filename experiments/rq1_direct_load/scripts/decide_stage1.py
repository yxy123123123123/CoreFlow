#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.common import config,now_utc,paths,write_json
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def main():
    cfg,p=config(),paths();q=read(p['reports']/'qualification_decision.json');e1=read(p['reports']/'e1_isvd_correctness.json') if (p['reports']/'e1_isvd_correctness.json').is_file() else {'status':'E1_ISVD_FAIL'};e1d=read(p['reports']/'e1d_direct_load.json')
    if q['status']!='QUALIFICATION_PASS': status='NO_GO_DATASET_FLOOR'
    elif e1d['status']!='E1D_PASS': status='NO_GO_DIRECT_LOAD'
    elif e1['status']=='E1_ISVD_PASS': status='STAGE1_PASS_THREE_METHODS'
    else: status='STAGE1_PASS_CORE_FULL_ONLY'
    payload={'status':status,'protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),'qualification':q['status'],'isvd':e1['status'],'direct_load':e1d['status'],
             'future_e2a_allowed':status.startswith('STAGE1_PASS'),'future_isvd_allowed':status=='STAGE1_PASS_THREE_METHODS','formal164_opened':False}
    write_json(p['reports']/'stage1_decision.json',payload);print(json.dumps(payload,sort_keys=True))
    if not status.startswith('STAGE1_PASS'): raise SystemExit(1)
if __name__=='__main__':main()
