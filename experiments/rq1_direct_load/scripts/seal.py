#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.common import config,now_utc,paths,sha256_file,write_json
def main():
    cfg,p=config(),paths();r=p['reports'];seal=r/'E0_SEAL.json'
    for name in ('preflight.json','data_audit.json'):
        if not (r/name).is_file(): raise RuntimeError(f'Missing {name}')
    if seal.exists():
        old=json.loads(seal.read_text(encoding='utf-8'))
        if old.get('status')!='SEALED_BEFORE_QUALIFICATION_OUTPUTS': raise RuntimeError('Invalid seal')
        print(json.dumps({'status':'PASS','reused':True,'seal':str(seal)}));return
    if p['results'].exists() and any(x.is_file() for x in p['results'].rglob('*')): raise RuntimeError('Results exist before seal')
    payload={'status':'SEALED_BEFORE_QUALIFICATION_OUTPUTS','protocol_id':cfg['protocol_id'],'timestamp_utc':now_utc(),
             'config_sha256':sha256_file(ROOT/'config'/'stage1_protocol.json'),'package_manifest_sha256':sha256_file(ROOT/'PACKAGE_MANIFEST.sha256'),
             'preflight_sha256':sha256_file(r/'preflight.json'),'data_audit_sha256':sha256_file(r/'data_audit.json'),
             'qualification_floor':cfg['qualification'],'formal_generation_forbidden':True,'q185_k5_frozen':True}
    write_json(seal,payload);print(json.dumps({'status':payload['status'],'seal':str(seal)}))
if __name__=='__main__':main()
