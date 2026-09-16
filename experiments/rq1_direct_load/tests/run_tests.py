#!/usr/bin/env python3
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):return [json.loads(x) for x in (ROOT/'data'/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def test_protocol():
 c=json.loads((ROOT/'config'/'stage1_protocol.json').read_text(encoding='utf-8'));assert c['protocol_id']=='coreflow-submission-final-stage1-v1';assert c['qualification']['minimum_correct']==6
 assert c['dataset']['qualification_rows']==64 and c['dataset']['formal_rows']==164 and c['dataset']['reserve_rows']==16;assert c['banks']['theoretical_adapter_mac_reduction_pct']==c['banks']['theoretical_parameter_reduction_pct']
 assert len(c['future_e3']['method_order_groups'])==6 and len({tuple(x) for x in c['future_e3']['method_order_groups']})==6
def test_embedded():
 c=json.loads((ROOT/'config'/'stage1_protocol.json').read_text(encoding='utf-8'))
 for rel,sha in c['embedded'].items():assert digest(ROOT/rel)==sha,(rel,digest(ROOT/rel))
def test_splits():
 q,f,r=rows('livecodebench_qualification64.jsonl'),rows('livecodebench_formal164.jsonl'),rows('livecodebench_reserve16.jsonl');assert [len(q),len(f),len(r)]==[64,164,16]
 ids=[{x['question_id'] for x in z} for z in [q,f,r]];assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2];assert all(x['difficulty'] in {'easy','medium'} for x in q+f+r)
def test_no_formal_runner():
 forbidden='livecodebench_formal164.jsonl'
 for p in (ROOT/'scripts').glob('run_*.py'):
  assert forbidden not in p.read_text(encoding='utf-8'),p
 q=(ROOT/'scripts'/'run_qualification.py').read_text(encoding='utf-8');assert "qualification_file" in q and "formal_file" not in q
def test_postprocessor():
 from coreflow.lcb_eval import extract_code
 assert extract_code('text\n```python\nprint(1)\n```')[0]=='print(1)';assert extract_code('```python\nprint(1)\n```\n```python\nprint(2)\n```')[0]=='print(2)'
 assert extract_code('print(1)\nnot prose !!!')[0]=='print(1)'
def test_direct_contract():
 text=(ROOT/'coreflow'/'direct_load.py').read_text(encoding='utf-8');assert 'PeftModel' not in text;assert "source_lora_loaded':False" in text.replace(' ','')
 runner=(ROOT/'scripts'/'run_e1d.py').read_text(encoding='utf-8');assert 'subprocess.run' in runner and 'independent_process_per_path' in runner
def test_canonical():assert digest(ROOT/'vendor_canonical'/'modeling_llama.py')=='868780e64eabdfd2f563373cbebe44d5e4b022b4b89817688f7a77eb107b0681'
def test_python_syntax():
 for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
def main():
 tests=[test_protocol,test_embedded,test_splits,test_no_formal_runner,test_postprocessor,test_direct_contract,test_canonical,test_python_syntax]
 for t in tests:t()
 print(json.dumps({'status':'PASS','tests':[t.__name__ for t in tests]}))
if __name__=='__main__':main()
