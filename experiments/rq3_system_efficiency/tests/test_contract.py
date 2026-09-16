#!/usr/bin/env python3
import ast, hashlib, itertools, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/"config"/"original_baselines_protocol.json").read_text(encoding="utf-8"))
assert cfg["protocol_id"]=="coreflow-applsci-original-baselines-v1.1"
assert list(cfg["methods"])==["full_legacy_original","coreflow_q185_frozen","isvd_legacy_padded_matched_q185"]
assert len(list(itertools.permutations(cfg["methods"])))==6
assert cfg["r2_systems"]["anchor"]["blocks"]==12
assert all(x["blocks"]==6 for x in cfg["r2_systems"]["boundaries"])
assert cfg["execution_contract"]["no_compact_isvd"] and cfg["execution_contract"]["no_formal250_access"]
assert cfg["r1_replay"]["seeds"]==[41,42,43]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(ROOT/"coreflow"/"loraflow.py")=="be5092f0588369d10df78822f11b6e73be6953a70247e913c3fbadbee0e771c2"
assert sha(ROOT/"coreflow"/"runtime.py")=="0bd8b614dd81913120c000b1522f883ec95ec40da049f98860aa48ca2fab0c02"
for path in list((ROOT/"scripts").glob("*.py"))+list((ROOT/"coreflow").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
assert "use_fast=False" not in (ROOT/"scripts"/"prepare.py").read_text(encoding="utf-8")
print(json.dumps({"status":"PASS","tests":["protocol","balanced_orders","formal_runtime_identity","python_syntax"]}))
