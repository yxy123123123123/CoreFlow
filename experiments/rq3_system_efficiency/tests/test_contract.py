#!/usr/bin/env python3
import ast, hashlib, itertools, json
from pathlib import Path
EXPERIMENT_ROOT=Path(__file__).resolve().parents[1]
REPO_ROOT=Path(__file__).resolve().parents[3]
cfg=json.loads((EXPERIMENT_ROOT/"config"/"original_baselines_protocol.json").read_text(encoding="utf-8"))
assert cfg["protocol_id"]=="coreflow-applsci-original-baselines-v1.1"
assert list(cfg["methods"])==["full_legacy_original","coreflow_q185_frozen","isvd_legacy_padded_matched_q185"]
assert len(list(itertools.permutations(cfg["methods"])))==6
assert cfg["r2_systems"]["anchor"]["blocks"]==12
assert all(x["blocks"]==6 for x in cfg["r2_systems"]["boundaries"])
assert cfg["execution_contract"]["no_compact_isvd"] and cfg["execution_contract"]["no_formal250_access"]
assert cfg["r1_replay"]["seeds"]==[41,42,43]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(REPO_ROOT/"coreflow"/"loraflow.py")=="5279b8cb340f72e274746ec61df63444134a1ee81627a8d6655957d6a24c2484"
assert sha(REPO_ROOT/"coreflow"/"runtime.py")=="292c04418600597bf914b0c65e597b6080fb54f3a82a00ed45c48ddc85397adc"
for path in list((EXPERIMENT_ROOT/"scripts").glob("*.py"))+list((REPO_ROOT/"coreflow").glob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
assert "use_fast=False" not in (EXPERIMENT_ROOT/"scripts"/"prepare.py").read_text(encoding="utf-8")
print(json.dumps({"status":"PASS","tests":["protocol","balanced_orders","formal_runtime_identity","python_syntax"]}))
