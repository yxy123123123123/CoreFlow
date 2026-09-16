#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths, write_json

def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
cfg,p=config(),paths(); out=[]
for seed in cfg["r1_replay"]["seeds"]:
    current=p["results"]/"r1_replay"/f"full_legacy_original_seed{seed}"
    if not (current/"COMPLETE.json").is_file(): raise RuntimeError(f"Missing R1 seed {seed}")
    old_g=rows(ROOT/"evidence"/f"phase2b_full_seed{seed}_generations.jsonl"); new_g=rows(current/"generations.jsonl")
    old_e=rows(ROOT/"evidence"/f"phase2b_full_seed{seed}_executor_results.jsonl"); new_e=rows(current/"executor_results.jsonl")
    if [x["task_id"] for x in old_g] != [x["task_id"] for x in new_g]: raise ValueError("Task ordering mismatch")
    if [x["prompt_sha256"] for x in old_g] != [x["prompt_sha256"] for x in new_g]: raise ValueError("Prompt hash mismatch")
    old_pass={x["task_id"]:bool(x["evaluation"]["passed"]) for x in old_e}; new_pass={x["task_id"]:bool(x["evaluation"]["passed"]) for x in new_e}
    out.append({"seed":seed,"rows":len(new_g),"original_full_correct":sum(new_pass.values()),"phase2b_full_vectorized_correct":sum(old_pass.values()),"both_pass":sum(old_pass[k] and new_pass[k] for k in old_pass),"original_only":sum(new_pass[k] and not old_pass[k] for k in old_pass),"vectorized_only":sum(old_pass[k] and not new_pass[k] for k in old_pass),"both_fail":sum(not old_pass[k] and not new_pass[k] for k in old_pass),"raw_outputs_identical":sum(a["raw_output"]==b["raw_output"] for a,b in zip(old_g,new_g))})
write_json(p["reports"]/"r1_identity_audit.json",{"status":"ANALYZED_IMPLEMENTATION_SENSITIVITY","role":"development baseline identity audit; no q selection","seeds":out})
print(json.dumps({"status":"ANALYZED_IMPLEMENTATION_SENSITIVITY","seeds":len(out)}))
