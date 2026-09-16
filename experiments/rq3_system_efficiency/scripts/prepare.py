#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths, sha256_file, write_json

cfg,p=config(),paths(); p["data"].mkdir(parents=True,exist_ok=True); p["reports"].mkdir(parents=True,exist_ok=True); p["results"].mkdir(parents=True,exist_ok=True)
if (p["reports"] / "PROTOCOL_SEAL.json").is_file():
    if not (p["reports"] / "prepare.json").is_file(): raise RuntimeError("Seal exists but prepare marker is missing")
    print(json.dumps({"status":"SKIP_ALREADY_SEALED","prepare":str(p["reports"] / "prepare.json")})); raise SystemExit(0)
source=ROOT/"data"/cfg["data"]["dev128_file"]; prompt_file=ROOT/"data"/cfg["data"]["prompt_file"]
rows=[json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
if len(rows)!=128 or sha256_file(source)!=cfg["data"]["dev128_sha256"]: raise ValueError("Frozen dev128 contract failed")
if sha256_file(prompt_file)!=cfg["data"]["prompt_sha256"]: raise ValueError("Prompt hash mismatch")
(p["data"]/source.name).write_bytes(source.read_bytes()); (p["data"]/prompt_file.name).write_bytes(prompt_file.read_bytes())
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(str(p["model"]),local_files_only=True)
template=prompt_file.read_text(encoding="utf-8"); filler=tokenizer("\n# CoreFlow deterministic systems benchmark context. This comment is intentionally inert.",add_special_tokens=False)["input_ids"]
if not filler: raise ValueError("Tokenizer produced empty filler")
needed={cfg["r2_systems"]["anchor"]["input_length"]}|{x["input_length"] for x in cfg["r2_systems"]["boundaries"]}
files={}
for length in sorted(needed):
    frozen=[]
    for row in rows[:cfg["r2_systems"]["input_source_rows"]]:
        prompt=template.format(problem=row["problem"],entry_point=row["entry_point"])
        ids=tokenizer(prompt,add_special_tokens=True,truncation=True,max_length=length)["input_ids"]
        while len(ids)<length: ids.extend(filler[:length-len(ids)])
        ids=ids[:length]
        if len(ids)!=length: raise AssertionError("Exact input construction failed")
        frozen.append({"task_id":row["task_id"],"source_task_id":row["source_task_id"],"input_length":length,"input_ids":ids,"attention_mask":[1]*length})
    path=p["data"]/f"effective_inputs_l{length}.json"; write_json(path,frozen); files[path.name]={"rows":len(frozen),"sha256":sha256_file(path),"all_attention_tokens_active":all(sum(x["attention_mask"])==length for x in frozen)}
write_json(p["reports"]/"prepare.json",{"status":"PASS","protocol_id":cfg["protocol_id"],"data":files})
print(json.dumps({"status":"PASS","effective_inputs":files}))
