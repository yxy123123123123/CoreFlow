#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, core_bank, gate_path, isvd_bank, now_utc, paths, sha256_file, write_json
cfg,p=config(),paths(); target=p["reports"]/"PROTOCOL_SEAL.json"
if target.is_file(): print(json.dumps({"status":"ALREADY_SEALED","seal":str(target)})); raise SystemExit(0)
if json.loads((p["reports"]/"preflight.json").read_text())["status"]!="PASS" or json.loads((p["reports"]/"prepare.json").read_text())["status"]!="PASS": raise RuntimeError("Run preflight and prepare first")
for stage in ("r1_replay","r2_systems","r3_profiler"):
    if (p["results"]/stage).exists() and any((p["results"]/stage).rglob("*")): raise RuntimeError(f"Cannot seal after runtime output: {stage}")
assets={"core_bank":sha256_file(core_bank(cfg,p)/"core_bank.safetensors"),"isvd_bank":sha256_file(isvd_bank(cfg,p)/"isvd_bank.safetensors"),"gates":{str(s):sha256_file(gate_path(cfg,p,s)) for s in cfg["r1_replay"]["seeds"]}}
inputs={x.name:sha256_file(x) for x in sorted(p["data"].glob("*")) if x.is_file()}
seal={"status":"SEALED_BEFORE_ORIGINAL_BASELINES_GPU_OUTPUTS","timestamp_utc":now_utc(),"protocol_id":cfg["protocol_id"],"protocol_sha256":sha256_file(ROOT/"config"/"original_baselines_protocol.json"),"package_manifest_sha256":sha256_file(ROOT/"CHECKSUMS.sha256"),"assets":assets,"inputs":inputs,"claims":{"historical_phase2a_2.02x_not_a_replication_target":True,"new_primary_metric":"decode_tokens_per_second","compact_isvd_forbidden":True}}
write_json(target,seal); print(json.dumps({"status":seal["status"],"seal":str(target)}))
