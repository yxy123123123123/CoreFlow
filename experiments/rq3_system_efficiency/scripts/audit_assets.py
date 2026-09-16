#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, core_bank, isvd_bank, paths, sha256_file, write_json
cfg,p=config(),paths(); existing=p["reports"] / "r0_asset_audit.json"
if (p["reports"] / "PROTOCOL_SEAL.json").is_file() and existing.is_file():
    print(json.dumps({"status":"SKIP_ALREADY_SEALED","report":str(existing)})); raise SystemExit(0)
experts={}; full_bytes=0
for name,item in cfg["assets"]["experts"].items():
    root=p["asset_root"]/"LoRAs"/item["relative"]; ckpt=next(x for x in [root/"adapter_model.safetensors",root/"adapter_model.bin"] if x.is_file()); size=ckpt.stat().st_size; full_bytes+=size; experts[name]={"checkpoint":str(ckpt),"sha256":sha256_file(ckpt),"bytes":size}
core=core_bank(cfg,p); isvd=isvd_bank(cfg,p); core_cfg=json.loads((core/"core_config.json").read_text()); isvd_cfg=json.loads((isvd/"isvd_config.json").read_text())
payload={"status":"PASS","method_identity":{"full_legacy_original":"unmodified formal-v2 dynamic PEFT LoRA-Flow path","coreflow_q185_frozen":"formal-v1 frozen q185 bank with formal-v2 fused runtime","isvd_legacy_padded_matched_q185":"formal-v2 original padded bank/runtime"},"experts":experts,"artifact_bytes":{"source_lora_checkpoints_total":full_bytes,"core_bank":(core/"core_bank.safetensors").stat().st_size,"isvd_bank":(isvd/"isvd_bank.safetensors").stat().st_size},"core_config_budget":core_cfg.get("budget"),"isvd_config_budget":isvd_cfg.get("budget"),"exclusions":["compact ISVD","Full-vectorized as a formal method","formal250 quality data","q reselection"]}
write_json(p["reports"]/"r0_asset_audit.json",payload); print(json.dumps({"status":"PASS","experts":len(experts)}))
