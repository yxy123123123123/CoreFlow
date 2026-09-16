#!/usr/bin/env python3
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths
cfg,p=config(),paths(); print("=== reports ===")
for name in ["preflight.json","prepare.json","r0_asset_audit.json","PROTOCOL_SEAL.json","r1_identity_audit.json","r2_complete.json","r3_complete.json","final/original_baselines_summary.json"]:
    path=p["reports"]/name; print(f"{name}: {'PRESENT' if path.is_file() else 'MISSING'}")
print("\n=== R1 replay ===")
for seed in cfg["r1_replay"]["seeds"]:
    path=p["results"]/"r1_replay"/f"full_legacy_original_seed{seed}"/"summary.json"
    if path.is_file():
        x=json.loads(path.read_text()); print(f"seed{seed}: COMPLETE {x['correct']}/{x['rows']}")
    elif Path(str(path.parent)+".partial").exists(): print(f"seed{seed}: PARTIAL/RUNNING")
    else: print(f"seed{seed}: PENDING")
print("\n=== R2 systems ===")
for item in [cfg["r2_systems"]["anchor"]]+cfg["r2_systems"]["boundaries"]:
    base=p["results"]/"r2_systems"/item["name"]; complete=len(list(base.glob("block*/*/COMPLETE.json"))) if base.exists() else 0; partial=len(list(base.glob("block*/*.partial"))) if base.exists() else 0; oom=len(list(base.glob("block*/*.oom/OOM.json"))) if base.exists() else 0; print(f"{item['name']}: complete={complete}/{item['blocks']*3}, partial={partial}, recorded_oom={oom}")
print("\n=== R3 profiler ===")
base=p["results"]/"r3_profiler"
for method in cfg["methods"]: print(f"{method}: {'COMPLETE' if (base/f'{method}.json').is_file() else 'PENDING'}")
