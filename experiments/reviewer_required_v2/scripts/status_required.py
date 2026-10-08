from __future__ import annotations
import json, os
from pathlib import Path
from common import CFG, WORK, REPORT, active_groups, group_root
def state(p):
    if not p.exists(): return "MISSING"
    try: return json.loads(p.read_text(encoding="utf-8")).get("status","PRESENT")
    except Exception: return "PRESENT"
def main():
    print("=== CoreFlow reviewer required-v2 ===")
    for name in ("preflight.json","prepare.json","PROTOCOL_SEAL.json","compile_interface_tests.json","storage_audit.json","lifecycle_audit.json","theory_error_analysis.json","statistics_required.json","q224_complete.json","drift_diagnostics.json","analysis_required.json"):
        print(f"{name}: {state(REPORT/name)}")
    for g in active_groups():
        root=group_root(g); print(f"group{g}: independent={state(root/'independent_loras'/'manifest.json')} comol={state(root/'comol_manifest.json')} gate={state(root/'gate'/'gate_calibration.json')} bank={state(root/'bank_manifest.json')}")
        for rel in ("quality/independent_full/metrics.json","quality/core_q185/metrics.json","quality/comol_metrics.json"):
            print(f"  {rel}: {state(root/rel)}")
    service=WORK/"results"/CFG["protocol_id"]/"service"; print(f"service metric files: {len(list(service.rglob('*.json'))) if service.exists() else 0}")
    print("=== recent logs ===")
    logs=WORK/"logs"
    if logs.exists():
        for p in sorted(logs.rglob("*.log"),key=lambda x:x.stat().st_mtime)[-10:]: print(p)
if __name__=="__main__": main()
