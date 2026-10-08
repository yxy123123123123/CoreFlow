from __future__ import annotations
import json, os
from pathlib import Path
from statistics import mean
from common import CFG, REPORT, WORK, active_groups, group_root, write_json

def load(p):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception: return None
def main():
    result=WORK/"results"/CFG["protocol_id"]
    group_summaries=[]
    for g in active_groups():
        root=group_root(g); quality=[]
        for label,rel in (("independent_full","quality/independent_full/metrics.json"),("core_q185","quality/core_q185/metrics.json"),("comol_native","quality/comol_metrics.json")):
            v=load(root/rel)
            if v is not None: quality.append({"method":label,"metrics":v})
        group_summaries.append({"group":g,"independent_manifest":load(root/"independent_loras"/"manifest.json"),"comol_manifest":load(root/"comol_manifest.json"),"gate":load(root/"gate"/"gate_calibration.json"),"bank":load(root/"bank_manifest.json"),"quality":quality})
    service=[]
    for p in sorted((result/"service").glob("*/*/*.json")) if (result/"service").exists() else []:
        v=load(p)
        if v is not None and p.name!="COMPLETE.json": service.append({"path":str(p),**v})
    audit={}
    for name in ("compile_interface_tests.json","storage_audit.json","lifecycle_audit.json","theory_error_analysis.json","statistics_required.json"):
        audit[name]=load(REPORT/name)
    payload={"status":"ANALYZED","protocol_id":CFG["protocol_id"],"required_group_summaries":group_summaries,"q224_report":load(REPORT/"q224_complete.json"),"drift":load(REPORT/"drift_diagnostics.json"),"audits":audit,"service":service,"claim_boundaries":CFG["claims"]}
    write_json(REPORT/"analysis_required.json",payload)
    lines=["# CoreFlow reviewer required-v2 analysis","",f"- active repeat groups: {', '.join(active_groups())}",f"- service records: {len(service)}",f"- q224 marker: {'present' if payload['q224_report'] else 'pending'}",f"- drift report: {'present' if payload['drift'] else 'pending'}","","The group2/group3 quality comparison uses the complete six-dataset protocol. q224 remains a sensitivity point; service c4 is queued-load, not continuous batching.","","## Required group quality outputs","","| Group | Independent Full | Core q185 | CoMoL |","|---|---|---|---|"]
    for x in group_summaries:
        names={v["method"] for v in x["quality"]}; lines.append(f"| {x['group']} | {'PASS' if 'independent_full' in names else 'PENDING'} | {'PASS' if 'core_q185' in names else 'PENDING'} | {'PASS' if 'comol_native' in names else 'PENDING'} |")
    (REPORT/"analysis_required.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ANALYZED","groups":len(group_summaries),"service":len(service)},ensure_ascii=False))
if __name__=="__main__": main()
