from __future__ import annotations
import json, os
from pathlib import Path
from common import CFG, REPORT, write_json

def ep(name,default): return Path(os.environ.get(name,default)).expanduser().resolve()
def main():
    formal=ep("FORMAL_V2_WORK_ROOT","/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    records=[]
    for q in (185,224):
        cfgp=formal/"m2_assets"/"core_banks"/"k5_code"/f"q{q}_bf16"/"core_config.json"
        if not cfgp.exists(): continue
        cfg=json.loads(cfgp.read_text(encoding="utf-8")); errors=[]
        for mod in cfg.get("modules",[]):
            errs=mod.get("fp32_sampled_output_relative_errors") or []
            if errs:
                vals=[float(x) for x in errs if isinstance(x,(int,float))]
                errors.append({"name":mod.get("name"),"sample_count":len(vals),"max_sampled_relative_error":max(vals),"mean_sampled_relative_error":sum(vals)/len(vals)})
        records.append({"label":f"llama_core_q{q}","q":q,"module_count":len(cfg.get("modules",[])),"efficiency":cfg.get("efficiency"),"sampled_module_errors":errors})
    payload={"status":"ANALYZED","protocol_id":CFG["protocol_id"],"bound":"||Delta y|| <= sum_k |g_k| ||Delta W_k|| ||x|| + sum_k |Delta g_k| ||W_k x||","interpretation":"empirical module residuals and gate perturbations are reported as an explanatory decomposition, not a universal theorem","banks":records}
    write_json(REPORT/"theory_error_analysis.json",payload); print(json.dumps({"status":payload["status"],"banks":len(records)},ensure_ascii=False))
if __name__=="__main__": main()
