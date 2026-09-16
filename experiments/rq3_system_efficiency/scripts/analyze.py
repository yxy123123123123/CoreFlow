#!/usr/bin/env python3
from __future__ import annotations
import json, math, random, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths, write_json

cfg,p=config(),paths(); r2=p["results"]/"r2_systems"

def load_lines(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def quantile(values,q):
    x=sorted(values); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1); return x[lo]*(hi-pos)+x[hi]*(pos-lo)
def collect(conf):
    blocks={}; ooms=[]
    for block in sorted((r2/conf).glob("block*")):
        methods={}; block_ooms=[]
        for run in block.iterdir():
            if run.is_dir() and (run/"COMPLETE.json").is_file():
                summary=json.loads((run/"summary.json").read_text()); methods[summary["method"]]={"rows":load_lines(run/"measurements.jsonl"),"summary":summary}
        for oom in block.glob("*.oom"):
            marker=oom/"OOM.json"
            if marker.is_file(): block_ooms.append(json.loads(marker.read_text()))
        ooms.extend(block_ooms)
        if set(methods)==set(cfg["methods"]): blocks[block.name]=methods
        elif not block_ooms: raise RuntimeError(f"Incomplete block without recorded OOM: {block}")
    return blocks,ooms
def comparison(blocks,candidate,reference):
    block_logs=[]
    for b,methods in blocks.items():
        c=[x["timing"]["decode_tokens_per_second"] for x in methods[candidate]["rows"]]; r=[x["timing"]["decode_tokens_per_second"] for x in methods[reference]["rows"]]
        if len(c)!=len(r): raise ValueError("Within-block request count mismatch")
        block_logs.append([math.log(a/b) for a,b in zip(c,r)])
    point=math.exp(statistics.fmean(statistics.fmean(x) for x in block_logs)); rng=random.Random(cfg["statistics"]["bootstrap_seed"]); samples=[]
    for _ in range(cfg["statistics"]["bootstrap_samples"]):
        chosen=[block_logs[rng.randrange(len(block_logs))] for _ in block_logs]; nested=[]
        for values in chosen: nested.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
        samples.append(math.exp(statistics.fmean(nested)))
    return {"candidate":candidate,"reference":reference,"blocks":len(block_logs),"paired_geometric_speedup":point,"ci95":[quantile(samples,.025),quantile(samples,.975)],"decision":"LOWER_CI_GT_1" if quantile(samples,.025)>1 else "NOT_CONFIRMED"}
def memory(blocks,candidate,reference):
    keys=["peak_memory_allocated_bytes","peak_memory_reserved_bytes","startup_resident_allocated_bytes","startup_resident_reserved_bytes","incremental_peak_allocated_bytes","incremental_peak_reserved_bytes"]
    out={}
    for key in keys:
        diffs=[]
        for methods in blocks.values(): diffs.append((methods[candidate]["summary"]["statistics"][key]-methods[reference]["summary"]["statistics"][key])/(1024**3))
        out[key+"_difference_gib"]={"median":statistics.median(diffs),"iqr":[quantile(diffs,.25),quantile(diffs,.75)],"min":min(diffs),"max":max(diffs)}
    return out

configs=[cfg["r2_systems"]["anchor"]]+cfg["r2_systems"]["boundaries"]; systems={}
for item in configs:
    blocks,ooms=collect(item["name"])
    if ooms:
        systems[item["name"]]={"role":"descriptive_capacity_boundary","speed_comparison":"NOT_ESTIMATED_DUE_TO_RECORDED_OOM","recorded_oom":ooms,"complete_paired_blocks":len(blocks)}
        continue
    h1=comparison(blocks,"coreflow_q185_frozen","full_legacy_original"); h2=comparison(blocks,"coreflow_q185_frozen","isvd_legacy_padded_matched_q185")
    methods={}
    for m in cfg["methods"]:
        vals=[statistics.median([r["timing"]["decode_tokens_per_second"] for r in methods_[m]["rows"]]) for methods_ in blocks.values()]
        methods[m]={"launches":len(vals),"launch_median_decode_tokens_per_second":{"median":statistics.median(vals),"iqr":[quantile(vals,.25),quantile(vals,.75)],"min":min(vals),"max":max(vals)}}
    systems[item["name"]]={"role":"primary" if item is configs[0] else "descriptive_boundary","methods":methods,"core_vs_full":h1,"core_vs_isvd":h2,"memory_core_minus_full":memory(blocks,"coreflow_q185_frozen","full_legacy_original"),"memory_core_minus_isvd":memory(blocks,"coreflow_q185_frozen","isvd_legacy_padded_matched_q185")}
anchor=systems[cfg["r2_systems"]["anchor"]["name"]]; h1_pass=anchor["core_vs_full"]["ci95"][0]>1; h2_status="PASS" if h1_pass and anchor["core_vs_isvd"]["ci95"][0]>1 else "FAIL" if h1_pass else "NOT_TESTED_HIERARCHY"
r1=json.loads((p["reports"]/"r1_identity_audit.json").read_text()); prof={x.stem:json.loads(x.read_text()) for x in sorted((p["results"]/"r3_profiler").glob("*.json"))}
deploy={}
blocks,_=collect(cfg["r2_systems"]["anchor"]["name"])
for m in cfg["methods"]:
    summaries=[x[m]["summary"] for x in blocks.values()]; deploy[m]={"fresh_process_launches":len(summaries),"client_startup_to_health_seconds_median":statistics.median(x["client_startup_to_health_seconds"] for x in summaries),"model_load_seconds_median":statistics.median(x["server_health"]["model_load_seconds"] for x in summaries),"runtime_install_seconds_median":statistics.median(x["server_health"]["runtime_install_seconds"] for x in summaries),"startup_allocated_gib_median":statistics.median(x["server_health"]["startup_memory_allocated_bytes"]/(1024**3) for x in summaries)}
summary={"status":"ANALYZED","protocol_id":cfg["protocol_id"],"r1_original_full_replay":r1,"r2_systems":systems,"r3_light_profiler":prof,"r4_deployment_from_r2_fresh_processes":deploy,"hierarchical_decision":{"H1_core_vs_original_full":"PASS" if h1_pass else "FAIL_OR_INCONCLUSIVE","H2_core_vs_original_isvd":h2_status},"claim_limits":["The historical Phase2A 2.02x total-token result is not treated as the replication target.","New system evidence uses fixed-length decode tokens/s.","Phase2A Full-vectorized remains implementation-sensitivity evidence, not a method in this formal three-way matrix.","LCB and compact ISVD are excluded.","R1 uses dev128 and is not new confirmatory quality evidence."]}
final=p["reports"]/"final"; final.mkdir(parents=True,exist_ok=True); write_json(final/"original_baselines_summary.json",summary); print(json.dumps({"status":"ANALYZED","H1":summary["hierarchical_decision"]["H1_core_vs_original_full"],"H2":h2_status}))
