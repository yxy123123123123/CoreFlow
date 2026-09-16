#!/usr/bin/env python3
import itertools, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths, write_json

cfg,p=config(),paths(); methods=list(cfg["methods"]); perms=list(itertools.permutations(methods)); result=p["results"]/"r2_systems"; result.mkdir(parents=True,exist_ok=True)
configs=[cfg["r2_systems"]["anchor"]]+cfg["r2_systems"]["boundaries"]
completed=[]; ooms=[]
for ci,item in enumerate(configs):
    orders=perms*2 if item["blocks"]==12 else perms
    if len(orders)!=item["blocks"]: raise ValueError("Block/permutation contract failed")
    for bi,order in enumerate(orders,1):
        block=f"block{bi:02d}"
        for pos,method in enumerate(order,1):
            output=result/item["name"]/block/f"{pos}_{method}"
            cmd=[str(p["runtime_python"]),str(ROOT/"scripts"/"r2_worker.py"),"--method",method,"--config-name",item["name"],"--block-id",block,"--order-position",str(pos),"--batch-size",str(item["batch_size"]),"--input-length",str(item["input_length"]),"--output-length",str(item["output_length"]),"--warmup",str(item["warmup"]),"--measurements",str(item["measurements"]),"--port",str(6800+ci*100+bi*3+pos),"--output",str(output)]
            env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=p["cuda_device"]
            print(f"[START] {item['name']} {block} position={pos} method={method}",flush=True)
            done=subprocess.run(cmd,env=env)
            if done.returncode==20 and ci>0:
                ooms.append(str(output)+".oom"); continue
            if done.returncode: raise SystemExit(done.returncode)
            completed.append(str(output))
write_json(p["reports"]/"r2_complete.json",{"status":"PASS_WITH_RECORDED_BOUNDARY_OOM" if ooms else "PASS","runs":len(completed),"boundary_oom":ooms,"configs":[x["name"] for x in configs],"execution":"single_gpu_exclusive_serial"})
print(f"R2 PASS: {len(completed)} complete launches; boundary OOM={len(ooms)}")
