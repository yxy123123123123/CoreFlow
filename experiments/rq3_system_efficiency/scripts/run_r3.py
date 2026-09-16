#!/usr/bin/env python3
import os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, paths, write_json
cfg,p=config(),paths(); root=p["results"]/"r3_profiler"; root.mkdir(parents=True,exist_ok=True); completed=[]
for method in cfg["methods"]:
    output=root/f"{method}.json"; env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=p["cuda_device"]
    print(f"[R3] method={method}",flush=True); done=subprocess.run([str(p["runtime_python"]),str(ROOT/"scripts"/"profile_worker.py"),"--method",method,"--output",str(output)],env=env)
    if done.returncode: raise SystemExit(done.returncode)
    completed.append(output.name)
write_json(p["reports"]/"r3_complete.json",{"status":"PASS","methods":completed,"lightweight":True,"chrome_trace":False}); print("R3 PASS")
