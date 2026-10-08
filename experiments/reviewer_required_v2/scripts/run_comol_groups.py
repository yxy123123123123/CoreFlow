from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
from common import CFG, active_groups, WORK, REPORT

def run(py, script, args, log):
    log.parent.mkdir(parents=True,exist_ok=True); print("[START] "+" ".join([py,str(script),*args]),flush=True)
    with log.open("w",encoding="utf-8") as h: r=subprocess.run([py,str(script),*args],stdout=h,stderr=subprocess.STDOUT,text=True)
    if r.returncode: raise RuntimeError(f"stage failed exit={r.returncode}; see {log}")

def main():
    if not (REPORT / "PROTOCOL_SEAL.json").exists():
        raise RuntimeError("Run preflight, prepare and seal first")
    ap=argparse.ArgumentParser(); ap.add_argument("--groups",default=",".join(active_groups())); ap.add_argument("--limit",type=int,default=0); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; py=os.environ.get("CORE_PYTHON",os.environ.get("PYTHON",sys.executable)); groups=[x.strip() for x in args.groups.split(",") if x.strip()]
    for group in groups:
        gr=WORK/"groups"/f"group{group}"; logdir=WORK/"logs"/f"group{group}"
        stages=[("train_independent.py",["--group",group]),("train_comol.py",["--group",group]),("calibrate_gate.py",["--group",group]),("compile_coreflow.py",["--group",group]),("quality_full",["--group",group,"--method","full"]),("quality_core_q185",["--group",group,"--method","core","--q","185"]),("quality_comol",["--group",group])]
        for name,args2 in stages:
            if name.startswith("quality_"):
                script="quality_comol.py" if name=="quality_comol" else "quality_core_worker.py"
                if args.limit: args2 += ["--limit",str(args.limit)]
            else: script=name
            if name == "quality_comol":
                marker = gr / "quality" / "comol_metrics.json"
            elif name == "quality_full":
                marker = gr / "quality" / "independent_full" / "metrics.json"
            elif name == "quality_core_q185":
                marker = gr / "quality" / "core_q185" / "metrics.json"
            else:
                marker = gr / "_stage" / f"{name}.done"
            if marker.exists(): print(f"[SKIP] group={group} stage={name}"); continue
            run(py,root/"scripts"/script,args2,logdir/(name.replace(".py","" )+".log"))
            if not name.startswith("quality_"):
                marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text("PASS\n",encoding="utf-8")
    print("[COMOL_GROUPS] complete",flush=True)
if __name__=="__main__": main()
