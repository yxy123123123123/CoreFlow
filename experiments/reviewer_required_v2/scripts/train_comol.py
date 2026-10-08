from __future__ import annotations
import argparse, os, subprocess, time, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from common import CFG, COMOL_ROOT, WORK, active_groups, group_root, write_json
from train_independent import find_adapter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); args=ap.parse_args(); group=args.group
    if group not in active_groups(): raise ValueError(f"Group not active: {group}")
    seed=CFG["expert_groups"][group]["comol_seed"]; out_root=group_root(group)/"comol_outputs"; out_root.mkdir(parents=True,exist_ok=True)
    existing=find_adapter(out_root)
    if existing:
        write_json(group_root(group)/"comol_manifest.json",{"status":"PASS","group":group,"seed":seed,"path":str(existing),"reused":True}); print(f"[COMOL] reuse {existing}"); return
    tr=CFG["training"]; cmd=[os.environ.get("COMOL_PYTHON","/root/autodl-tmp/comol_official_env/bin/python"),str(COMOL_ROOT/"train.py"),f"--model_path={os.environ.get('MODEL_PATH','/root/autodl-tmp/coreflow_qwen3_load_smoke_v1_upload/assets/base_model')}",f"--data_path={str(COMOL_ROOT/'datasets'/'math_14k')}",f"--output_dir={out_root}","--peft_type=mocorelora",f"--lora_rank={tr['rank']}","--target_modules",*tr["target_modules"],"--num_experts=8",f"--max_length={tr['max_length']}",f"--batch_size={tr['batch_size']}",f"--gradient_accumulation_steps={tr['gradient_accumulation_steps']}",f"--num_train_epochs={tr['epochs']}",f"--learning_rate={tr['learning_rate']}",f"--lr_scheduler_type={tr['scheduler']}",f"--warmup_steps={tr['warmup_steps']}",f"--weight_decay={tr['weight_decay']}","--core_router=True",f"--seed={seed}"]
    log=out_root/"train.log"; started=time.time()
    with log.open("w",encoding="utf-8") as h: result=subprocess.run(cmd,cwd=COMOL_ROOT,stdout=h,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"CoMoL training failed group={group}; see {log}")
    path=find_adapter(out_root)
    if path is None: raise RuntimeError(f"No CoMoL output under {out_root}")
    write_json(group_root(group)/"comol_manifest.json",{"status":"PASS","group":group,"seed":seed,"path":str(path),"elapsed_seconds":time.time()-started,"training":CFG["training"]})
    print(f"[COMOL] PASS group={group} path={path}")
if __name__=="__main__": main()
