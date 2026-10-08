from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from common import CFG, COMOL_ROOT, REPORT, RESULT, WORK, active_groups, group_root, manifest_path, write_json

def find_adapter(root:Path):
    hits=[]
    for p in root.rglob("adapter_config.json"):
        if (p.parent/"adapter_model.bin").is_file() or (p.parent/"adapter_model.safetensors").is_file(): hits.append(p.parent)
    if not hits: return None
    return max(hits,key=lambda p:p.stat().st_mtime)

def normalize_standard_peft_config(adapter: Path):
    """Expose CoMoL's lora_rank under the standard PEFT r field.

    CoMoL serializes ordinary LoRA checkpoints with ``lora_rank`` while the
    upstream PEFT loader expects ``r``.  Keep the original field and add the
    equivalent compatibility field; weights are not changed.
    """
    path = adapter / "adapter_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    rank = int(cfg.get("lora_rank", cfg.get("r", 0)))
    if rank <= 0:
        raise ValueError(f"Missing positive LoRA rank in {path}")
    if int(cfg.get("r", rank)) != rank:
        raise ValueError(f"Conflicting r/lora_rank in {path}: {cfg.get('r')} vs {rank}")
    if "r" not in cfg:
        backup = adapter / "adapter_config.comol_original.json"
        if not backup.exists():
            shutil.copy2(path, backup)
        cfg["r"] = rank
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(cfg, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

def train_one(group, idx, seed):
    out_root=group_root(group)/"independent_outputs"/f"expert_{idx:02d}"
    existing=find_adapter(out_root)
    if existing:
        normalize_standard_peft_config(existing)
        return existing
    out_root.mkdir(parents=True,exist_ok=True)
    t=time.time(); tr=CFG["training"]
    cmd=[os.environ.get("COMOL_PYTHON","/root/autodl-tmp/comol_official_env/bin/python"),str(COMOL_ROOT/"train.py"),f"--model_path={os.environ.get('MODEL_PATH','/root/autodl-tmp/coreflow_qwen3_load_smoke_v1_upload/assets/base_model')}",f"--data_path={str(COMOL_ROOT/'datasets'/'math_14k')}",f"--output_dir={out_root}","--peft_type=lora",f"--lora_rank={tr['rank']}","--target_modules",*tr["target_modules"],"--num_experts=1",f"--max_length={tr['max_length']}",f"--batch_size={tr['batch_size']}",f"--gradient_accumulation_steps={tr['gradient_accumulation_steps']}",f"--num_train_epochs={tr['epochs']}",f"--learning_rate={tr['learning_rate']}",f"--lr_scheduler_type={tr['scheduler']}",f"--warmup_steps={tr['warmup_steps']}",f"--weight_decay={tr['weight_decay']}",f"--seed={seed}"]
    log=out_root/"train.log"
    with log.open("w",encoding="utf-8") as h:
        result=subprocess.run(cmd,cwd=COMOL_ROOT,stdout=h,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Independent LoRA failed group={group} expert={idx} seed={seed}; see {log}")
    adapter=find_adapter(out_root)
    if adapter is None: raise RuntimeError(f"No adapter output under {out_root}")
    normalize_standard_peft_config(adapter)
    return adapter

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); args=ap.parse_args(); group=args.group
    if group not in active_groups(): raise ValueError(f"Group not active: {group}")
    seeds=CFG["expert_groups"][group]["independent_seeds"]; records=[]; started=time.time()
    for i,seed in enumerate(seeds,1):
        print(f"[INDEPENDENT] group={group} expert={i}/8 seed={seed}",flush=True); path=train_one(group,i,seed); records.append({"expert":i,"seed":seed,"path":str(path)})
    write_json(manifest_path(group),{"status":"PASS","group":group,"records":records,"elapsed_seconds":time.time()-started,"training":CFG["training"]})
    print(json.dumps({"status":"PASS","group":group,"experts":len(records),"elapsed_seconds":time.time()-started}))
if __name__=="__main__": main()
