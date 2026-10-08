from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from common import COMOL_ROOT, CFG, group_root, load_json, write_json
from train_independent import find_adapter
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--group",required=True); args=ap.parse_args(); group=args.group
    root=group_root(group)/"comol_outputs"; model=find_adapter(root)
    if model is None: raise RuntimeError(f"CoMoL checkpoint not found under {root}")
    # The official evaluator selects the benchmark family from the basename of
    # model_path. Trainer checkpoints are named checkpoint-N, so expose the
    # same checkpoint through a deterministic alias containing "math". This
    # changes neither model weights nor the official prompt/scoring code.
    eval_model=model
    if "math" not in model.name.lower():
        alias=model.parent/f"math-{model.name}"
        if alias.is_symlink():
            if alias.resolve()!=model.resolve(): raise RuntimeError(f"Unexpected evaluation alias target: {alias}")
        elif alias.exists():
            raise RuntimeError(f"Evaluation alias exists and is not a symlink: {alias}")
        else:
            alias.symlink_to(model.name,target_is_directory=True)
        eval_model=alias
    predictions=model/"predictions"; predictions.mkdir(parents=True,exist_ok=True)
    py=os.environ.get("COMOL_PYTHON","/root/autodl-tmp/comol_official_env/bin/python"); log=root/"evaluation.log"; data=COMOL_ROOT/"datasets"/"math_commonsense"; cmd=[py,str(COMOL_ROOT/"test_math.py"),"--model_path",str(eval_model),"--data_path",str(data),"--max_new_tokens",str(CFG["quality"]["max_new_tokens"]),"--batch_size",str(CFG["quality"]["batch_size"])]
    with log.open("w",encoding="utf-8") as h:
        r=subprocess.run(cmd,cwd=COMOL_ROOT,stdout=h,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f"CoMoL evaluation failed; see {log}")
    prediction=predictions/"addsub_responses.jsonl"
    if not prediction.is_file(): raise RuntimeError(f"CoMoL prediction file missing: {prediction}")
    score=model/"acc_score.jsonl"; score.unlink(missing_ok=True)
    for stale in predictions.glob("*_predict_checkanswer.jsonl"): stale.unlink()
    score_log=root/"scoring.log"; score_cmd=[py,str(COMOL_ROOT/"evaluate_math.py"),"--predict_file",str(prediction)]
    with score_log.open("w",encoding="utf-8") as h:
        r=subprocess.run(score_cmd,cwd=COMOL_ROOT,stdout=h,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f"CoMoL scoring failed; see {score_log}")
    if not score.is_file(): raise RuntimeError(f"CoMoL score file missing after scoring: {score}")
    scores=[line for line in score.read_text(encoding="utf-8").splitlines() if line.strip()]
    write_json(group_root(group)/"quality"/"comol_metrics.json",{"status":"PASS","group":group,"checkpoint":str(model),"evaluation_alias":str(eval_model),"official_score_records":scores,"generation_log":str(log),"scoring_log":str(score_log)})
    print(f"[QUALITY] CoMoL PASS group={group} checkpoint={model}")
if __name__=="__main__": main()
