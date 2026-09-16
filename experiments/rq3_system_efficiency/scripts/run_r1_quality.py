#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coreflow.m2_eval import evaluate_mbppplus_program
from scripts.common import config, gate_path, paths, request_json, sha256_file, wait_health, write_json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args(); cfg, p = config(), paths()
    if not (p["reports"] / "PROTOCOL_SEAL.json").is_file(): raise RuntimeError("Run seal first")
    final = p["results"] / "r1_replay" / f"full_legacy_original_seed{args.seed}"; partial = Path(str(final) + ".partial")
    if (final / "COMPLETE.json").is_file(): print(f"[SKIP] {final}"); return
    if final.exists() or partial.exists(): raise RuntimeError(f"Retained partial/final output exists: {final}")
    partial.mkdir(parents=True)
    command = [str(p["runtime_python"]), str(ROOT/"scripts"/"serve_method.py"), "--model", str(p["model"]), "--asset-root", str(p["asset_root"]), "--official-root", str(p["official_root"]), "--adapter-order", ",".join(cfg["expert_order"]), "--gate", str(gate_path(cfg,p,args.seed)), "--method", "full_legacy_original", "--port", str(args.port), "--max-input-tokens", str(cfg["r1_replay"]["max_input_tokens"]), "--output-length", str(cfg["r1_replay"]["max_new_tokens"])]
    write_json(partial/"resolved_run.json", {"command":command,"seed":args.seed,"protocol_id":cfg["protocol_id"]})
    log=(partial/"server.log").open("w",encoding="utf-8",newline="\n"); proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,text=True)
    rows=[json.loads(x) for x in (p["data"]/cfg["data"]["dev128_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    template=(p["data"]/cfg["data"]["prompt_file"]).read_text(encoding="utf-8"); evaluated=[]; started=time.time()
    try:
        health=wait_health(f"http://127.0.0.1:{args.port}/health",proc); write_json(partial/"health.json",health)
        for i in range(cfg["r1_replay"]["warmup_requests"]): request_json(f"http://127.0.0.1:{args.port}/infer",{"prompt":"warmup","seed":990000+i})
        with (partial/"generations.jsonl").open("w",encoding="utf-8",newline="\n") as gf,(partial/"executor_results.jsonl").open("w",encoding="utf-8",newline="\n") as ef:
            for i,row in enumerate(rows):
                prompt=template.format(problem=row["problem"],entry_point=row["entry_point"]); response=request_json(f"http://127.0.0.1:{args.port}/infer",{"prompt":prompt,"seed":args.seed*100000+i})
                raw=response["answer"]; gen={"task_id":row["task_id"],"order_index":i,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"raw_output":raw,"timing":response["timing"]}; gf.write(json.dumps(gen,ensure_ascii=False,sort_keys=True)+"\n"); gf.flush()
                ev=evaluate_mbppplus_program(raw,row["hidden_test"],{"code_timeout_seconds":12,"cpu_limit_seconds":8,"address_space_limit_mb":4096,"file_size_limit_mb":8,"process_limit":64},row["entry_point"])
                item={"task_id":row["task_id"],"order_index":i,"raw_output_sha256":hashlib.sha256(raw.encode()).hexdigest(),"evaluation":ev}; ef.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+"\n"); ef.flush(); evaluated.append(item)
                if i==0 or (i+1)%10==0 or i+1==len(rows): print(f"[R1] seed={args.seed} rows={i+1}/{len(rows)} correct={sum(bool(x['evaluation']['passed']) for x in evaluated)} elapsed_min={(time.time()-started)/60:.1f}",flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        log.close()
    correct=sum(bool(x["evaluation"]["passed"]) for x in evaluated); counts=Counter(x["evaluation"]["status"] for x in evaluated)
    summary={"status":"PASS","method":"full_legacy_original","seed":args.seed,"rows":len(rows),"correct":correct,"pass_at_1":correct/len(rows),"status_counts":dict(counts),"elapsed_seconds":time.time()-started}
    write_json(partial/"summary.json",summary); complete={**summary,"generation_sha256":sha256_file(partial/"generations.jsonl"),"executor_sha256":sha256_file(partial/"executor_results.jsonl")}; write_json(partial/"COMPLETE.json",complete); shutil.move(str(partial),str(final)); print(json.dumps(summary))


if __name__ == "__main__": main()
