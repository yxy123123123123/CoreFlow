#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, core_bank, gate_path, isvd_bank, paths, request_json, sha256_file, wait_health, write_json

def snapshot(device):
    q=subprocess.run(["nvidia-smi",f"--id={device}","--query-gpu=index,name,uuid,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,check=False)
    return {"values_csv":q.stdout.strip(),"captured_unix":time.time()}

def batch(rows,index,size):
    start=(index*size)%len(rows); return [rows[(start+j)%len(rows)] for j in range(size)]

def stats(rows):
    keys=["decode_tokens_per_second","total_tokens_per_second","engine_wall_total_seconds","engine_cuda_total_seconds","prefill_wall_seconds","decode_wall_seconds","server_ttft_seconds"]
    out={}
    for key in keys:
        vals=[float(r["timing"][key]) for r in rows if r["timing"].get(key) is not None]
        out[key]={"n":len(vals),"mean":statistics.fmean(vals),"median":statistics.median(vals),"min":min(vals),"max":max(vals),"sd":statistics.stdev(vals) if len(vals)>1 else 0.0}
    for key in ["peak_memory_allocated_bytes","peak_memory_reserved_bytes","startup_resident_allocated_bytes","startup_resident_reserved_bytes","incremental_peak_allocated_bytes","incremental_peak_reserved_bytes"]:
        out[key]=max(int(r["timing"][key]) for r in rows)
    return out

def main():
    ap=argparse.ArgumentParser()
    for name in ["method","config-name","block-id","output"]: ap.add_argument(f"--{name}",required=True)
    for name in ["order-position","batch-size","input-length","output-length","warmup","measurements","port"]: ap.add_argument(f"--{name}",type=int,required=True)
    args=ap.parse_args(); cfg,p=config(),paths()
    if not (p["reports"]/"PROTOCOL_SEAL.json").is_file(): raise RuntimeError("Run seal first")
    final=Path(args.output); partial=Path(str(final)+".partial")
    if (final/"COMPLETE.json").is_file(): print(f"[SKIP] {final}"); return
    if final.exists() or partial.exists(): raise RuntimeError(f"Retained partial/final output exists: {final}")
    partial.mkdir(parents=True); input_path=p["data"]/f"effective_inputs_l{args.input_length}.json"; inputs=json.loads(input_path.read_text(encoding="utf-8"))
    if any(len(x["input_ids"])!=args.input_length or sum(x["attention_mask"])!=args.input_length for x in inputs): raise ValueError("True effective input contract failed")
    command=[str(p["runtime_python"]),str(ROOT/"scripts"/"serve_method.py"),"--model",str(p["model"]),"--asset-root",str(p["asset_root"]),"--official-root",str(p["official_root"]),"--adapter-order",",".join(cfg["expert_order"]),"--gate",str(gate_path(cfg,p,cfg["r2_systems"]["gate_seed"])),"--method",args.method,"--port",str(args.port),"--output-length",str(args.output_length),"--fixed-output"]
    if args.method=="coreflow_q185_frozen": command += ["--bank-dir",str(core_bank(cfg,p))]
    if args.method=="isvd_legacy_padded_matched_q185": command += ["--bank-dir",str(isvd_bank(cfg,p))]
    write_json(partial/"resolved_run.json",{"command":command,"block_id":args.block_id,"order_position":args.order_position,"input_sha256":sha256_file(input_path),"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")})
    log=(partial/"server.log").open("w",encoding="utf-8",newline="\n"); before=snapshot(p["cuda_device"]); startup=time.perf_counter(); proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,text=True); measured=[]
    try:
        health=wait_health(f"http://127.0.0.1:{args.port}/health",proc); client_startup=time.perf_counter()-startup
        for i in range(args.warmup):
            b=batch(inputs,i,args.batch_size); request_json(f"http://127.0.0.1:{args.port}/infer",{"input_ids":[x["input_ids"] for x in b],"attention_mask":[x["attention_mask"] for x in b],"seed":970000+i})
        for i in range(args.measurements):
            b=batch(inputs,i,args.batch_size); client=time.perf_counter(); response=request_json(f"http://127.0.0.1:{args.port}/infer",{"input_ids":[x["input_ids"] for x in b],"attention_mask":[x["attention_mask"] for x in b],"seed":980000+i}); elapsed=time.perf_counter()-client
            if response["input_tokens_per_sequence"]!=args.input_length or response["new_tokens_per_sequence"]!=args.output_length: raise ValueError("Fixed length response contract failed")
            timing=dict(response["timing"]); timing["client_total_seconds"]=elapsed
            row={"protocol_id":cfg["protocol_id"],"config_name":args.config_name,"block_id":args.block_id,"order_position":args.order_position,"method":args.method,"request_index":i,"task_ids":[x["task_id"] for x in b],"timing":timing,"generated_token_ids_sha256":response["generated_token_ids_sha256"]}; measured.append(row)
            with (partial/"measurements.jsonl").open("a",encoding="utf-8",newline="\n") as f: f.write(json.dumps(row,sort_keys=True)+"\n")
            print(f"[R2] config={args.config_name} block={args.block_id} method={args.method} measured={i+1}/{args.measurements} decode_tok_s={timing['decode_tokens_per_second']:.3f}",flush=True)
    except Exception as exc:
        log.flush()
        log_text=(partial/"server.log").read_text(encoding="utf-8",errors="replace")
        if "out of memory" in (repr(exc)+"\n"+log_text).lower():
            write_json(partial/"OOM.json",{"status":"OOM","method":args.method,"config_name":args.config_name,"block_id":args.block_id,"batch_size":args.batch_size,"input_length":args.input_length,"output_length":args.output_length,"error":repr(exc),"gpu_before":before})
            oom=Path(str(final)+".oom"); shutil.move(str(partial),str(oom)); print(json.dumps({"status":"OOM","method":args.method,"config":args.config_name,"block":args.block_id})); raise SystemExit(20)
        write_json(partial/"PARTIAL.json",{"status":"PARTIAL","error":repr(exc)}); raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)
        log.close()
    summary={"status":"PASS","method":args.method,"config_name":args.config_name,"block_id":args.block_id,"order_position":args.order_position,"requests":len(measured),"warmup":args.warmup,"measurements":args.measurements,"batch_size":args.batch_size,"input_length":args.input_length,"output_length":args.output_length,"client_startup_to_health_seconds":client_startup,"server_health":health,"gpu_before":before,"gpu_after":snapshot(p["cuda_device"]),"statistics":stats(measured)}
    write_json(partial/"summary.json",summary); write_json(partial/"COMPLETE.json",{"status":"PASS","measurements_sha256":sha256_file(partial/"measurements.jsonl"),"summary_sha256":sha256_file(partial/"summary.json")}); shutil.move(str(partial),str(final)); print(json.dumps({"status":"PASS","method":args.method,"config":args.config_name,"block":args.block_id}))

if __name__=="__main__": main()
