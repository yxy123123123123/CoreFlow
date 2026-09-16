#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.common import config, core_bank, gate_path, isvd_bank, paths, sha256_file, write_json

cfg,p=config(),paths(); errors=[]
if not p["runtime_python"].is_file(): errors.append(f"runtime python missing: {p['runtime_python']}")
if not p["model"].is_dir(): errors.append(f"model missing: {p['model']}")
for seed,expected in cfg["assets"]["gate_sha256"].items():
    path=gate_path(cfg,p,int(seed));
    if not path.is_file() or sha256_file(path)!=expected: errors.append(f"gate hash mismatch: {path}")
for bank,file,expected in [(core_bank(cfg,p),"core_bank.safetensors",cfg["assets"]["core_bank_sha256"]),(isvd_bank(cfg,p),"isvd_bank.safetensors",cfg["assets"]["isvd_bank_sha256"])]:
    target=bank/file
    if not target.is_file() or sha256_file(target)!=expected: errors.append(f"bank hash mismatch: {target}")
for name,item in cfg["assets"]["experts"].items():
    root=p["asset_root"]/"LoRAs"/item["relative"]; conf=root/"adapter_config.json"; candidates=[root/"adapter_model.safetensors",root/"adapter_model.bin"]
    ckpt=next((x for x in candidates if x.is_file()),None)
    if not conf.is_file() or sha256_file(conf)!=item["config_sha256"]: errors.append(f"expert config mismatch: {name}")
    if ckpt is None or sha256_file(ckpt)!=item["checkpoint_sha256"]: errors.append(f"expert checkpoint mismatch: {name}")
dep=subprocess.run([str(p["runtime_python"]),"-c","import torch,transformers,peft,flask,safetensors,numpy; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],capture_output=True,text=True,env={**os.environ,"CUDA_VISIBLE_DEVICES":p["cuda_device"]}) if p["runtime_python"].is_file() else None
if dep is not None and dep.returncode: errors.append("dependency/GPU check failed: "+dep.stderr[-1000:])
if dep is not None and dep.returncode==0 and "4090" not in dep.stdout: errors.append("Required RTX 4090 not detected")
data=ROOT/"data"/cfg["data"]["dev128_file"]; prompt=ROOT/"data"/cfg["data"]["prompt_file"]
if sha256_file(data)!=cfg["data"]["dev128_sha256"] or len(data.read_text(encoding="utf-8").splitlines())!=128: errors.append("embedded dev128 mismatch")
if sha256_file(prompt)!=cfg["data"]["prompt_sha256"]: errors.append("embedded prompt mismatch")
report={"status":"PASS" if not errors else "FAIL","protocol_id":cfg["protocol_id"],"gpu":None if dep is None else dep.stdout.strip(),"errors":errors}
p["reports"].mkdir(parents=True,exist_ok=True); write_json(p["reports"]/"preflight.json",report); print(json.dumps(report));
if errors: raise SystemExit(1)
