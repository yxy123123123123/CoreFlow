# AutoDL Stage-1 执行手册

## 1. 上传与解压

上传以下两个文件到 `/root/autodl-tmp`：

- `coreflow_submission_final_stage1_v1_upload_20260802.tar.gz`
- `coreflow_submission_final_stage1_v1_upload_20260802.tar.gz.sha256`

```bash
cd /root/autodl-tmp
sed -i 's/\r$//' coreflow_submission_final_stage1_v1_upload_20260802.tar.gz.sha256
sha256sum -c coreflow_submission_final_stage1_v1_upload_20260802.tar.gz.sha256
tar -xzf coreflow_submission_final_stage1_v1_upload_20260802.tar.gz
cd /root/autodl-tmp/coreflow_submission_final_stage1_v1_upload
```

## 2. 环境变量

```bash
set -o pipefail
export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export M2_ASSET_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
export M2_ASSET_LOCK=/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace
export CUDA_DEVICE=0
export PYTHONPATH="$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/transformers/src:$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/peft-group_lora/src:${PYTHONPATH:-}"
mkdir -p "$WORK_ROOT/logs"
```

## 3. 推荐分阶段运行

先做不产生模型输出的封印：

```bash
bash scripts/run_stage1.sh all-prepare 2>&1 | tee "$WORK_ROOT/logs/stage1_prepare_console.log"
bash scripts/run_stage1.sh status
```

然后运行qualification。预计约2–5小时，取决于生成长度和执行超时：

```bash
bash scripts/run_stage1.sh qualification 2>&1 | tee "$WORK_ROOT/logs/stage1_qualification_console.log"
bash scripts/run_stage1.sh qualify
bash scripts/run_stage1.sh status
```

只有输出 `QUALIFICATION_PASS` 才继续：

```bash
bash scripts/run_stage1.sh e1-isvd 2>&1 | tee "$WORK_ROOT/logs/stage1_e1_isvd_console.log"
bash scripts/run_stage1.sh e1d 2>&1 | tee "$WORK_ROOT/logs/stage1_e1d_console.log"
bash scripts/run_stage1.sh decide
bash scripts/run_stage1.sh pack
```

E1预计1–3小时；E1D包含多个独立模型进程，预计1–3小时。整包通常约4–11 GPU小时。

## 4. 一次跑完

如果可以接受遇错立即停止且不自动重试：

```bash
nohup bash -lc '
cd /root/autodl-tmp/coreflow_submission_final_stage1_v1_upload
export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export M2_ASSET_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
export M2_ASSET_LOCK=/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace
export CUDA_DEVICE=0
export PYTHONPATH="$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/transformers/src:$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/peft-group_lora/src:${PYTHONPATH:-}"
bash scripts/run_stage1.sh all
' > /root/autodl-tmp/coreflow_submission_final_stage1_v1_all.log 2>&1 &
echo $!
```

## 5. 监控

```bash
tail -f /root/autodl-tmp/coreflow_submission_final_stage1_v1_all.log
```

另开终端查看结构化状态：

```bash
cd /root/autodl-tmp/coreflow_submission_final_stage1_v1_upload
bash scripts/run_stage1.sh status
nvidia-smi
```

检查是否退出或报错：

```bash
pgrep -af 'run_stage1|run_qualification|serve_full|run_e1_isvd|run_e1d' || true
grep -nE 'Traceback|OutOfMemory|\[ERROR\]|E1_ISVD_FAIL|NO_GO' /root/autodl-tmp/coreflow_submission_final_stage1_v1_all.log | tail -n 30 || true
```

## 6. 下载结果

成功pack后下载：

```text
/root/autodl-tmp/coreflow_submission_final_stage1_v1_results.tar.gz
/root/autodl-tmp/coreflow_submission_final_stage1_v1_results.tar.gz.sha256
```

不要删除 `.partial` 目录。失败后先把日志和状态发回本地分析，禁止自动重跑。
