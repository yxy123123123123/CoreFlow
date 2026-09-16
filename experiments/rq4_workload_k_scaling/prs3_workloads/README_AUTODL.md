# CoreFlow PRS-3 Pilot v1 — AutoDL 单卡执行

## 实验性质

这是 PRS-3A/3B 的最小工程 pilot，用于验证全部指标链路和趋势，不是正式确认性 PRS-3 证据。

- 单张 RTX 4090；
- Original LoRA-Flow、CoreFlow q185、Original padded Per-Expert SVD；
- PRS-3A：6 个工作负载、24 个独立方法进程；
- PRS-3B：K3/5/8、T=1/32/128/512、七类 layer-0 模块；
- 串行执行，断点时只跳过 `COMPLETE` 或已记录的 `OOM`；
- 不自动重试、不自动降低 batch/长度、不训练 gate、不产生任务质量输出。

## 所需既有资产

默认需要：

```text
/root/autodl-tmp/Model
/root/autodl-tmp/coreflow_m0
/root/autodl-tmp/coreflow_formal_v1_workspace
/root/autodl-tmp/coreflow_formal_v2_primary_workspace
/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace
```

K8 新增 LoRA 会自动从以下任一位置寻找：

```text
/root/autodl-tmp/coreflow_m3_k_scaling_mini_upload/lora_assets/m3_k8_strict
/root/autodl-tmp/coreflow_applsci_supplement_v2_system_r1_upload/lora_assets/m3_k8_strict
```

如在其他位置，设置：

```bash
export K8_ASSET_ROOT=/你的路径/m3_k8_strict
```

## 解压与校验

```bash
cd /root/autodl-tmp
sed -i 's/\r$//' coreflow_prs3_pilot_v1_upload_20260805.tar.gz.sha256
sha256sum -c coreflow_prs3_pilot_v1_upload_20260805.tar.gz.sha256
tar -xzf coreflow_prs3_pilot_v1_upload_20260805.tar.gz
cd /root/autodl-tmp/coreflow_prs3_pilot_v1_upload
```

## 一次命令完整运行

前台运行：

```bash
set -o pipefail

export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export FORMAL_V1_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export M3_WORK_ROOT=/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs3_pilot_v1_workspace
export CUDA_DEVICE=0

bash scripts/run_prs3_pilot.sh all 2>&1 | tee /root/autodl-tmp/prs3_pilot_all_console.log
```

睡觉时推荐后台运行：

```bash
nohup bash -lc '
cd /root/autodl-tmp/coreflow_prs3_pilot_v1_upload
export MODEL=/root/autodl-tmp/Model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export FORMAL_V1_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export M3_WORK_ROOT=/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs3_pilot_v1_workspace
export CUDA_DEVICE=0
bash scripts/run_prs3_pilot.sh all
' > /root/autodl-tmp/prs3_pilot_all_console.log 2>&1 &

echo $!
```

## 查看进度和错误

```bash
cd /root/autodl-tmp/coreflow_prs3_pilot_v1_upload

export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export FORMAL_V1_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export M3_WORK_ROOT=/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs3_pilot_v1_workspace

bash scripts/run_prs3_pilot.sh status
tail -f /root/autodl-tmp/prs3_pilot_all_console.log
```

检查是否仍在运行：

```bash
ps -ef | grep -E 'run_prs3|prs3a_worker|prs3b_worker|serve_method.py' | grep -v grep || true
grep -nE 'Traceback|OutOfMemoryError|\[ERROR\]' /root/autodl-tmp/prs3_pilot_all_console.log | tail -n 30 || true
```

## 预计时间

约 3.5–6 小时。W6 如果较早记录 OOM，可能更快。

## 完成后下载

```text
/root/autodl-tmp/coreflow_prs3_pilot_v1_results.tar.gz
/root/autodl-tmp/coreflow_prs3_pilot_v1_results.tar.gz.sha256
```

如果中途失败，不要删除 `.partial`。先下载或粘贴主日志和对应 `.partial/PARTIAL.json`，由 Codex 判断如何修复。
