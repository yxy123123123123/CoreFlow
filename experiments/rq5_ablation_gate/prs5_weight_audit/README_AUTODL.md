# CoreFlow PRS-5 A0–A6 weight audit

这是一个只做权重级审计的小包。它不会加载基座模型、不会读取任务数据、不会生成文本、不会训练 gate/LoRA，也不会选择新的 q。

## 依赖的既有目录

默认要求以下内容仍在当前 AutoDL 主机：

- `/root/autodl-tmp/coreflow_m0`
- `/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets`
- `/root/autodl-tmp/coreflow_applsci_phase2b_s6s7_v1_hotfix5_workspace`

第三个目录必须包含 Phase-2B 已构建的 A1–A6 bank。如果你的目录名不同，只需修改 `PHASE2B_WORK_ROOT`。

## 一次运行完整实验

```bash
cd /root/autodl-tmp
sha256sum -c coreflow_prs5_weight_audit_v1_upload_20260805.tar.gz.sha256
tar -xzf coreflow_prs5_weight_audit_v1_upload_20260805.tar.gz
cd /root/autodl-tmp/coreflow_prs5_weight_audit_v1_upload

set -o pipefail
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export M2_ASSET_ROOT=/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets
export PHASE2B_WORK_ROOT=/root/autodl-tmp/coreflow_applsci_phase2b_s6s7_v1_hotfix5_workspace
export WORK_ROOT=/root/autodl-tmp/coreflow_prs5_weight_audit_v1_workspace
export CUDA_DEVICE=0

mkdir -p "$WORK_ROOT/logs"
bash scripts/run_prs5.sh all 2>&1 | tee "$WORK_ROOT/logs/prs5_all.log"
```

典型耗时预计 30–90 分钟；它取决于磁盘读取速度和 4090 上 7 个 bank 的子空间矩阵计算。脚本没有自动重试，若失败会保留 `.partial` 目录。

## 查看进度

另开终端执行：

```bash
export WORK_ROOT=/root/autodl-tmp/coreflow_prs5_weight_audit_v1_workspace
tail -f "$WORK_ROOT/logs/prs5_all.log"
```

或查看阶段状态：

```bash
cd /root/autodl-tmp/coreflow_prs5_weight_audit_v1_upload
bash scripts/run_prs5.sh status
```

审计时会打印：

```text
[PRS5] variant=A0_bilateral_normed_joint modules=32/224
```

## 下载结果

成功结束后下载：

```text
/root/autodl-tmp/coreflow_prs5_weight_audit_v1_results.tar.gz
/root/autodl-tmp/coreflow_prs5_weight_audit_v1_results.tar.gz.sha256
```

将这两个文件放回 Mate 工程即可继续分析。

## 分阶段恢复

```bash
bash scripts/run_prs5.sh preflight
bash scripts/run_prs5.sh seal
bash scripts/run_prs5.sh audit
bash scripts/run_prs5.sh analyze
bash scripts/run_prs5.sh pack
```

已经完成的 `audit` 会复用正式输出。若存在 `.partial`，脚本会停止并要求人工检查，避免悄悄覆盖失败证据。

