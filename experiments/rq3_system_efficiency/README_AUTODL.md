# CoreFlow Original Full / ISVD 补充实验（AutoDL）

协议：`coreflow-applsci-original-baselines-v1.1`。本包只新增 Original Full dev128 回放与三方固定长度系统证据；不会访问 formal250、训练 gate、重选 q 或运行 compact ISVD。

## 依赖的旧资产

新主机必须已有：

- `/root/autodl-tmp/Model`
- `/root/autodl-tmp/coreflow_m0`
- `/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets`
- `/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json`
- `/root/autodl-tmp/coreflow_formal_v2_primary_workspace/isvd_banks/k5_code/matched_q185_bf16`

## 一条命令运行完整实验

解压并进入目录后执行：

```bash
bash scripts/run_original_baselines.sh all
```

脚本会依次完成：完整性校验 → 预检 → 输入制作 → R0资产审计 → 协议封印 → R1三种子回放 → R2单卡串行系统矩阵 → R3轻量profiler → 分层统计 → 打包。

默认使用 GPU 0。若需指定：

```bash
CUDA_DEVICE=0 bash scripts/run_original_baselines.sh all
```

进度与错误日志写入：

```text
/root/autodl-tmp/coreflow_applsci_original_baselines_v1_workspace/logs/
```

另开终端查看状态：

```bash
cd /root/autodl-tmp/coreflow_applsci_original_baselines_v1_upload
bash scripts/run_original_baselines.sh status
```

持续查看主阶段日志：

```bash
tail -F /root/autodl-tmp/coreflow_applsci_original_baselines_v1_workspace/logs/*.log
```

成功后下载：

```text
/root/autodl-tmp/coreflow_applsci_original_baselines_v1_results.tar.gz
/root/autodl-tmp/coreflow_applsci_original_baselines_v1_results.tar.gz.sha256
```

预计单张 RTX 4090 总耗时约 7–11 小时。任何失败都会立即停止，保留 `.partial`，且不会自动重试或覆盖。
