# 复现流程

## A. 无 GPU 的最低验证

```bash
python tools/verify_package.py
python tests/test_shared_core_math.py
```

第二条命令需要 PyTorch，但不需要模型权重。

## B. 结果重分析

所有用于论文的轻量决策文件位于 `results/`。优先读取：

```text
results/rq2_mbppplus/primary_decision.json
results/rq2_classeval/primary_decision.json
results/rq3_system/original_baselines_summary.json
results/rq4_scaling/prs3_pilot_summary.json
results/rq5_ablation_gate/phase2b_summary.json
results/rq5_ablation_gate/prs5_extract/.../prs5_summary.json
```

这些文件可以复核汇总值、置信区间、方法身份和协议状态。原始 generations/executor long tables 未纳入 GitHub 轻量包。

## C. 完整 GPU 重跑

1. 准备 `docs/ASSETS.md` 中的第三方资产。
2. 安装 LoRA-Flow 所需的 Transformers/PEFT fork。
3. 设置仓库根目录为 `PYTHONPATH`。
4. 设置每个冻结脚本需要的路径环境变量。
5. 先执行测试、preflight 和 seal，再运行正式生成或计时。
6. 不自动重试失败任务；保留 `.partial` 和日志。
7. 使用独立进程、固定方法顺序和协议规定的 warm-up/同步方式。

不同实验包形成于不同阶段，默认工作目录仍是历史 AutoDL 路径。环境变量具有更高优先级时应使用环境变量；若某脚本硬编码旧路径，应先复制为新协议版本并记录 diff，不能无记录地修改冻结文件。

## D. 复现结论的分级

- `VERIFIED`：代码、资产哈希和原始输出都重跑一致。
- `REANALYZED`：基于仓库中的冻结结果重新计算并一致。
- `CODE_AVAILABLE`：实现和协议存在，但缺少必要第三方资产或原始运行包。
- `PROVENANCE_GAP`：论文数值存在，但当前归档无法定位一一对应的机器原始结果。

