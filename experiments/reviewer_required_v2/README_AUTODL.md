# CoreFlow reviewer required-v2

本包对应已确认的“五层实验结构”，并采用 CoMoL group2/group3 完整六数据集质量评测。

## 会做什么

必须阶段：

1. 本地/CPU 编译接口拒绝测试；
2. CoreFlow/ISVD logical、effective、stored、runtime 口径审计；
3. group1 生命周期成本整理；
4. 理论误差与逐层传播分析；
5. 多 seed/gate 统计口径冻结；
6. Llama q224 MBPP+ 正式敏感性质量：250×3；
7. Llama q224 ClassEval 正式敏感性质量：202×3；
8. Qwen3 group2：8 个独立 LoRA + CoMoL + gate + CoreFlow q185 + 六数据集质量；
9. Qwen3 group3：同上；
10. 统一分析和结果打包。

强烈建议阶段：

- RTX 4080 SUPER 与 RTX 4090 跨硬件服务矩阵；
- TTFT/TPOT/P50/P95/P99/功耗/J token；
- Llama 32 条 hidden-state drift；
- 异构 rank 与部分模块覆盖接口测试；
- K=8 Pareto 统一表。

group2/group3 每组完整跑六个数学数据集，默认不重复 q16/q24 和系统实验，因为 group1 已经承担 q-grid 和方法路线系统对照。

## 依赖的已有目录

不需要重新上传模型和基础资产，但新主机必须存在：

```text
/root/autodl-tmp/Model
/root/autodl-tmp/coreflow_m0
/root/autodl-tmp/coreflow_formal_v2_primary_upload
/root/autodl-tmp/coreflow_formal_v2_primary_workspace
/root/autodl-tmp/coreflow_indep_confirm_v1_upload
/root/autodl-tmp/coreflow_indep_confirm_v1_workspace
/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace
/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload
```

其中 Qwen matched workspace 的 `groups/group1` 是已完成 group1 的只读复用源。

## 上传与环境

```bash
cd /root/autodl-tmp
tr -d '\r' < coreflow_reviewer_required_v2_upload.tar.gz.sha256 > /tmp/required_v2.sha256
sha256sum -c /tmp/required_v2.sha256
tar -xzf coreflow_reviewer_required_v2_upload.tar.gz
cd /root/autodl-tmp/coreflow_reviewer_required_v2_upload

export MODEL_PATH=/root/autodl-tmp/coreflow_qwen3_load_smoke_v1_upload/assets/base_model
export SOURCE_WORK_ROOT=/root/autodl-tmp/coreflow_m0
export FORMAL_V2_PACKAGE_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_upload
export FORMAL_V2_WORK_ROOT=/root/autodl-tmp/coreflow_formal_v2_primary_workspace
export CLASSEVAL_PACKAGE_ROOT=/root/autodl-tmp/coreflow_indep_confirm_v1_upload
export CLASSEVAL_WORK_ROOT=/root/autodl-tmp/coreflow_indep_confirm_v1_workspace
export QWEN_MATCH_WORK_ROOT=/root/autodl-tmp/coreflow_joint_vs_posthoc_v1_workspace
export COMOL_PACKAGE_ROOT=/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload
export MATH_DATA_ROOT=/root/autodl-tmp/coreflow_comol_official_reproduction_v1_upload/datasets
export CORE_PYTHON=/root/autodl-tmp/coreflow_m0/runtime_env/bin/python
export COMOL_PYTHON=/root/autodl-tmp/comol_official_env/bin/python
export WORK_ROOT=/root/autodl-tmp/coreflow_reviewer_required_v2_workspace
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$WORK_ROOT/logs"
```

## 推荐运行顺序

先做封印和本地审计：

```bash
bash scripts/run_reviewer_required.sh preflight
bash scripts/run_reviewer_required.sh prepare
bash scripts/run_reviewer_required.sh seal
bash scripts/run_reviewer_required.sh local-audit 2>&1 | tee "$WORK_ROOT/logs/01_local_audit.log"
```

然后运行必须 GPU 阶段：

```bash
nohup bash scripts/run_reviewer_required.sh all-must \
  > "$WORK_ROOT/logs/all_must.log" 2>&1 &
echo $! | tee "$WORK_ROOT/logs/all_must.pid"
```

如果 `all-must` 已完成某些阶段，重复运行会跳过已有 group manifest、bank、quality metrics 和 q224 完成目录，不会重新训练已完成的部分。

监控：

```bash
bash scripts/run_reviewer_required.sh status
tail -f "$WORK_ROOT/logs/all_must.log"
```

## 强烈建议的服务阶段

当前 Qwen matched group1 已有 q16 bank，因此服务阶段使用 `CoreFlow q16`，不会错误声称 q185。Llama 的 q185 系统结果沿用既有 Phase2A 证据。

在第一台卡上：

```bash
export HARDWARE_LABEL=RTX_4080_SUPER
bash scripts/run_reviewer_required.sh service 2>&1 | tee "$WORK_ROOT/logs/service_4080.log"
```

在另一台卡上，复用同一 workspace 和结果目录：

```bash
export HARDWARE_LABEL=RTX_4090
bash scripts/run_reviewer_required.sh service 2>&1 | tee "$WORK_ROOT/logs/service_4090.log"
```

`c4` 是单 worker 顺序排队负载，不是 continuous batching；论文不能将其描述为真实动态批处理吞吐。

## 分析与打包

```bash
bash scripts/run_reviewer_required.sh analyze
bash scripts/run_reviewer_required.sh pack
```

结果默认位于：

```text
/root/autodl-tmp/coreflow-reviewer-required-v2_results.tar.gz
/root/autodl-tmp/coreflow-reviewer-required-v2_results.tar.gz.sha256
```

不要删除 group1 的原始 workspace，也不要用 group2/group3 输出覆盖 group1。最终报告必须将 group1 视为既有结果，将 group2/group3 视为新增完整六数据集重复。
