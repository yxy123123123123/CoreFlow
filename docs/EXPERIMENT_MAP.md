# 实验—代码—结果映射

## RQ1：编译正确性与独立加载

| 内容 | 入口 | 说明 |
|---|---|---|
| 完整子空间数值测试 | `experiments/rq1_compilation_correctness/run_e0_correctness.py` | 使用无截断的合成异构 rank 配置验证 CoreFlow 代数和精度 |
| 正式包测试 | `experiments/rq1_compilation_correctness/tests/` | 协议、张量和实现合同测试 |
| direct-load 原型 | `experiments/rq1_direct_load/scripts/run_e1d.py` | 比较同一方法的 legacy 路径与直接加载路径 |
| 单方法 direct-load | `experiments/rq1_direct_load/scripts/run_e1d_method.py` | 在独立进程中执行物化和数值检查 |

注意：`rq1_direct_load` 来自 submission-final Stage-1，包含 LoRA-Flow、CoreFlow 和当时实验性 ISVD-compact 路径。论文最终系统比较使用的是原始 LoRA-Flow 与原始 padded Per-Expert SVD；不得把 compact ISVD 的失败验收结果写成正式系统基线。

## RQ2：有限共享维度下的任务质量

### MBPP+

| 内容 | 入口 |
|---|---|
| 完整主流程 | `experiments/rq2_mbppplus_quality/scripts/run_v2_primary.sh` |
| 单方法生成/评测 | `experiments/rq2_mbppplus_quality/scripts/run_formal_method.py` |
| 两卡调度 | `experiments/rq2_mbppplus_quality/scripts/run_formal_parallel.py` |
| 统计分析 | `experiments/rq2_mbppplus_quality/scripts/analyze_v2_primary.py` |
| 协议封印 | `experiments/rq2_mbppplus_quality/scripts/v2_control.py` |

主结果在 `results/rq2_mbppplus/primary_decision.json`。PRS-1 重分析表位于同目录的 `prs1_*.csv`。

### ClassEval

| 内容 | 入口 |
|---|---|
| 完整确认流程 | `experiments/rq2_classeval_confirmation/scripts/run_indep_confirm.sh` |
| 数据拆分 | `experiments/rq2_classeval_confirmation/scripts/build_classeval_splits.py` |
| qualification 决策 | `experiments/rq2_classeval_confirmation/scripts/decide_qualification.py` |
| 正式生成/评测 | `experiments/rq2_classeval_confirmation/scripts/run_formal_method.py` |
| 统计分析 | `experiments/rq2_classeval_confirmation/scripts/analyze_v2_primary.py` |

主结果在 `results/rq2_classeval/primary_decision.json`。

## RQ3：容量、显存和运行效率

| 内容 | 入口 |
|---|---|
| 完整三方法流程 | `experiments/rq3_system_efficiency/scripts/run_original_baselines.sh` |
| 固定负载矩阵 | `experiments/rq3_system_efficiency/scripts/run_r2_matrix.py` |
| 单进程 worker | `experiments/rq3_system_efficiency/scripts/r2_worker.py` |
| profiler | `experiments/rq3_system_efficiency/scripts/run_r3.py`、`profile_worker.py` |
| 汇总统计 | `experiments/rq3_system_efficiency/scripts/analyze.py` |

论文主系统结果来自 `results/rq3_system/original_baselines_summary.json`，而不是 Phase-2A 的 Full-vectorized 路径。

## RQ4：工作负载和专家数扩展

| 内容 | 入口 |
|---|---|
| W0--W6 与 K×T pilot | `experiments/rq4_workload_k_scaling/prs3_workloads/scripts/run_prs3_pilot.sh` |
| 工作负载矩阵 | `.../scripts/run_prs3a.py` |
| adapter-only K×T | `.../scripts/run_prs3b.py` |
| 早期 K 扩展重构实验 | `experiments/rq4_workload_k_scaling/m3_k_scaling/scripts/m3_k_scaling.py` |

结果分别位于 `results/rq4_scaling/prs3_pilot_summary.json` 和 `m3_*.json`。PRS-3 文件自身标记为 pilot；论文若使用其数值，应保持这一来源说明。

## RQ5：结构消融与 gate 机制

| 内容 | 入口 |
|---|---|
| q-grid 与 A0--A6 bank | `experiments/rq5_ablation_gate/phase2b_qgrid_ablation/scripts/build_qgrid_banks.py` |
| q-grid 质量运行 | `.../scripts/run_qgrid.py` |
| PRS-5 权重审计 | `experiments/rq5_ablation_gate/prs5_weight_audit/scripts/run_prs5.sh` |
| 真实 gate 统计 | `experiments/rq5_ablation_gate/m3g_real_gate/scripts/m3g_real_gate_entropy.py` |
| 路由干预与残差 | `experiments/rq5_ablation_gate/m3r_routing_diagnostics/scripts/m3r_gate_diagnostics.py` |

对应结果在 `results/rq5_ablation_gate/`。

## Reviewer-required v2：跨路线、跨模型与补充敏感性

| 内容 | 入口 | 轻量结果 |
|---|---|---|
| q224 在 MBPP+ 和 ClassEval 上的正式敏感性 | `experiments/reviewer_required_v2/scripts/run_q224.py` | `results/reviewer_required_v2/q224_complete.json`、`derived_statistics.json` |
| Qwen3-8B Independent-Full 与 CoreFlow | `.../scripts/quality_core_worker.py`、`system_core_worker.py` | `derived_statistics.json` |
| Qwen3-8B CoMoL 联合训练路线 | `.../scripts/train_comol.py`、`run_comol_groups.py`、`quality_comol.py`、`system_comol_worker.py` | `derived_statistics.json` |
| RTX 4080 SUPER 描述性 service 测量 | `.../scripts/run_service.py` | `derived_statistics.json` |
| 隐藏状态漂移诊断 | `.../scripts/run_drift.py` | `drift_diagnostics.json`、`derived_statistics.json` |
| 编译接口、存储和生命周期审计 | `.../scripts/compile_interface_tests.py`、`storage_audit.py`、`lifecycle_audit.py` | 同名 JSON 审计文件 |

可公开重分析的逐题二元结果、RTX 4080 SUPER 逐请求记录及 Qwen group 2/3 生命周期计时集中在 `results/submission_reproducibility_package/`。该包不重新分发 Qwen、LoRA、gate、CoMoL 权重、基准题目、原始生成文本或编译 bank。资产与数据身份分别见 `docs/manifests/QWEN3_ASSET_MANIFEST.json` 和 `docs/DATA_PROVENANCE.md`。
