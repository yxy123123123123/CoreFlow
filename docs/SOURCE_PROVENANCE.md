# 原始实验包来源映射

本文件记录本复现包各部分从 `E:\Mate` 中哪些冻结实验包整理而来。整理过程只复制代码、配置、协议封印和轻量结果，不改写原始实验逻辑，也不重新分发模型、LoRA、gate、数据集或生成长表。

| 整理后内容 | 原始来源 | 在论文中的角色 |
|---|---|---|
| `coreflow/` 主体 | `E:\Mate\coreflow_submission_final_stage1_v1_upload\coreflow` | CoreFlow 编译、运行时、direct-load 与统计实现 |
| `coreflow/evaluator.py`、`classeval_eval.py`、`data_audit.py` | `E:\Mate\coreflow_indep_confirm_v1_upload\coreflow` | MBPP+/ClassEval 评测与数据审计补充 |
| `vendor_canonical/modeling_llama.py` | `E:\Mate\coreflow_submission_final_stage1_v1_upload\vendor_canonical` | LoRA-Flow 所需的模型接口快照 |
| `experiments/rq1_compilation_correctness/` | `E:\Mate\coreflow_formal_v2_primary_upload` | full-subspace 代数与数值正确性 |
| `experiments/rq1_direct_load/` | `E:\Mate\coreflow_submission_final_stage1_v1_upload` | direct-load 实现与验收原型 |
| `experiments/rq2_mbppplus_quality/` | `E:\Mate\coreflow_formal_v2_primary_upload` | MBPP+ 冻结主质量实验 |
| `experiments/rq2_classeval_confirmation/` | `E:\Mate\coreflow_indep_confirm_v1_upload` | ClassEval 独立确认实验 |
| `experiments/rq3_system_efficiency/` | `E:\Mate\coreflow_applsci_original_baselines_v1_upload` | 原始 LoRA-Flow、CoreFlow 与 Per-Expert SVD 系统比较 |
| `experiments/rq4_workload_k_scaling/prs3_workloads/` | `E:\Mate\coreflow_prs3_pilot_v1_upload` | W0--W6 工作负载与 adapter-only 扩展 |
| `experiments/rq4_workload_k_scaling/m3_k_scaling/` | `E:\Mate\coreflow_m3_k_scaling_mini_upload` | K=3/5/8 早期机制实验 |
| `experiments/rq5_ablation_gate/phase2b_qgrid_ablation/` | `E:\Mate\coreflow_applsci_phase2b_final_v1` | q-grid、A0--A6 与预填充/TTFT 实验 |
| `experiments/rq5_ablation_gate/prs5_weight_audit/` | `E:\Mate\coreflow_prs5_weight_audit_v1_upload` | gate/effective-weight 审计 |
| `experiments/rq5_ablation_gate/m3g_real_gate/` | `E:\Mate\coreflow_m3g_real_gate_entropy_upload` | 真实 gate 熵与权重统计 |
| `experiments/rq5_ablation_gate/m3r_routing_diagnostics/` | `E:\Mate\coreflow_m3r_gate_diagnostics_upload` | 路由干预、输出误差和结构相关性 |
| `experiments/archive/prs4_reference_method_failed/` | `E:\Mate\coreflow_prs4a_prs4p_pilot_v1_upload` | 未通过质量地板的参考方法实验，作为失败归档 |

## 轻量结果来源

| 整理后目录 | 原始结果目录或文件 |
|---|---|
| `results/rq1_correctness/` | `E:\Mate\_formal_v2_primary_results_20260729\reports\formal_v2_primary` |
| `results/rq2_mbppplus/` | `E:\Mate\_formal_v2_primary_results_20260729\reports\formal_v2_primary` 及 PRS-1 重分析表 |
| `results/rq2_classeval/` | `E:\Mate\indep_results\reports\indep_confirm_classeval` |
| `results/rq3_system/` | `E:\Mate\_original_baselines_results_20260803\coreflow_applsci_original_baselines_v1_results\reports\original_baselines_v1` |
| `results/rq4_scaling/` | PRS-3 final summary/tables 与 M3 K-scaling 决策 JSON |
| `results/rq5_ablation_gate/` | Phase-2B、PRS-5、M3G 与 M3R 的汇总 JSON/CSV |
| `results/archive/prs4_reference_method_failed/` | `E:\Mate\_prs4a_prs4p_results_20260805\workspace\reports\prs4a_prs4p\final` |

## 未合并的内容

- 未复制 `E:\Mate` 中的 `.tar`、`.tar.gz`、模型权重、LoRA/gate checkpoint、编译 bank、缓存、trace 和大体积逐题生成结果。
- Phase-2A Full-vectorized 路径不属于论文最终三方法比较，未纳入 RQ3 主入口。
- ISVD-compact 未通过严格自回归 token 等价验收，不作为正式系统基线。
- PRS-4 的参考方法低于预设质量地板，不进入论文主证据链；其代码和轻量失败证据已单独归档。
- 早期开发、重复上传包和已被后续冻结实验取代的脚本不重复收录。

## 仍需作者补档的来源

论文 RQ1 所述五次完整编译时间和三方法 direct-load 生命周期汇总，在当前可访问的 `E:\Mate` 材料中没有找到一一对应的原始返回包。本仓库当前定位为核心算法代码包，因此将其记录为实验档案范围限制，而不列为代码发布阻塞项；若以后宣称提供完整结果复现，再补入对应日志、命令、产物哈希和汇总脚本。
