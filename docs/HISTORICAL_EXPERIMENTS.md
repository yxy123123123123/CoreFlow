# 历史实验整理与采用状态

本项目经历了边界搜索、实现验收、确认性质量实验、系统比较和投稿前补充实验。GitHub 复现包不按 `E:\Mate` 的数十个重复上传/校验目录逐份复制，而是按“同一研究问题的最终冻结实现”去重。下表说明各阶段的去向，避免只保留有利结果。

| 阶段 | 主要问题 | 当前处理 | 原因 |
|---|---|---|---|
| M0 rank/oracle/precision 系列 | 找到截断边界，区分代数错误与有限 q 误差 | 不作为正式入口；来源目录保留在 `E:\Mate` | 探索性开发与实现诊断，已被 full-subspace 正确性和正式 q-grid 覆盖 |
| M1a | q=185 早期代码确认 | 不作为主证据 | 存在审计偏差，后续由 clean/正式实验替代 |
| M1b-clean | q=185 与 q=224 的新数据比较 | 作为 q 工作点历史依据记录，不作为独立主结论 | 绝对正确率处于低地板区 |
| M2 K-exp/cross-task | K 与跨任务边界探索 | 不作为最终系统扩展入口 | 被 M3、PRS-3 和正式质量实验取代 |
| formal-v1 / F0 / F0R / F0R2 | 数据、评测器和离线重评分验收 | 保留其最终协议含义，不重复复制中间包 | formal-v2 primary 是最终冻结主实验 |
| formal-v2 primary | MBPP+ 三方法主质量实验 | **已纳入 RQ2** | 论文主质量证据 |
| M3 / M3G / M3R | K 扩展、真实 gate 与路由诊断 | **已纳入 RQ4/RQ5** | 机制和适用边界证据 |
| Phase-1S / Phase-2A | Full-vectorized、compact ISVD 与 profiler 工程验收 | 不作为论文主入口 | 最终论文明确比较原始 LoRA-Flow、CoreFlow 与原始 Per-Expert SVD；compact ISVD 未通过 token 等价门槛 |
| Phase-2B | q-grid、A0--A6、prefill/TTFT | **已纳入 RQ5** | 正式敏感性与消融材料 |
| OriginalFull/ISVD | 原始 LoRA-Flow、CoreFlow、原始 Per-Expert SVD 系统比较 | **已纳入 RQ3** | 论文主系统证据 |
| PRS-0/1 | 数据集与统计口径审计 | 轻量重分析表已纳入 RQ2 | 支持问题级配对与失败类型分析 |
| PRS-2 | 编译时间与 direct-load 生命周期 | 代码原型已纳入 RQ1，最终原始结果缺失 | 当前归档无法定位与论文汇总值一一对应的返回包 |
| PRS-3 | W0--W6 与 K×T | **已纳入 RQ4，并保留 pilot 标签** | 可用于工作负载诊断，但不可改写成未执行的正式重复实验 |
| PRS-4 | 新参考方法确认 | **代码和失败摘要纳入 `experiments/archive/`、`results/archive/`** | 参考方法低于预设质量地板，不能进入主比较，但失败记录应公开 |
| PRS-5 | gate/effective-weight 审计 | **已纳入 RQ5** | 解释 signed effective weights 和 gate 行为 |
| ClassEval independent confirmation | 独立代码基准确认 | **已纳入 RQ2** | 独立确认性质量证据 |

## 为什么没有逐份复制所有 `_verify_*` 目录

`E:\Mate` 中大量 `_verify_*`、hotfix、archive-verify 和重复上传目录是同一实验包的传输/校验副本。逐份复制会造成代码重复和版本歧义，不会增加可复现性。本包保留：

1. 论文实际采用的最终冻结代码；
2. 与之对应的协议和轻量结果；
3. 关键失败实验 PRS-4；
4. 原始上传包哈希清单；
5. 明确的来源路径和未解决 provenance gap。

若需要做取证式全量归档，应另外制作只读冷存储，而不应把数 GB 的权重、trace、tar 包和重复校验目录推送到 GitHub。
