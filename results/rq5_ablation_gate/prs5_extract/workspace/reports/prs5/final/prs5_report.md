# CoreFlow PRS-5 A0–A6 结构证据矩阵

状态：`PRS5_COMPLETE_AS_STRUCTURAL_EVIDENCE_NO_METHOD_CHANGE`

本报告没有生成任何任务答案、没有重新选择 q，也没有改变冻结主方法。A0–A3 的质量数值全部复用 Phase-2B；A4–A6 只报告权重级与系统诊断。

## 质量与重构

| 变体 | dev128 strict pass@1 | 相对 A0 (pp, 95% CI) | fused error mean | per-expert error mean |
|---|---:|---:|---:|---:|
| A0_bilateral_normed_joint | 13/128 (10.16%) | +0.00 [+0.00, +0.00] | 0.4358 | 0.2778 |
| A1_bilateral_no_norm | 15/128 (11.72%) | +1.56 [-2.34, +6.25] | 0.3836 | 0.3973 |
| A2_input_only | 15/128 (11.72%) | +1.56 [-2.34, +5.47] | 0.6018 | 0.2934 |
| A3_output_only | 18/128 (14.06%) | +3.91 [-1.56, +9.38] | 0.5897 | 0.3793 |
| A4_plain_concat_svd | — | — | 0.5747 | 0.4834 |
| A5_gate_frequency_weighted | — | — | 0.4012 | 0.3148 |
| A6_random_orthogonal | — | — | 0.9993 | 0.9994 |

## 专家离散度与子空间

| 变体 | expert error std | IQR | range | left capture | right capture | joint capture | zero-rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0_bilateral_normed_joint | 0.1375 | 0.0069 | 0.3668 | 0.9121 | 0.9448 | 0.9024 | 0 |
| A1_bilateral_no_norm | 0.0839 | 0.0236 | 0.2255 | 0.8458 | 0.8561 | 0.8052 | 0 |
| A2_input_only | 0.2574 | 0.0017 | 0.6684 | — | 0.8502 | — | 0 |
| A3_output_only | 0.1917 | 0.0052 | 0.5300 | 0.8127 | — | — | 0 |
| A4_plain_concat_svd | 0.1257 | 0.0259 | 0.3919 | 0.8201 | 0.8927 | 0.7397 | 0 |
| A5_gate_frequency_weighted | 0.0931 | 0.0715 | 0.2590 | 0.9009 | 0.9413 | 0.8885 | 0 |
| A6_random_orthogonal | 0.0002 | 0.0003 | 0.0004 | 0.0371 | 0.0411 | 0.0015 | 0 |

## 资源与历史系统诊断

| 变体 | logical params / MAC | artifact MiB | 历史端到端 speedup vs Full-vectorized |
|---|---:|---:|---:|
| A0_bilateral_normed_joint | 500,565,600 | 954.95 | 1.0328× |
| A1_bilateral_no_norm | 500,565,600 | 969.67 | 0.9990× |
| A2_input_only | 497,860,608 | 949.75 | 1.3737× |
| A3_output_only | 497,958,912 | 949.94 | 1.3975× |
| A4_plain_concat_svd | 500,565,600 | 969.36 | 0.9848× |
| A5_gate_frequency_weighted | 500,565,600 | 969.56 | 0.9738× |
| A6_random_orthogonal | 500,565,600 | 954.97 | 0.9935× |

## 解释边界

- A0 仍是冻结正式工作点；单 seed 点估计不能触发方法替换。
- A1–A3 的配对置信区间均覆盖 0，属于机制性开发结果，不是新的确认性质量证据。
- A4–A6 没有 task-quality 数值是预先设计的空白，不得用重构误差替代 strict pass@1。
- 历史系统比率以 Full-vectorized 为参照，仅用于诊断；论文正式 Original Full 系统证据来自独立补充实验。
- `zero-rank` 是 CoreFlow/one-sided bank 的结构分配审计，不能与 Independent-SVD 的逐专家 rank allocation 混为一谈。
