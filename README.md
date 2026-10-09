# CoreFlow 复现代码包

[English](README_EN.md) | 中文

本目录整理了 CoreFlow 论文所使用的核心算法、实验驱动脚本、冻结协议和轻量结果摘要。目录面向 GitHub 发布，不包含基础模型、LoRA 权重、gate checkpoint、编译后的大体积 bank、原始生成文本或第三方基准数据。

## 1. 最重要的代码在哪里

| 文件 | 作用 | 类型 |
|---|---|---|
| `coreflow/decompose.py` | 构造迹归一化的共享输入/输出空间，并把每个 LoRA 更新投影为专家核心矩阵 | **CoreFlow 核心编译算法** |
| `coreflow/runtime.py` | 实现共享输入投影、gate 加权的核心空间融合和共享输出投影，并把编译产物安装到运行时 | **CoreFlow 核心在线算法** |
| `coreflow/direct_load.py` | 从独立部署产物物化运行时张量，不依赖源 LoRA bank | 部署加载代码 |
| `coreflow/io.py` | LoRA 因子、配置、模块名和哈希的读取与校验 | 资产接口 |
| `coreflow/ablation.py` | 单侧共享、无归一化、随机基等结构变体 | 消融实现 |
| `coreflow/isvd.py` | matched-budget Per-Expert SVD 对照 | 基线代码 |
| `coreflow/loraflow.py` | 原始 LoRA-Flow 资产与 gate 的加载/对接 | 源系统适配代码 |
| `coreflow/statistics.py` | 问题级聚类配对 bootstrap 等统计工具 | 统计代码 |
| `coreflow/evaluator.py` | MBPP+/代码执行评测 | 评测代码 |
| `coreflow/classeval_eval.py` | ClassEval method-level 评测 | 评测代码 |

核心公式对应的最短阅读路径是：

```text
coreflow/decompose.py::build_module_bases
  -> coreflow/decompose.py::project_core
  -> coreflow/runtime.py::coreflow_delta
  -> coreflow/runtime.py::install_coreflow
```

## 2. 论文实验与代码对应关系

| 论文问题 | 实验内容 | 代码目录 | 轻量结果 |
|---|---|---|---|
| RQ1 | 完整子空间代数正确性、编译与直接加载 | `experiments/rq1_compilation_correctness/`、`experiments/rq1_direct_load/` | `results/rq1_correctness/` |
| RQ2 | MBPP+ 主质量实验 | `experiments/rq2_mbppplus_quality/` | `results/rq2_mbppplus/` |
| RQ2 | ClassEval 独立确认 | `experiments/rq2_classeval_confirmation/` | `results/rq2_classeval/` |
| RQ2/RQ5 | q224 正式敏感性与隐藏状态漂移诊断 | `experiments/reviewer_required_v2/` | `results/reviewer_required_v2/`；公开逐题记录见 `results/submission_reproducibility_package/` |
| RQ3 | LoRA-Flow、CoreFlow、Per-Expert SVD 的吞吐、显存和 profiler 对比 | `experiments/rq3_system_efficiency/` | `results/rq3_system/` |
| RQ4 | W0--W6 工作负载与 K=3/5/8 adapter-runtime 扩展 | `experiments/rq4_workload_k_scaling/` | `results/rq4_scaling/` |
| RQ4 | Qwen3-8B 上的 Independent-Full、CoreFlow 与 CoMoL 路线对照，以及 RTX 4080 SUPER 描述性测量 | `experiments/reviewer_required_v2/` | `results/reviewer_required_v2/derived_statistics.json`；逐请求/生命周期记录见 `results/submission_reproducibility_package/` |
| RQ5 | q-grid、A0--A6、真实 gate 和路由干预 | `experiments/rq5_ablation_gate/` | `results/rq5_ablation_gate/` |

更详细的逐脚本说明见 `docs/EXPERIMENT_MAP.md`；整理前后的原始包来源见 `docs/SOURCE_PROVENANCE.md`；数据选择、哈希和再分发边界见 `docs/DATA_PROVENANCE.md`。

早期探索、被取代实现和失败实验的采用状态见 `docs/HISTORICAL_EXPERIMENTS.md`。其中 PRS-4 的代码与失败摘要保存在 `experiments/archive/` 和 `results/archive/`，不会与论文主实验混用。

## 3. 环境

论文 GPU 实验的冻结环境为：

- Ubuntu Linux，Python 3.12.3；
- NVIDIA RTX 4090 24 GB；
- NVIDIA driver 580.105.08，CUDA 12.4；
- PyTorch 2.5.1+cu124；
- Transformers 4.36.2；
- LoRA-Flow 随附的 PEFT fork（运行时报告为 `0.7.2.dev0`）；
- NumPy 1.26.4、Flask 3.0.0、Safetensors 0.8.0。

安装核心依赖前，请先根据 CUDA 环境安装匹配的 PyTorch，然后运行：

```bash
python -m pip install -r requirements.txt
```

LoRA-Flow 的 Transformers/PEFT 修改不是普通 PyPI 包。完整生成实验应先按 `docs/ASSETS.md` 准备 LoRA-Flow 代码与冻结资产。

## 4. 快速验证

不下载模型即可进行代码包静态检查：

```bash
python tools/verify_package.py
python tools/check_syntax.py
```

若修改了发布包，可重新生成文件哈希清单：

```bash
python tools/build_experiment_manifests.py
python tools/build_manifest.py
```

已安装 PyTorch 时，可额外运行共享核心代数测试：

```bash
python tests/test_shared_core_math.py
```

完整 GPU 实验需要先设置：

```bash
export PYTHONPATH="$PWD"
export MODEL=/path/to/Llama-2-7b-hf
export SOURCE_WORK_ROOT=/path/to/loraflow_assets
export FORMAL_V1_WORK_ROOT=/path/to/formal_v1_workspace
export FORMAL_V2_WORK_ROOT=/path/to/formal_v2_workspace
export CUDA_DEVICE=0
```

Qwen/CoMoL 路线对照还需要设置 `QWEN_MODEL_PATH`、`COMOL_PACKAGE_ROOT`、`QWEN_MATCH_WORK_ROOT` 和相应 Python 环境；详见 `experiments/reviewer_required_v2/README_AUTODL.md`。模型、训练权重和第三方数据不在本仓库中分发。

随后进入相应实验目录，按其冻结 README/脚本运行。原始脚本保留了实验时的 `/root/autodl-tmp/...` 默认值；推荐用环境变量覆盖，而不是直接修改冻结协议。

## 5. 复现层级

本仓库区分三种复现：

1. **算法复现**：运行合成张量测试，验证共享空间、核心投影和在线融合代数。
2. **结果重分析**：直接读取 `results/` 中的决策 JSON、CSV 和协议封印，核对论文中的统计量。
3. **端到端重跑**：自行获取第三方模型、LoRA、gate 和数据集，并根据哈希校验后运行 GPU 实验。

`results/` 保存审计文件、逐题二元结果、逐请求数值记录、生命周期记录和汇总结果，不保存大体积生成文本。这样可以重新分析论文统计量，但不能在没有第三方资产的情况下重新生成全部 token。

## 6. 发布范围

本仓库发布核心算法代码、主要实验入口、冻结协议、逐题质量结果、RTX 4080 SUPER 逐请求数值记录、Qwen 生命周期记录和轻量汇总。中性的公开证据入口为 `results/submission_reproducibility_package/`。仓库不包含模型生成文本、全部 GPU 原始日志或大体积中间产物。RQ1 五次编译与三方法 direct-load 的原始返回包未被保留，因此仅发布冻结的逐次/汇总记录，并明确标记为描述性生命周期证据。

APPS-derived 257 题开发集仅公开任务 ID、源索引和逐题哈希，不重新分发第三方题目与测试。Qwen3-8B 的固定 revision、配置哈希及 tokenizer 哈希见 `docs/manifests/QWEN3_ASSET_MANIFEST.json`；APPS-derived 清单见 `docs/manifests/APPS_DERIVED_257_MANIFEST.json`；完整的数据来源说明见 `docs/DATA_PROVENANCE.md`。

本仓库当前不授予开源许可证。使用任何材料前请阅读 `LICENSE_NOTICE.md` 和 `THIRD_PARTY_NOTICES.md`。

本次整理实际执行的静态检查和未执行项见 `docs/VALIDATION_REPORT.md`。

## 7. 第三方内容和许可证

基础模型、LoRA、gate 和基准数据不在本仓库重新分发。`vendor_canonical/modeling_llama.py` 来源于 Apache-2.0 许可的 Transformers 实现，并包含 LoRA-Flow 所需接口；详情见 `THIRD_PARTY_NOTICES.md`。

CoreFlow 自有代码当前不授予开源许可证，著作权由作者保留，详见 `LICENSE_NOTICE.md`。
