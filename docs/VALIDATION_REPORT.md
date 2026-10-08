# 本地整理与校验报告

校验日期：2026-10-08

| 检查 | 结果 |
|---|---|
| Python 源码 AST 解析 | 通过，176 个 `.py` 文件 |
| JSON 结构解析 | 通过，72 个 `.json` 文件 |
| 13 个实验/归档子包完整性清单 | 全部重建并通过 |
| Reviewer-required v2 编译接口合同测试 | 通过，5 个拒绝用例和 1 个最小合法用例均符合预期 |
| 大文件、权重、压缩包、缓存与常见密钥模式扫描 | 通过 |
| 本机 Windows 绝对路径扫描 | 通过；公开文件中未保留作者工作区路径 |
| GitHub 单文件 100 MiB 限制 | 通过 |
| 整理后总体积 | 约 2.43 MiB |
| 合成共享核心 PyTorch 单元测试 | 当前 Windows 整理环境缺少 PyTorch，启动时报告 `ModuleNotFoundError: No module named 'torch'`，未执行 |

`tests/test_shared_core_math.py` 不依赖模型、LoRA、gate 或 GPU，但需要 PyTorch。应在论文冻结的 Linux/PyTorch 环境或后续 CI 中运行，并保存测试输出。

本次校验只能证明代码包结构、文本格式和完整性清单正常，不能替代模型资产齐备后的 GPU 端到端复现。当前发布范围是核心算法代码与主要实验入口，不包含完整 GPU 原始运行档案。
