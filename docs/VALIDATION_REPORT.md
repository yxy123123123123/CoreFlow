# 本地整理与校验报告

校验日期：2026-10-09

| 检查 | 结果 |
|---|---|
| Python 源码 AST 解析 | 通过，177 个 `.py` 文件 |
| JSON 结构与公开证据结构检查 | 通过；逐题结果 5,424 行、RTX 4080 SUPER 9 个配置且每配置 50 条请求记录、Qwen 生命周期 4 条 PASS 记录 |
| 13 个实验/归档子包完整性清单 | 全部重建并通过 |
| Reviewer-required v2 编译接口合同测试 | 通过，5 个拒绝用例和 1 个最小合法用例均符合预期 |
| 大文件、权重、压缩包、缓存与常见密钥模式扫描 | 通过 |
| 本机 Windows 绝对路径扫描 | 通过；公开文件中未保留作者工作区路径 |
| GitHub 单文件 100 MiB 限制 | 通过 |
| 作者元数据一致性 | 通过；`CITATION.cff` 已按论文顺序加入 Han Hong |
| 顶层完整性清单 | 已重建，349 个条目 |
| 整理后总体积 | 约 3.15 MiB |
| GitHub Actions 静态校验 | 已加入 `.github/workflows/validate.yml`，每次 push/PR 运行包范围与 Python 语法检查 |
| 合成共享核心 PyTorch 单元测试 | 当前 Windows 整理环境缺少 PyTorch，启动时报告 `ModuleNotFoundError: No module named 'torch'`，未执行 |

`tests/test_shared_core_math.py` 不依赖模型、LoRA、gate 或 GPU，但需要 PyTorch。应在论文冻结的 Linux/PyTorch 环境或后续 CI 中运行，并保存测试输出。

本次校验能证明代码包结构、文本格式、公开最小证据和完整性清单正常，不能替代模型资产齐备后的 GPU 端到端复现。MBPP+/ClassEval 的逐题二元结果、RTX 4080 SUPER 请求级数值记录和 Qwen group 2/3 生命周期计时已进入公开包；RQ1 原始返回包无法恢复，已用明确标注的归档汇总记录代替，不能表述为原始日志。
