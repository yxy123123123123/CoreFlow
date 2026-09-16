# CoreFlow 独立确认实验包 v1（ClassEval method-level）

协议 ID：`coreflow-indep-confirm-classeval-v1`

本包用于在第二个未暴露代码基准上独立确认 CoreFlow q185 相对 Original LoRA-Flow 的质量非劣性（95% CI 下界 > -3 pp），并把 Per-Expert SVD（matched-q185）作为二级对照。数据集为 ClassEval 官方 method-level 拆分：

- qualification：25 类 / 98 methods
- formal：48 类 / 202 methods
- reserve：10 类 / 39 methods

## 冻结口径

- 基础模型、LoRA 专家池、gates、CoreFlow q185 bank 全部复用主实验资产，不重训、不重编译、不重选 q。
- Per-Expert SVD bank 直接复用主实验 `coreflow_formal_v2_primary_workspace/isvd_banks`，包内不重建。
- 生成：greedy、batch 1、max_input_tokens 2048、max_new_tokens_code 768、无自动重试。
- 统计：问题级聚类配对 bootstrap 20,000 次、95% CI、seed `20260806`；主判据下界 > -3 pp，另报 -1/-2/-3 pp 敏感性。
- qualification 门槛：n=97、k=8、p_null=0.03、p_target=0.15，通过后才打开 formal。

## 包结构

```text
config/indep_confirm_classeval_protocol.json   冻结协议
coreflow/classeval_eval.py                      ClassEval 组合式评测器
data/                                          prompt、拆分文件、manifest
provenance/                                     主实验资产锁 + ISVD 资产锁
raw_sources/ClassEval_data.json                 官方原始数据集（含 solution/test，仅评测用）
scripts/                                       控制、生成、评测、统计、打包
tests/run_tests.py                              本地冻结测试
vendor_canonical/modeling_llama.py              LoRAFlow 规范化模型文件
```

## 本地验证

```bash
python scripts/build_classeval_splits.py --tokenizer-json <tokenizer.json>
python tests/run_tests.py
python scripts/package_checksums.py --root . --write
```

AutoDL 上每次执行都会重新校验 `PACKAGE_MANIFEST.sha256`。
