# GitHub 发布前检查清单

## 必须完成

- [x] 作者决定当前不授予开源许可证，并以 `LICENSE_NOTICE.md` 明确保留全部权利。
- [x] 增加 `THIRD_PARTY_NOTICES.md`，说明随仓库分发的第三方文件及未重新分发的外部软件与研究资产。
- [ ] 确认所有公开模型、LoRA、gate 和数据链接仍可访问且允许使用。
- [ ] 用准备公开的仓库从干净环境至少完整重跑一次 RQ1 合成测试和一个 GPU smoke case。
- [ ] 核对 README 中的模型名称、资产顺序、q、K、seed、哈希与最终论文完全一致。

## 安全与体积

- [ ] 运行 `python tools/verify_package.py`。
- [ ] 确认没有 token、cookie、邮箱密码、云平台凭据和绝对私人路径。
- [ ] 确认没有 `.pt/.bin/.safetensors/.pth`、压缩包、trace、数据库或模型缓存。
- [ ] GitHub 单文件小于 100 MiB；建议整个仓库保持轻量。
- [ ] 不上传第三方 benchmark 原始测试答案，除非许可证明确允许。

## 推荐完成

- [ ] 若以后扩展为完整实验归档，再补回 RQ1/PRS-2 原始日志，并明确 W0--W6 的 pilot 来源；这两项不阻止仅发布核心算法代码。
- [ ] 为每个正式实验增加独立的 `README.md`，把历史 AutoDL 绝对路径改写为变量说明。
- [ ] 增加 GitHub Actions，仅运行静态检查与无模型单元测试。
- [ ] 增加论文 BibTeX 与引用说明。
- [ ] 发布 release tag，并记录最终 commit SHA 到论文 Data Availability Statement。
