# GitHub 发布前检查清单

## 必须完成

- [x] 作者决定当前不授予开源许可证，并以 `LICENSE_NOTICE.md` 明确保留全部权利。
- [x] 增加 `THIRD_PARTY_NOTICES.md`，说明随仓库分发的第三方文件及未重新分发的外部软件与研究资产。
- [ ] 确认所有公开模型、LoRA、gate 和数据链接仍可访问且允许使用。
- [ ] 用准备公开的仓库从干净环境至少完整重跑一次 RQ1 合成测试和一个 GPU smoke case。
- [x] 核对 README 中的模型名称、资产顺序、q、K、seed、哈希与当前 PeerJ 稿件一致。

## 安全与体积

- [x] 运行 `python tools/verify_package.py`。
- [x] 确认没有 token、cookie、邮箱密码、云平台凭据和本机私有绝对路径。
- [x] 确认没有 `.pt/.bin/.safetensors/.pth`、压缩包、trace、数据库或模型缓存。
- [x] GitHub 单文件小于 100 MiB；整个仓库约 3.15 MiB。
- [x] 未上传第三方 benchmark 题目、测试答案或生成文本；仅发布任务 ID、哈希和二元结果。

## 推荐完成

- [x] RQ1 原始返回包无法恢复，已发布冻结汇总记录并明确其不是原始日志；W0--W6 的 pilot 来源已在实验映射中标明。
- [ ] 为每个正式实验增加独立的 `README.md`，把历史 AutoDL 绝对路径改写为变量说明。
- [x] 增加 GitHub Actions，运行静态检查与公开证据结构检查。
- [x] 增加 `CITATION.cff` 与引用说明，并与六位作者顺序保持一致。
- [x] 发布固定版本标签 `v0.3.0-peerj-submission`；最终 commit SHA 在推送后记录到论文 Data Availability Statement。
