# 第三方资产准备与哈希

本仓库不分发 Llama-2-7B、Qwen3-8B、LoRA、gate/CoMoL checkpoint、CoreFlow/ISVD bank 或基准数据。Llama 主实验使用下列 K=5 专家顺序：

```text
zh, ru, es, math, code
```

## 冻结资产哈希

| 资产 | SHA-256 |
|---|---|
| Llama-2-7B `config.json` | `9242e7db1bc2a17873e66084c3b1c6ed10883076e156b338fd6a7775748e2e3c` |
| zh LoRA checkpoint | `46b992598135f1f1351613d4fa50e6aad5650f24eb0ef7a80d233aceba1b83fb` |
| ru LoRA checkpoint | `b2e6becfbe82a70690d3632ff2028ba9c22a005658ffc04cb09e2202e1ce41c2` |
| es LoRA checkpoint | `00826c33ff5b61307cfcaa88ec424d263b52eb3bfb2df041698566090beae9da` |
| math LoRA checkpoint | `8b938cfe4941860068aebbfa0e11816c0733b3a40ff5862246f6c9440fb66e61` |
| code LoRA checkpoint | `929e484e57b64bb6676ac22850ad05a1f585b9952bb45f52c0463aa281c2b0c4` |
| gate-41 | `640e1a72f66d965731a2033c4326c634d49515e8fd8b74e0750d72f235e1d2c7` |
| gate-42 | `28e3c1dffc97107ddc140a98186d38163b1b2296f3581c9f00a15481ee2f1966` |
| gate-43 | `ee7bef0a196b2544598f65df88935dd7d333e6368d2d251b9aced96335a71636` |
| CoreFlow K5/q185 bank | `1a3eaf175981a2c914af8400490ee603308851fd5ca22279a6ac76c2d121eaf2` |
| CoreFlow K5/q185 config | `1e002abd6145aeee4fb6b5967aed1c638d117e7d9d8430af133d203969f80427` |
| Per-Expert SVD K5/matched-q185 bank | `a8b5a3bf6ba3265fd8c35ec26f100d541be6f27e5b2fb4aaf8cc37f673804433` |
| Per-Expert SVD K5 config | `3bb89fcb6ddbac70dc1c1527f564be264f9238caa5567d3a4a24a1a260b021a9` |

更完整的锁文件保存在各实验目录的 `provenance/` 或 `evidence/` 中。

Qwen3-8B 路线对照使用固定 revision `b968826d9c46dd6066d109eabc6255188de91218`。其配置和 tokenizer 哈希见 `docs/manifests/QWEN3_ASSET_MANIFEST.json`；训练后的 Independent-Full、CoreFlow 和 CoMoL 权重不在本仓库分发。

## 数据

- MBPP+：使用 EvalPlus 官方 test split，经固定 SHA-256 顺序确定 dev128/formal250。
- ClassEval：使用官方 method-level 数据，经客观排除规则形成 qualification98/formal202/reserve39。
- APPS-derived：从固定 `codeparrot/apps` revision 的 test split 按公开规则确定 257 题开发集，仅发布任务 ID、源索引和哈希。
- 数据文件不随仓库发布；拆分 manifest、prompt 和协议保留在对应实验目录。

具体来源、选择规则、哈希和 exact-overlap 审计定义见 `docs/DATA_PROVENANCE.md`。

请在生成任何输出前先验证来源许可证、任务 ID、文件哈希和拆分 manifest。
