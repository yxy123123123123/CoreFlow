# CoreFlow M3-mini K-scaling mechanism experiment

This package runs a minimal mechanism-only K-scaling audit after formal-v2 primary.

It does not alter formal-v2 primary results and does not run endpoint generation.

Main question:

> As K grows from 3/5 to 8, does CoreFlow's shared basis preserve adapter/fused-output structure better than matched-budget Independent-SVD while keeping better adapter-kernel efficiency?

Pools:

- `k3_strict`: zh, math, code
- `k5_formal`: zh, ru, es, math, code
- `k8_strict`: zh, ru, es, math, code, magicoder, openwebmath, gsm8k_loftq

Methods:

- Full source LoRA factors
- CoreFlow q185
- Independent-SVD matched to CoreFlow q185 resident adapter parameter budget

Outputs:

- asset audit
- CoreFlow/ISVD bank lock
- per-expert sampled output reconstruction error
- synthetic fused-output error under uniform/one-hot/top2/Dirichlet gates
- adapter-only CUDA microbenchmark
- Go/No-Go decision for whether a larger K-scaling study is worth running

The three new LoRA assets are bundled under `lora_assets/m3_k8_strict`, so AutoDL does not need Hugging Face access.
