# Required-v2 experiment matrix

| Stage | Model | Methods | Data | New GPU? | Output |
|---|---|---|---|---|---|
| Local audit | Llama/Qwen | contract, storage, lifecycle, theory, statistics | frozen manifests/results | No | audit JSON/MD |
| q224 quality | Llama-2-7B | Core q224 | MBPP+ 250×3; ClassEval 202×3 | Yes | q224 executor results |
| drift | Llama-2-7B | Full/q185/q224 | frozen drift32 | Yes | hidden/gate/logit diagnostics |
| group2 | Qwen3-8B | 8 independent LoRA, CoMoL, Core q185 | full math six datasets | Yes | repeat quality metrics |
| group3 | Qwen3-8B | 8 independent LoRA, CoMoL, Core q185 | full math six datasets | Yes | repeat quality metrics |
| service | Qwen3-8B | Full/Core q16/CoMoL | c1/c4 and 512/1024 | Yes, per hardware | latency/throughput/memory/power |

The existing Qwen group1 is read-only reuse. It is not overwritten by group2/group3.

## Statistical unit

The quality analysis treats task IDs as paired observations within each seed/gate block. Group2 and group3 are independent training repetitions. The primary q185 formal result remains the frozen reference; q224 is a sensitivity point.

## Service interpretation

`c4` sends four requests sequentially through one worker. It measures queued-load behavior and tail latency. It is not continuous batching and must not be described as such.
