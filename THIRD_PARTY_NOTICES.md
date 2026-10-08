# Third-Party Notices

This repository does not redistribute the foundation model, LoRA expert
checkpoints, gate checkpoints, or benchmark datasets used in the paper.
Users must obtain those assets from their original sources and comply with
the corresponding licenses and terms.

## Redistributed source snapshot

`vendor_canonical/modeling_llama.py` contains code derived from the
Hugging Face Transformers LLaMA implementation. Its original copyright and
Apache License 2.0 header are retained in the file. The Apache License 2.0 is
available at <https://www.apache.org/licenses/LICENSE-2.0>.

## External software and research assets

The project depends on or interoperates with external projects and research
assets including PyTorch, Transformers, PEFT, Safetensors, NumPy, Flask,
LoRA-Flow, CoMoL, Llama-2-7B, Qwen3-8B, EvalPlus/MBPP+, ClassEval, and APPS.
The PEFT fork used by the frozen LoRA-Flow runtime reported version
`0.7.2.dev0`; the original Llama experiments used Transformers 4.36.2.

Except for the source snapshot identified above, these projects, models,
trained weights, and datasets are not incorporated into this repository.
Their names are provided for attribution and reproducibility and do not imply
endorsement. Each remains governed by its own upstream license, access rules,
and terms. Model and dataset fingerprints in `docs/` identify the authors'
frozen experimental inputs but do not grant redistribution or usage rights.
