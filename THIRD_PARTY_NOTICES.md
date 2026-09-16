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

The project depends on or interoperates with external projects including
PyTorch, Transformers 4.36.2, the PEFT fork distributed with LoRA-Flow
(reported as 0.7.2.dev0 in the frozen runtime), Safetensors, NumPy, Flask,
EvalPlus/MBPP+, and ClassEval. Except for the source snapshot identified
above, these projects and datasets are not incorporated into this repository.
Their names are provided for attribution and reproducibility and do not imply
endorsement. Each remains governed by its own upstream license and terms.
