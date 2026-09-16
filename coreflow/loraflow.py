from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import torch


CANONICAL_LLAMA_SHA256 = "868780e64eabdfd2f563373cbebe44d5e4b022b4b89817688f7a77eb107b0681"


def activate_official_pythonpath(official_root: str | Path) -> None:
    root = Path(official_root)
    paths = [
        root / "UltraEval" / "lora-fusion" / "transformers" / "src",
        root / "UltraEval" / "lora-fusion" / "peft-group_lora" / "src",
        root / "UltraEval" / "human-eval-master",
        root / "UltraEval",
    ]
    for path in reversed(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))


ADAPTER_PATHS = {
    "zh": "zh_lora",
    "ru": "ru_lora",
    "es": "es_lora",
    "math": "math_lora",
    "code": "code_lora",
}


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_unique_tensors_(model, device: str | torch.device, dtype: torch.dtype | None = None) -> dict:
    """Materialize each registered tensor once without ``Module.to``'s peak.

    The five-expert PEFT wrapper contains a large registered tensor graph.  On a
    24 GiB card, recursively applying ``Module.to(cuda)`` can transiently retain
    both old and new storages and OOM even though the final model is only about
    15 GiB.  Moving the unique parameters and buffers in place matches the
    deployment object graph while keeping peak allocation close to final size.
    """
    target = torch.device(device)
    seen: set[int] = set()
    parameter_count = 0
    buffer_count = 0
    logical_bytes = 0

    def move_tensor_(tensor: torch.Tensor) -> bool:
        nonlocal logical_bytes
        identity = id(tensor)
        if identity in seen:
            return False
        seen.add(identity)
        target_dtype = dtype if dtype is not None and tensor.is_floating_point() else tensor.dtype
        tensor.data = tensor.data.to(device=target, dtype=target_dtype)
        logical_bytes += tensor.numel() * tensor.element_size()
        return True

    with torch.no_grad():
        for parameter in model.parameters():
            if move_tensor_(parameter):
                parameter_count += 1
            if parameter.grad is not None:
                grad_dtype = dtype if dtype is not None and parameter.grad.is_floating_point() else parameter.grad.dtype
                parameter.grad.data = parameter.grad.data.to(device=target, dtype=grad_dtype)
        for buffer in model.buffers():
            if move_tensor_(buffer):
                buffer_count += 1
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    return {
        "strategy": "unique_registered_tensor_in_place_move",
        "device": str(target),
        "floating_dtype": str(dtype) if dtype is not None else "preserve",
        "unique_parameters": parameter_count,
        "unique_buffers": buffer_count,
        "logical_bytes_after_move": int(logical_bytes),
    }


def _canonical_llama_path() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor_canonical" / "modeling_llama.py"


def patch_lora_num(official_root: str | Path, k: int) -> dict:
    """Restore the vendored modeling_llama.py from the packaged canonical copy and
    apply the lora_num / weight_bias patches in one deterministic step.

    The previous implementation mutated the vendored file in place with regex
    substitutions on every model load. That file drifted into a corrupted state
    (gate parameters initialized with garbage values such as absmax ~1e22),
    which poisoned every from-scratch gate training run. Patching from the
    hash-pinned canonical copy makes every load deterministic; the vendored file
    is rewritten atomically and only when its content differs.
    """
    canonical = _canonical_llama_path()
    canonical_bytes = canonical.read_bytes()
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    if digest != CANONICAL_LLAMA_SHA256:
        raise ValueError(f"Canonical modeling_llama.py hash mismatch: {digest}")
    text = canonical_bytes.decode("utf-8")
    text, n_lora = re.subn(r"lora_num\s*=\s*\d+", f"lora_num = {int(k)}", text, count=1)
    text, n_bias = re.subn(
        r"self\.weight_bias\s*=\s*nn\.Parameter\(torch\.tensor\(\[\[-0\.5,0\.5\]\]\)\.to\(self\.lora_fusion_gate\.weight\.dtype\)\)",
        "self.weight_bias = nn.Parameter(torch.zeros(1, lora_num, dtype=self.lora_fusion_gate.weight.dtype))",
        text,
        count=1,
    )
    if n_lora != 1 or n_bias != 1:
        raise ValueError(f"Llama gate patch did not apply exactly once: lora_num={n_lora}, weight_bias={n_bias}")
    target = (
        Path(official_root)
        / "UltraEval"
        / "lora-fusion"
        / "transformers"
        / "src"
        / "transformers"
        / "models"
        / "llama"
        / "modeling_llama.py"
    )
    current = target.read_text(encoding="utf-8") if target.exists() else None
    rewrote = False
    if current != text:
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        rewrote = True
    return {
        "patched_file": str(target),
        "lora_num": int(k),
        "canonical_sha256": digest,
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "rewrite_performed": rewrote,
    }


def transplant_gate(model, gate_path: str | Path, expected_k: int | None = None) -> dict:
    state = _torch_load(Path(gate_path))
    gate = {key: value for key, value in state.items() if "lora_fusion_gate" in key}
    bias = {key: value for key, value in state.items() if "weight_bias" in key}
    copied = 0
    max_error = 0.0
    layer_pattern = re.compile(r"(?:model|encoder)\.layers\.(\d+)\.")
    for name, parameter in model.named_parameters():
        if "lora_fusion_gate" not in name and "weight_bias" not in name:
            continue
        match = layer_pattern.search(name)
        if not match:
            raise ValueError(f"Cannot infer decoder layer from gate parameter: {name}")
        layer = int(match.group(1))
        source_pool = gate if "lora_fusion_gate" in name else bias
        source_suffix = "lora_fusion_gate.weight" if "lora_fusion_gate" in name else "weight_bias"
        layer_token = f"layers.{layer}."

        candidates = [
            key for key in source_pool
            if layer_token in key and key.endswith(source_suffix)
        ]
        if not candidates:
            expected = (
                f"encoder.layers.{layer}.lora_fusion_gate.weight"
                if "lora_fusion_gate" in name
                else f"encoder.layers.{layer}.weight_bias"
            )
            candidates = [expected] if expected in source_pool else []

        if not candidates:
            sample_keys = sorted(source_pool)[:8]
            raise KeyError(
                f"Cannot find gate tensor for model parameter {name}; "
                f"layer={layer}, suffix={source_suffix}, sample_gate_keys={sample_keys}"
            )

        source_key = sorted(candidates)[0]
        source = source_pool[source_key]
        if expected_k is not None and int(source.shape[0] if "lora_fusion_gate" in name else source.shape[-1]) != int(expected_k):
            raise ValueError(f"Gate tensor K mismatch for {source_key}: shape={tuple(source.shape)}, expected_k={expected_k}")
        parameter.data.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
        expected_after_cast = source.to(dtype=parameter.dtype).float()
        error = float((parameter.detach().float().cpu() - expected_after_cast).abs().max())
        max_error = max(max_error, error)
        copied += 1
    if copied != 64:
        raise ValueError(f"Expected 64 gate tensors in model, copied={copied}")
    return {"copied_tensors": copied, "max_abs_error_after_dtype_cast": max_error}


def load_loraflow_model(
    model_path: str | Path,
    asset_root: str | Path,
    official_root: str | Path,
    task: str | None = None,
    adapter_order: list[str] | tuple[str, ...] | None = None,
    gate_path: str | Path | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if adapter_order is None:
        if task not in {"zh_math", "zh_code"}:
            raise ValueError(task)
        adapter_order = ("zh", "math" if task == "zh_math" else "code")
    adapter_order = tuple(adapter_order)
    if len(adapter_order) < 1:
        raise ValueError(f"Need at least one adapter, got {adapter_order}")
    unknown = sorted(set(adapter_order) - set(ADAPTER_PATHS))
    if unknown:
        raise ValueError(f"Unknown adapter names: {unknown}")
    patch_report = patch_lora_num(official_root, len(adapter_order))
    activate_official_pythonpath(official_root)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    temperature = 1.1 if "math" in adapter_order and len(adapter_order) > 1 else 1.0
    model_path = Path(model_path)
    asset_root = Path(asset_root)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Assemble the base model and every adapter on CPU first.  Moving the base
    # to CUDA before PEFT loads all five adapters creates a large transient
    # allocation on 24 GiB cards (and can even move an unexpectedly fp32 base).
    # The final runtime model is still converted to the requested dtype and
    # device in one deterministic step below.
    base = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(dtype=dtype)
    for layer in base.model.layers:
        layer.temperature = temperature
    model = PeftModel.from_pretrained(
        base,
        model_id=str(asset_root / "LoRAs" / ADAPTER_PATHS[adapter_order[0]]),
        adapter_name=adapter_order[0],
        is_trainable=False,
    )
    for adapter in adapter_order[1:]:
        model.load_adapter(
            model_id=str(asset_root / "LoRAs" / ADAPTER_PATHS[adapter]),
            adapter_name=adapter,
            is_trainable=False,
        )
    model.base_model.set_adapter(list(adapter_order))
    if gate_path is None and len(adapter_order) == 2 and task in {"zh_math", "zh_code"}:
        gate_path = asset_root / "Gates" / ("zh_math.pt" if task == "zh_math" else "zh_code.pt")
    gate_report = {"loaded_gate": None}
    if gate_path is not None:
        gate_report = {"loaded_gate": str(gate_path), **transplant_gate(model, gate_path, expected_k=len(adapter_order))}
    elif len(adapter_order) > 1:
        raise ValueError("A frozen gate is required for every multi-expert method")
    materialization_report = move_unique_tensors_(model, device=device, dtype=dtype)
    model.eval()
    return tokenizer, model, {
        "task": task,
        "adapter_order": list(adapter_order),
        "temperature": temperature,
        "materialization": materialization_report,
        **patch_report,
        **gate_report,
    }
