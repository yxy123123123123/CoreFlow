from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch


@dataclass(frozen=True)
class AdapterModule:
    name: str
    a: torch.Tensor
    b: torch.Tensor
    scaling: float

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.a.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.b.shape[0])


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: str | Path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def canonical_module_name(raw: str) -> str:
    name = raw
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    marker = "model.layers."
    if marker in name:
        name = name[name.index(marker) :]
    return name


def _load_torch_state(path: Path) -> Mapping[str, torch.Tensor]:
    kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(path, weights_only=True, **kwargs)
    except TypeError:
        state = torch.load(path, **kwargs)
    if not isinstance(state, Mapping):
        raise TypeError(f"Adapter checkpoint is not a state dict: {path}")
    return state


def load_adapter(adapter_dir: str | Path) -> tuple[dict, Dict[str, AdapterModule]]:
    root = Path(adapter_dir)
    config = load_json(root / "adapter_config.json")
    checkpoint = root / "adapter_model.safetensors"
    if checkpoint.exists():
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        checkpoint = root / "adapter_model.bin"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No adapter_model.bin/safetensors under {root}")
        state = _load_torch_state(checkpoint)

    alpha = float(config["lora_alpha"])
    default_rank = int(config.get("r", config.get("lora_rank", 0)))
    if default_rank <= 0:
        raise ValueError(f"Missing positive r/lora_rank in {root / 'adapter_config.json'}")
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for raw_key, tensor in state.items():
        suffix = None
        for candidate in (
            ".lora_A.default.weight",
            ".lora_B.default.weight",
            ".lora_A.weight",
            ".lora_B.weight",
        ):
            if raw_key.endswith(candidate):
                suffix = candidate
                break
        if suffix is None:
            continue
        module = canonical_module_name(raw_key[: -len(suffix)])
        pairs.setdefault(module, {})["a" if "lora_A" in suffix else "b"] = tensor.detach().cpu()

    modules: Dict[str, AdapterModule] = {}
    for module, tensors in sorted(pairs.items()):
        if set(tensors) != {"a", "b"}:
            raise ValueError(f"Incomplete A/B pair for {module}: {sorted(tensors)}")
        a = tensors["a"]
        b = tensors["b"]
        rank = int(rank_pattern.get(module, rank_pattern.get(module.split(".")[-1], default_rank)))
        alpha_here = float(alpha_pattern.get(module, alpha_pattern.get(module.split(".")[-1], alpha)))
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] != rank:
            raise ValueError(f"Bad LoRA shapes/rank for {module}: A={tuple(a.shape)} B={tuple(b.shape)} r={rank}")
        modules[module] = AdapterModule(module, a.float(), b.float(), alpha_here / rank)
    if not modules:
        raise ValueError(f"No LoRA A/B tensors found in {checkpoint}")
    return config, modules


def adapter_checkpoint_path(adapter_dir: str | Path) -> Path:
    root = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(root)


def module_signature(modules: Mapping[str, AdapterModule]) -> list[dict]:
    return [
        {
            "name": name,
            "rank": module.rank,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "scaling": module.scaling,
        }
        for name, module in sorted(modules.items())
    ]


def ensure_same_modules(groups: Iterable[Mapping[str, AdapterModule]]) -> list[str]:
    sets = [set(group) for group in groups]
    if not sets:
        raise ValueError("No adapter groups")
    if any(current != sets[0] for current in sets[1:]):
        details = [sorted(current.symmetric_difference(sets[0]))[:20] for current in sets[1:]]
        raise ValueError(f"Source adapters target different modules: {details}")
    names = sorted(sets[0])
    for name in names:
        shapes = [(group[name].in_features, group[name].out_features) for group in groups]
        if len(set(shapes)) != 1:
            raise ValueError(f"Dimension mismatch at {name}: {shapes}")
    return names
