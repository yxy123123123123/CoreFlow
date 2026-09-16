"""Small CPU test for the CoreFlow shared-core algebra.

This test uses synthetic heterogeneous LoRA ranks and does not require model,
LoRA, gate, or benchmark assets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coreflow.decompose import build_module_bases, project_core  # noqa: E402
from coreflow.io import AdapterModule  # noqa: E402
from coreflow.runtime import coreflow_delta  # noqa: E402


def main() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    in_features, out_features = 17, 13
    ranks = (3, 5, 4)
    experts = []
    for index, rank in enumerate(ranks):
        a = torch.randn(rank, in_features, generator=generator)
        b = torch.randn(out_features, rank, generator=generator)
        experts.append(AdapterModule(f"expert-{index}", a, b, 0.5 + 0.25 * index))

    u, v, report = build_module_bases(experts, [1.0 / len(experts)] * len(experts), balanced=True)
    cores = torch.stack([project_core(u, v, expert) for expert in experts])

    x = torch.randn(7, in_features, generator=generator)
    weights = torch.softmax(torch.randn(7, len(experts), generator=generator), dim=-1)
    reference = torch.zeros(7, out_features)
    for expert_index, expert in enumerate(experts):
        delta = ((x @ expert.a.T) @ expert.b.T) * expert.scaling
        reference += weights[:, expert_index : expert_index + 1] * delta

    candidate = coreflow_delta(x, weights, u, v, cores)
    relative_error = torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(reference)
    assert report["q_upper"] == sum(ranks)
    assert float(relative_error) < 2e-5, relative_error
    print(f"PASS shared-core synthetic relative_error={float(relative_error):.3e}")


if __name__ == "__main__":
    main()

