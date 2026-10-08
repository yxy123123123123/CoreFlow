from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch


def p(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def tensor_from_output(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = tensor_from_output(item)
            if found is not None:
                return found
    return None


def entropy(prob: torch.Tensor) -> float:
    p = prob.clamp_min(1e-8)
    return float((-p * p.log()).sum().item())


def run_method(method: str, rows: list[dict], prompt_template: str, model_path: Path, source: Path, formal_pkg: Path, formal_work: Path):
    sys.path.insert(0, str(formal_pkg))
    from coreflow.loraflow import load_loraflow_model
    from coreflow.runtime import install_coreflow

    order = ["zh", "ru", "es", "math", "code"]
    gate = formal_work / "m2_assets" / "gates" / "code_k5_code_seed41.pt"
    bank = formal_work / "m2_assets" / "core_banks" / "k5_code" / f"q{method[4:]}_bf16"
    tokenizer, model, _ = load_loraflow_model(
        model_path,
        source / "official_assets",
        source / "vendor" / "LoRAFlow",
        adapter_order=order,
        gate_path=gate,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if method != "full":
        install_coreflow(model, bank, uniform=False, release_sources=True, expected_expert_order=order)
    layer_hooks = []
    gate_hooks = []
    layer_values: dict[int, list[torch.Tensor]] = {}
    gate_values: dict[int, list[torch.Tensor]] = {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
        if match:
            idx = int(match.group(1))
            def layer_hook(_module, _inputs, output, idx=idx):
                tensor = tensor_from_output(output)
                if tensor is not None:
                    layer_values.setdefault(idx, []).append(tensor.detach())
            layer_hooks.append(module.register_forward_hook(layer_hook))
        if "lora_fusion_gate" in name:
            def gate_hook(_module, _inputs, output, idx=len(gate_hooks)):
                tensor = tensor_from_output(output)
                if tensor is not None:
                    gate_values.setdefault(idx, []).append(tensor.detach())
            gate_hooks.append(module.register_forward_hook(gate_hook))
    if not layer_hooks:
        raise RuntimeError("No transformer layer hooks were found")
    records = []
    try:
        for index, row in enumerate(rows):
            prompt = prompt_template.format(problem=row["problem"], entry_point=row.get("entry_point", ""))
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=True).to("cuda")
            layer_values.clear(); gate_values.clear()
            with torch.inference_mode():
                output = model(**enc, use_cache=False, return_dict=True)
            hidden = {}
            for layer, values in layer_values.items():
                if values:
                    hidden[layer] = values[-1].float().mean(dim=1).squeeze(0).cpu()
            gates = {}
            for layer, values in gate_values.items():
                if values:
                    tensor = values[-1]
                    gates[layer] = tensor.float().reshape(-1, tensor.shape[-1])[-1].cpu()
            records.append({
                "task_id": row["task_id"],
                "hidden": hidden,
                "gate": gates,
                "logits": output.logits[0, -1].float().cpu(),
            })
            if index == 0 or (index + 1) % 8 == 0 or index + 1 == len(rows):
                print(f"[DRIFT] method={method} rows={index + 1}/{len(rows)}", flush=True)
    finally:
        for hook in layer_hooks + gate_hooks:
            hook.remove()
        del model
        torch.cuda.empty_cache()
    return records


def compare(reference, candidate):
    layer_stats = {}
    gate_stats = {}
    kl_values = []
    topk_values = []
    for ref, cand in zip(reference, candidate):
        for layer in sorted(set(ref["hidden"]) & set(cand["hidden"])):
            a, b = ref["hidden"][layer], cand["hidden"][layer]
            rel = float(torch.linalg.vector_norm(b - a).item() / max(torch.linalg.vector_norm(a).item(), 1e-8))
            cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
            layer_stats.setdefault(str(layer), {"relative_l2": [], "cosine": []})
            layer_stats[str(layer)]["relative_l2"].append(rel)
            layer_stats[str(layer)]["cosine"].append(cos)
        for layer in sorted(set(ref["gate"]) & set(cand["gate"])):
            a, b = ref["gate"][layer], cand["gate"][layer]
            pa, pb = torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)
            gate_stats.setdefault(str(layer), {"logit_l2": [], "top1_agreement": [], "reference_entropy": [], "candidate_entropy": []})
            gate_stats[str(layer)]["logit_l2"].append(float(torch.linalg.vector_norm(b - a).item()))
            gate_stats[str(layer)]["top1_agreement"].append(float(pa.argmax().item() == pb.argmax().item()))
            gate_stats[str(layer)]["reference_entropy"].append(entropy(pa))
            gate_stats[str(layer)]["candidate_entropy"].append(entropy(pb))
        p = torch.softmax(ref["logits"], dim=-1)
        q = torch.softmax(cand["logits"], dim=-1)
        kl_values.append(float(torch.sum(p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).item()))
        topk_values.append(float(torch.equal(ref["logits"].topk(5).indices, cand["logits"].topk(5).indices)))
    for values in layer_stats.values():
        values["relative_l2_mean"] = sum(values["relative_l2"]) / len(values["relative_l2"])
        values["cosine_mean"] = sum(values["cosine"]) / len(values["cosine"])
    for values in gate_stats.values():
        for key in ("logit_l2", "top1_agreement", "reference_entropy", "candidate_entropy"):
            values[key + "_mean"] = sum(values[key]) / len(values[key])
    return {
        "rows": len(reference),
        "hidden_by_layer": layer_stats,
        "gate_by_layer": gate_stats,
        "final_logits_kl_mean": sum(kl_values) / len(kl_values),
        "final_top5_exact_agreement": sum(topk_values) / len(topk_values),
    }


def main() -> None:
    work = p("WORK_ROOT", "/root/autodl-tmp/coreflow_reviewer_required_v2_workspace")
    source = p("SOURCE_WORK_ROOT", "/root/autodl-tmp/coreflow_m0")
    model = p("MODEL", "/root/autodl-tmp/Model")
    formal_pkg = p("FORMAL_V2_PACKAGE_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_upload")
    formal_work = p("FORMAL_V2_WORK_ROOT", "/root/autodl-tmp/coreflow_formal_v2_primary_workspace")
    report = work / "reports" / "coreflow-reviewer-required-v2"
    if not (report / "PROTOCOL_SEAL.json").exists():
        raise RuntimeError("Run seal first")
    rows = [json.loads(line) for line in (work / "data" / "llama_drift32.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    template = (formal_pkg / "data" / "prompt_code_mbppplus.txt").read_text(encoding="utf-8")
    full = run_method("full", rows, template, model, source, formal_pkg, formal_work)
    outputs = {}
    for q in (185, 224):
        candidate = run_method(f"q{q}", rows, template, model, source, formal_pkg, formal_work)
        outputs[f"q{q}_vs_full"] = compare(full, candidate)
    out = report / "drift_diagnostics.json"
    out.write_text(json.dumps({"status": "ANALYZED", "protocol_id": "coreflow-reviewer-minimal-phasea-v1", "selection": "frozen drift32", "comparisons": outputs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "stage": "drift", "rows": len(rows), "comparisons": list(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
