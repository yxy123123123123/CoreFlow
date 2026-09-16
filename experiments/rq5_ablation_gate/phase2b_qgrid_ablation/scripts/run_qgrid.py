#!/usr/bin/env python3
"""Phase-2B B1: q-grid quality evaluation runner.
Iterates over (task=q{code,math}, q=q{q_values}, seed=q{gate_seeds}) plus Full reference,
starts serve_benchmark_v2, generates/evaluates dev answers, and saves results."""
from __future__ import annotations
import json, sys, argparse, subprocess, time, urllib.request, hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import load_json, write_json, sha256_file
from coreflow.data_audit import load_jsonl

def request_json(url, payload=None, timeout=900.0):
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def wait_health(url, process, timeout=900.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None: raise RuntimeError(f"Server exited code={process.returncode}")
        try: return request_json(url, timeout=3)
        except Exception: time.sleep(2)
    raise TimeoutError("Server not healthy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-assets", required=True)
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=5601)
    args = parser.parse_args()
    cfg = load_json(ROOT / "config" / "protocol.json")
    dev = Path(args.dev_data)
    banks = Path(args.banks)
    out = Path(args.output_root)

    q_values = cfg["b1_qgrid"]["q_values"]
    gate_seeds = cfg["b1_qgrid"]["gate_seeds"]
    order = cfg["expert_order"]
    runtime_python = Path(args.work_root) / "runtime_env" / "bin" / "python"
    if not runtime_python.exists(): runtime_python = Path(sys.executable)
    serve_script = ROOT / "scripts" / "serve_benchmark_v2.py"

    methods = []
    # Full reference per gate seed
    for seed in gate_seeds:
        gate_file = str(Path(args.source_assets) / "official_assets" / "Gates" / f"code_k5_code_seed{seed}.pt")
        methods.append({"label": f"full_seed{seed}", "kind": "full", "gate": gate_file, "seed": seed, "q": None})
    # CoreFlow per q per seed
    for q_val in q_values:
        bank_dir = banks / f"q{q_val}_bf16"
        for seed in gate_seeds:
            gate_file = str(Path(args.source_assets) / "official_assets" / "Gates" / f"code_k5_code_seed{seed}.pt")
            methods.append({"label": f"core_q{q_val}_seed{seed}", "kind": "core", "gate": gate_file, "q": q_val, "bank": str(bank_dir), "seed": seed})

    for task_name, dev_file, max_tokens in [("code", "code_dev128.jsonl", cfg["b1_qgrid"]["generation"]["max_new_tokens_code"]),
                                               ("math", "math_dev128.jsonl", cfg["b1_qgrid"]["generation"]["max_new_tokens_math"])]:
        rows = load_jsonl(dev / dev_file)
        for method in methods:
            label = f"{task_name}_{method['label']}"
            final = out / label
            if (final / "COMPLETE.json").exists():
                print(f"[SKIP] {label}")
                continue
            if final.exists():
                raise RuntimeError(f"Retained partial/final output exists for {final}")
            partial = final.with_name(final.name + ".partial")
            partial.mkdir(parents=True)

            cmd = [str(runtime_python), str(serve_script),
                   "--model", args.model, "--source-assets", args.source_assets,
                   "--vendor-root", args.vendor_root,
                   "--adapter-order", ",".join(order), "--gate", method["gate"],
                   "--kind", method["kind"],
                   "--max-input-tokens", str(cfg["b1_qgrid"]["generation"]["max_input_tokens"]),
                   "--max-new-tokens", str(max_tokens),
                   "--port", str(args.port), "--do-sample", "false", "--temperature", "1.0", "--top-p", "1.0"]
            if method["bank"]: cmd.extend(["--bank", method["bank"]])
            log = (partial / "server.log").open("w", encoding="utf-8")
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            started = time.time()
            generated = []; evaluated = []
            try:
                wait_health(f"http://127.0.0.1:{args.port}/health", server)
                genf = (partial / "generations.jsonl").open("w", encoding="utf-8", newline="\n")
                evf = (partial / "executor_results.jsonl").open("w", encoding="utf-8", newline="\n")
                for idx, row in enumerate(rows):
                    prompt = str(row["problem"])
                    resp = request_json(f"http://127.0.0.1:{args.port}/infer", {"prompt": prompt, "seed": method["seed"] * 100000 + idx})
                    gen = {"task_id": row["task_id"], "order_index": idx,
                           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                           "raw_output": resp["answer"], "timing": resp["timing"]}
                    genf.write(json.dumps(gen, ensure_ascii=False, sort_keys=True) + "\n"); genf.flush()
                    generated.append(gen)
                    # evaluate
                    if task_name == "math":
                        from coreflow.m2_eval import evaluate_math_answer
                        ev = {"task_id": row["task_id"], "order_index": idx, "timing": gen["timing"],
                              "raw_output_sha256": hashlib.sha256(gen["raw_output"].encode()).hexdigest(),
                              "evaluation": evaluate_math_answer(gen["raw_output"], row.get("answer", ""))}
                    else:
                        from coreflow.m2_eval import evaluate_apps_program
                        ev = {"task_id": row["task_id"], "order_index": idx, "timing": gen["timing"],
                              "raw_output_sha256": hashlib.sha256(gen["raw_output"].encode()).hexdigest(),
                              "evaluation": evaluate_apps_program(gen["raw_output"], row.get("io_cases", []), cfg["b1_qgrid"]["evaluator"])}
                    evf.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n"); evf.flush()
                    evaluated.append(ev)
            finally:
                if server.poll() is None: server.terminate(); server.wait(timeout=30)
                log.close()
            n = len(evaluated)
            correct = sum(bool(e["evaluation"]["passed"]) for e in evaluated)
            summary = {"rows": n, "correct": correct, "pass_at_1": correct / n if n else 0,
                       "status_counts": dict(sorted(Counter(e["evaluation"]["status"] for e in evaluated).items()))}
            write_json(partial / "summary.json", summary)
            complete = {"status": "PASS", "method": label, "task": task_name, "q": method["q"],
                        "seed": method["seed"], "row_count": n, "elapsed_seconds": time.time() - started}
            write_json(partial / "COMPLETE.json", complete)
            partial.rename(final)
            print(json.dumps({"status": "PASS", "method": label, "correct": correct, "rows": n}, ensure_ascii=False))

if __name__ == "__main__":
    main()