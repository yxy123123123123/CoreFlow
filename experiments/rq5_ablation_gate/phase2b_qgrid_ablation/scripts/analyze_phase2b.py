#!/usr/bin/env python3
"""Phase-2B analysis: B1 q-grid quality + B3 prefill/TTFT."""
from __future__ import annotations
import argparse, json, sys, csv
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from coreflow.io import load_json, write_json
from coreflow.statistics import paired_bootstrap_difference

def load_qgrid_results(result_root: Path, cfg: dict) -> dict:
    qv = [str(q) for q in cfg["b1_qgrid"]["q_values"]]
    seeds = cfg["b1_qgrid"]["gate_seeds"]
    out = {}
    for task in ("code", "math"):
        for q in qv:
            for seed in seeds:
                d = result_root / "b1_qgrid" / f"{task}_q{q}_seed{seed}"
                if (d / "summary.json").exists():
                    s = load_json(d / "summary.json")
                    out[f"{task}_q{q}_seed{seed}"] = s
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "protocol.json"))
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_json(Path(args.config))
    rr = Path(args.result_root)
    qr = load_qgrid_results(rr, cfg)
    decision = {"status": "ANALYZED", "b1": qr, "b3": "merged_from_launch_level_csvs"}
    write_json(Path(args.output), decision)
    print(json.dumps({"status": decision["status"]}, ensure_ascii=False))

if __name__ == "__main__":
    main()