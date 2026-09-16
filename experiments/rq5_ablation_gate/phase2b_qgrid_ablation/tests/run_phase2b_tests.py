#!/usr/bin/env python3
"""Phase-2B protocol tests."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

def test_protocol_contract():
    cfg = json.loads((ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == "coreflow-applsci-phase2b-final-v1"
    assert cfg["k_formal"] == 5
    assert cfg["expert_order"] == ["zh", "ru", "es", "math", "code"]
    assert set(cfg["b1_qgrid"]["q_values"]) == {128, 160, 185, 224, 256}

def test_s2_core_slower_than_full():
    csv_path = ROOT.parent / "phase2a_corrected_s2.csv"
    if not csv_path.exists():
        print("SKIP: corrected_s2.csv not in parent dir")
        return
    import csv
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            core_over_full = float(row["core_over_full_time_ratio"])
            assert core_over_full > 1.0, f"core_over_full must be >1.0 (Core slower), got {core_over_full} for K={row['k']}"

def test_module_cost():
    cfg = json.loads((ROOT / "config" / "protocol.json").read_text(encoding="utf-8"))
    assert sum(m["d_in"] + m["d_out"] for m in cfg["target_modules"]) > 0

def test_python_syntax():
    import ast
    for path in sorted(ROOT.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

def main():
    tests = [test_protocol_contract, test_s2_core_slower_than_full, test_module_cost, test_python_syntax]
    for t in tests: t()
    print(json.dumps({"status": "PASS", "tests": [t.__name__ for t in tests]}, ensure_ascii=False))

if __name__ == "__main__":
    main()