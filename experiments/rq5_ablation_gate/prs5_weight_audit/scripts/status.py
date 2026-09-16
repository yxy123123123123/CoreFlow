#!/usr/bin/env python3
from common import paths


def main():
    p = paths()
    print("=== PRS-5 reports ===")
    for rel in ("preflight.json", "PRS5_SEAL.json", "weight_audit_complete.json", "final/prs5_summary.json", "final/prs5_report.md"):
        path = p["reports"] / rel
        print(f"{rel}: {'PRESENT' if path.is_file() else 'MISSING'}")
    output = p["results"] / "weight_audit" / "weight_audit.json"
    partial = output.parent.with_name(output.parent.name + ".partial")
    print("\n=== weight audit ===")
    print(f"final: {'PRESENT' if output.is_file() else 'MISSING'}")
    print(f"partial: {'PRESENT' if partial.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
