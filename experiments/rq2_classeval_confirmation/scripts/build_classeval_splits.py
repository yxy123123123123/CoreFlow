#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Classes whose official canonical tests are not reproducibly runnable in the
# frozen evaluator environment: niche third-party packages, current-time
# dependence, machine-specific values, or official dataset defects.
EXCLUDED_CLASSES = {
    "ClassEval_0": "current-time dependent (date.today/datetime.now/time.time)",
    "ClassEval_17": "current-time dependent (datetime.now)",
    "ClassEval_20": "current-time dependent (datetime.now)",
    "ClassEval_25": "official defect: load_cookies expects cookies.json not created by tests",
    "ClassEval_28": "external dependency pandas",
    "ClassEval_31": "official float-assertion inconsistency in correlation_coefficient",
    "ClassEval_34": "external dependency docx",
    "ClassEval_36": "current-time dependent (datetime.now)",
    "ClassEval_38": "external dependency openpyxl",
    "ClassEval_44": "external dependencies bs4+gensim",
    "ClassEval_45": "external dependency PIL",
    "ClassEval_48": "machine-specific hostname assertion",
    "ClassEval_49": "official defect: matches_requirements helper missing from dataset",
    "ClassEval_52": "external dependency nltk",
    "ClassEval_69": "external dependency PyPDF2 + TestPDFHandler missing",
    "ClassEval_87": "current-time dependent (datetime.now/time.time)",
    "ClassEval_93": "external dependencies gensim+numpy",
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def add_desc_to_init(desc: str, class_init: str) -> str:
    lines = class_init.split("\n")
    lines[0] += " \n" + desc
    return "\n".join(lines)


def get_method_signature(method_description: str, method_name: str) -> str:
    prefix = f"def {method_name}("
    for segment in method_description.split("):"):
        if prefix in segment:
            return "    " + segment + "):"
    return ""


def build_skeleton(cls: dict, target: dict) -> str:
    imports = "\n".join(cls["import_statement"])
    class_init = add_desc_to_init(cls["class_description"], cls["class_constructor"])
    text = imports + "\n" + class_init
    for method in cls["methods_info"]:
        if method["method_name"] == target["method_name"]:
            continue
        signature = get_method_signature(method["method_description"], method["method_name"])
        if signature:
            text += signature + "\n        pass\n\n"
    return text


def build_problem(cls: dict, target: dict) -> str:
    skeleton = build_skeleton(cls, target)
    problem = (
        f'please complete {target["method_name"]} method in the following class {cls["class_name"]}\n\n'
        + skeleton
        + "\n\n    "
        + target["method_description"]
    )
    return problem


def make_row(cls: dict, method: dict, split: str, method_order: list[str]) -> dict:
    method_name = str(method["method_name"])
    problem = build_problem(cls, method)
    return {
        "task_id": f'{cls["task_id"]}::{method_name}',
        "class_id": cls["task_id"],
        "class_name": cls["class_name"],
        "split": split,
        "entry_point": method_name,
        "method_index": method_order.index(method_name),
        "method_order": method_order,
        "problem": problem,
        "prompt_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
        "evaluator": "classeval_method",
        "import_statement": list(cls["import_statement"]),
        "class_constructor": cls["class_constructor"],
        "method_description": method["method_description"],
        "test_class": method["test_class"],
        "test_code": method["test_code"],
        "dependencies": method["dependencies"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-id", default="coreflow-indep-confirm-classeval-v1")
    parser.add_argument("--raw-source", default=str(ROOT / "raw_sources" / "ClassEval_data.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "data"))
    parser.add_argument("--prompt-template", default=str(ROOT / "data" / "prompt_classeval.txt"))
    parser.add_argument("--qual-methods-min", type=int, default=96)
    parser.add_argument("--formal-methods-min", type=int, default=200)
    parser.add_argument("--tokenizer-json")
    parser.add_argument(
        "--exclude-classes",
        default=",".join(sorted(EXCLUDED_CLASSES)),
        help="Comma-separated ClassEval class IDs to exclude from splits",
    )
    args = parser.parse_args()

    source = Path(args.raw_source).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing ClassEval raw source: {source}")
    classes = json.loads(source.read_text(encoding="utf-8"))
    all_ids = {cls["task_id"] for cls in classes}
    exclude_ids = {item.strip() for item in args.exclude_classes.split(",") if item.strip()}
    unknown = sorted(exclude_ids - all_ids)
    if unknown:
        raise ValueError(f"Unknown excluded class IDs: {unknown}")
    excluded_reasons = {cid: EXCLUDED_CLASSES.get(cid, "explicit CLI exclusion") for cid in sorted(exclude_ids)}
    classes = [cls for cls in classes if cls["task_id"] not in exclude_ids]

    ordered = sorted(
        classes,
        key=lambda cls: hashlib.sha256(f'{args.protocol_id}|class|{cls["task_id"]}'.encode("utf-8")).hexdigest(),
    )
    qual_classes: list[dict] = []
    formal_classes: list[dict] = []
    reserve_classes: list[dict] = []
    qual_count = 0
    for cls in ordered:
        if qual_count < args.qual_methods_min:
            qual_classes.append(cls)
            qual_count += len(cls["methods_info"])
        elif sum(len(item["methods_info"]) for item in formal_classes) < args.formal_methods_min:
            formal_classes.append(cls)
        else:
            reserve_classes.append(cls)

    def rows_for(items: list[dict], split: str) -> list[dict]:
        rows: list[dict] = []
        for cls in items:
            order = [str(method["method_name"]) for method in cls["methods_info"]]
            for method in cls["methods_info"]:
                rows.append(make_row(cls, method, split, order))
        return rows

    qual_rows = rows_for(qual_classes, "qualification")
    formal_rows = rows_for(formal_classes, "formal")
    reserve_rows = rows_for(reserve_classes, "reserve")
    for row in qual_rows + formal_rows + reserve_rows:
        if "solution_code" in row or "solutions" in row or "hidden_test" in row:
            raise ValueError(f"Forbidden label leaked into row: {row['task_id']}")

    qual_file = output_dir / "classeval_qualification.jsonl"
    formal_file = output_dir / "classeval_formal.jsonl"
    reserve_file = output_dir / "classeval_reserve.jsonl"
    hashes = {
        "qualification": write_jsonl(qual_file, qual_rows),
        "formal": write_jsonl(formal_file, formal_rows),
        "reserve": write_jsonl(reserve_file, reserve_rows),
    }

    qual_ids = {row["task_id"] for row in qual_rows}
    formal_ids = {row["task_id"] for row in formal_rows}
    reserve_ids = {row["task_id"] for row in reserve_rows}
    if len(qual_ids) != len(qual_rows) or len(formal_ids) != len(formal_rows) or len(reserve_ids) != len(reserve_rows):
        raise ValueError("Duplicate method task_id in a split")
    if qual_ids & formal_ids or qual_ids & reserve_ids or formal_ids & reserve_ids:
        raise ValueError("Split overlap detected")

    max_tokens = None
    tokenizer_note = None
    if args.tokenizer_json:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer_json)
        template = Path(args.prompt_template).read_text(encoding="utf-8")
        if template.count("{problem}") != 1:
            raise ValueError("Prompt template must contain exactly one {problem}")
        lengths = []
        for row in qual_rows + formal_rows:
            prompt = template.format(problem=row["problem"])
            lengths.append(len(tokenizer.encode(prompt).ids))
        max_tokens = max(lengths)
        tokenizer_note = {
            "tokenizer_json": args.tokenizer_json,
            "prompt_template_sha256": sha256_file(Path(args.prompt_template)),
            "max_prompt_tokens": max_tokens,
            "min_prompt_tokens": min(lengths),
            "median_prompt_tokens": sorted(lengths)[len(lengths) // 2],
        }

    manifest = {
        "status": "PASS",
        "protocol_id": args.protocol_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "split_rule": f'sha256("{args.protocol_id}|class|{{class_task_id}}") sorted after excluding {len(exclude_ids)} non-reproducible classes; qual until >= {args.qual_methods_min} methods; formal until >= {args.formal_methods_min}; remainder reserve',
        "runnable_subset_note": "Excluded classes whose official canonical tests require niche third-party packages, depend on current time or machine identity, or contain official dataset defects.",
        "excluded_class_ids_with_reasons": excluded_reasons,
        "class_counts": {
            "qualification": len(qual_classes),
            "formal": len(formal_classes),
            "reserve": len(reserve_classes),
        },
        "method_counts": {
            "qualification": len(qual_rows),
            "formal": len(formal_rows),
            "reserve": len(reserve_rows),
        },
        "files": {
            "qualification": {"path": "data/classeval_qualification.jsonl", "sha256": hashes["qualification"]},
            "formal": {"path": "data/classeval_formal.jsonl", "sha256": hashes["formal"]},
            "reserve": {"path": "data/classeval_reserve.jsonl", "sha256": hashes["reserve"]},
        },
        "qual_class_ids": [cls["task_id"] for cls in qual_classes],
        "formal_class_ids": [cls["task_id"] for cls in formal_classes],
        "reserve_class_ids": [cls["task_id"] for cls in reserve_classes],
        "tokenizer_note": tokenizer_note,
        "checks": {
            "qualification_ge_96": len(qual_rows) >= args.qual_methods_min,
            "formal_ge_200": len(formal_rows) >= args.formal_methods_min,
            "no_split_overlap": True,
            "no_solution_code_in_rows": True,
            "no_excluded_class_in_splits": True,
        },
    }
    manifest_path = output_dir / "classeval_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(
        {
            "status": "PASS",
            "method_counts": manifest["method_counts"],
            "class_counts": manifest["class_counts"],
            "max_prompt_tokens": max_tokens,
            "manifest_sha256": sha256_file(manifest_path),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
