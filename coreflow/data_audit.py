from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TEXT_KEYS = ("text", "prompt", "question", "instruction", "query", "input", "problem", "description")
ID_KEYS = ("task_id", "id", "problem_id", "question_id")
TEST_KEYS = ("test_list", "tests", "test_cases", "assertions")
VISIBLE_TEST_KEYS = ("visible_tests", "public_tests", "examples")
HIDDEN_TEST_KEYS = ("hidden_tests", "private_tests", "scoring_tests")
SETUP_KEYS = ("test_setup_code", "test_imports", "setup")
CODE_KEYS = ("code", "canonical_solution", "solution", "ground_truth", "answer")


@dataclass(frozen=True)
class Collection:
    source: Path
    label: str
    rows: list[dict]

    @property
    def identity(self) -> str:
        return f"{self.source}::{self.label}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_collection_binding(
    collection: Collection,
    *,
    expected_filename: str | None = None,
    expected_sha256: str | None = None,
    expected_sha256_prefix: str | None = None,
) -> str:
    observed = sha256_file(collection.source)
    if expected_filename and collection.source.name != expected_filename:
        raise ValueError(
            f"Explicit data binding expected {expected_filename}, got {collection.source.name}: {collection.source}"
        )
    if expected_sha256 and observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {collection.source}: {observed}")
    if expected_sha256_prefix and not observed.startswith(expected_sha256_prefix):
        raise ValueError(f"SHA-256 prefix mismatch for {collection.source}: {observed}")
    return observed


def canonical_json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def collection_fingerprint(rows: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
    return digest.hexdigest()


def _row_dict(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    nested = value.get("data")
    if isinstance(nested, dict) and any(key in nested for key in TEXT_KEYS + ID_KEYS + TEST_KEYS):
        merged = dict(nested)
        for key in ("ground_truth", "prompt_inputs"):
            if key in value and key not in merged:
                merged[key] = value[key]
        return merged
    if isinstance(nested, (list, tuple)) and nested:
        merged = dict(value)
        prompt = nested[0]
        response = nested[1] if len(nested) > 1 else ""
        merged.setdefault("prompt", prompt)
        merged.setdefault("text", prompt)
        merged.setdefault("response", response)
        merged.setdefault("code", response)
        merged.pop("data", None)
        return merged
    return value


def _collections_from_json(value, source: Path, label: str = "root") -> list[Collection]:
    if isinstance(value, list):
        rows = [_row_dict(item) for item in value]
        rows = [item for item in rows if item is not None]
        return [Collection(source, label, rows)] if rows else []
    if not isinstance(value, dict):
        return []
    direct = _row_dict(value)
    if direct is not None and any(key in direct for key in TEXT_KEYS + ID_KEYS + TEST_KEYS):
        return [Collection(source, label, [direct])]
    found: list[Collection] = []
    for key, child in value.items():
        if isinstance(child, (list, dict)):
            found.extend(_collections_from_json(child, source, f"{label}.{key}"))
    return found


def read_collections(path: str | Path) -> list[Collection]:
    source = Path(path).resolve()
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                row = _row_dict(value)
                if row is None:
                    raise ValueError(f"Non-object row at {source}:{line_no}")
                rows.append(row)
        return [Collection(source, "jsonl", rows)]
    if suffix == ".json":
        return _collections_from_json(json.loads(source.read_text(encoding="utf-8")), source)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(f"pyarrow is required to inspect {source}") from exc
        table = pq.read_table(source)
        rows = [_row_dict(item) for item in table.to_pylist()]
        return [Collection(source, "parquet", [item for item in rows if item is not None])]
    return []


def discover_collections(roots: Iterable[str | Path], *, max_file_mb: int = 32) -> list[Collection]:
    allowed = {".json", ".jsonl", ".parquet"}
    seen: set[Path] = set()
    found: list[Collection] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            try:
                resolved = path.resolve()
                if resolved in seen or not resolved.is_file() or resolved.suffix.lower() not in allowed:
                    continue
                seen.add(resolved)
                if resolved.stat().st_size > max_file_mb * 1024 * 1024:
                    continue
                if any(part in {"core_banks", ".git", "__pycache__"} for part in resolved.parts):
                    continue
                found.extend(read_collections(resolved))
            except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError):
                continue
    return found


def _first(row: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def _conversation_text(row: dict) -> str:
    conversations = row.get("conversations") or row.get("messages")
    if not isinstance(conversations, list):
        return ""
    parts = []
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", item.get("role", ""))).lower()
        if role in {"human", "user"}:
            parts.append(str(item.get("value", item.get("content", ""))))
    return "\n".join(parts)


def parse_tests(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [line.strip() for line in text.splitlines() if line.strip().startswith("assert ")]
    return []


def normalize_eval_row(row: dict, source_index: int) -> dict | None:
    text = _first(row, TEXT_KEYS, "") or _conversation_text(row)
    hidden_tests = parse_tests(_first(row, HIDDEN_TEST_KEYS))
    tests = hidden_tests or parse_tests(_first(row, TEST_KEYS))
    visible_tests = parse_tests(_first(row, VISIBLE_TEST_KEYS))
    if not str(text).strip() or not tests:
        return None
    task_id = _first(row, ID_KEYS, f"source-{source_index}")
    setup = _first(row, SETUP_KEYS, "")
    if isinstance(setup, list):
        setup = "\n".join(str(item) for item in setup)
    elif isinstance(setup, str) and setup.strip().startswith(("[", "(")):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(setup)
                if isinstance(parsed, (list, tuple)):
                    setup = "\n".join(str(item) for item in parsed)
                    break
            except Exception:
                pass
    code = _first(row, CODE_KEYS, "")
    return {
        "task_id": str(task_id),
        "text": str(text).strip(),
        "code": str(code or ""),
        "test_list": tests,
        "visible_tests": visible_tests,
        "hidden_tests": tests,
        "test_setup_code": str(setup or ""),
        "source_index": int(source_index),
    }


def reference_identity(row: dict, source_index: int) -> dict | None:
    text = _first(row, TEXT_KEYS, "") or _conversation_text(row)
    if not str(text).strip():
        prompt_inputs = row.get("prompt_inputs")
        if isinstance(prompt_inputs, list) and prompt_inputs:
            text = prompt_inputs[0]
    if not str(text).strip():
        return None
    task_id = _first(row, ID_KEYS)
    return {
        "task_id": None if task_id is None else str(task_id),
        "text": str(text),
        "normalized_text": normalize_problem_text(str(text)),
        "source_index": int(source_index),
    }


def normalize_problem_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = value.replace("<s>", " ").replace("</s>", " ")
    match = re.search(r"为这个问题创建一个python脚本\s*[:：]\s*(.*?)(?:\[/inst\]|$)", value, re.S)
    if match:
        value = match.group(1)
    value = re.sub(r"\[/?inst\]", " ", value, flags=re.I)
    value = re.sub(r"```.*?```", " ", value, flags=re.S)
    value = "\n".join(line for line in value.splitlines() if not line.strip().startswith("assert "))
    value = re.sub(r"[\W_]+", "", value, flags=re.UNICODE)
    return value


def chinese_ratio(text: str) -> float:
    compact = [char for char in text if not char.isspace()]
    if not compact:
        return 0.0
    chinese = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    return chinese / len(compact)


def char_ngrams(text: str, n: int) -> set[str]:
    value = normalize_problem_text(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _path_score(collection: Collection, tokens: tuple[str, ...]) -> int:
    path = collection.identity.lower().replace("\\", "/")
    return sum(2 if token in path else 0 for token in tokens)


def choose_eval_source(collections: list[Collection], expected_rows: int, chinese_minimum: float) -> tuple[Collection, list[dict]]:
    candidates = []
    for collection in collections:
        rows = [normalize_eval_row(row, index) for index, row in enumerate(collection.rows)]
        rows = [row for row in rows if row is not None]
        if len(rows) != expected_rows:
            continue
        ratio = sum(chinese_ratio(row["text"]) for row in rows) / len(rows)
        if ratio < chinese_minimum:
            continue
        score = _path_score(collection, ("mbpp", "zh", "chinese", "test")) + int(ratio * 10)
        candidates.append((score, collection_fingerprint(rows), collection, rows))
    if not candidates:
        raise ValueError(f"No held-out code collection with exactly {expected_rows} valid rows")
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    fingerprints = {item[1] for item in best}
    if len(fingerprints) != 1:
        names = [item[2].identity for item in best]
        raise ValueError(f"Ambiguous held-out MBPP sources: {names}")
    _, _, collection, rows = sorted(best, key=lambda item: item[2].identity)[0]
    return collection, rows


def choose_reference(
    collections: list[Collection],
    expected_rows: int,
    role: str,
    *,
    exclude_identity: str | None = None,
) -> tuple[Collection, list[dict]]:
    candidates = []
    role_tokens = ("train", "training") if role == "train" else ("dev", "valid", "validation")
    for collection in collections:
        if exclude_identity is not None and collection.identity == exclude_identity:
            continue
        rows = [reference_identity(row, index) for index, row in enumerate(collection.rows)]
        rows = [row for row in rows if row is not None]
        if len(rows) != expected_rows:
            continue
        ratio = sum(chinese_ratio(row["text"]) for row in rows) / len(rows)
        score = _path_score(collection, ("mbpp", "code", "zh", "chinese") + role_tokens) + int(ratio * 10)
        candidates.append((score, collection_fingerprint(rows), collection, rows))
    if not candidates:
        raise ValueError(f"No {role} contamination reference with exactly {expected_rows} text rows")
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    fingerprints = {item[1] for item in best}
    if len(fingerprints) != 1:
        names = [item[2].identity for item in best]
        raise ValueError(f"Ambiguous {role} contamination references: {names}")
    _, _, collection, rows = sorted(best, key=lambda item: item[2].identity)[0]
    return collection, rows


def audit_and_filter(
    eval_rows: list[dict],
    references: list[dict],
    *,
    ngram_n: int,
    near_threshold: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    reference_ids = {row["task_id"] for row in references if row.get("task_id") is not None}
    reference_text = {row["normalized_text"] for row in references if row.get("normalized_text")}
    reference_ngrams = [(row, char_ngrams(row["text"], ngram_n)) for row in references]
    kept, excluded = [], []
    for row in eval_rows:
        normalized = normalize_problem_text(row["text"])
        reasons: list[dict] = []
        if row["task_id"] in reference_ids:
            reasons.append({"type": "task_id", "value": row["task_id"]})
        if normalized in reference_text:
            reasons.append({"type": "normalized_text_exact"})
        grams = char_ngrams(row["text"], ngram_n)
        best_score, best_ref = 0.0, None
        for reference, ref_grams in reference_ngrams:
            score = jaccard(grams, ref_grams)
            if score > best_score:
                best_score, best_ref = score, reference
        if best_score >= near_threshold:
            reasons.append(
                {
                    "type": "near_duplicate",
                    "jaccard": best_score,
                    "reference_task_id": None if best_ref is None else best_ref.get("task_id"),
                }
            )
        if reasons:
            excluded.append({"task_id": row["task_id"], "source_index": row["source_index"], "reasons": reasons})
        else:
            frozen = dict(row)
            frozen["normalized_text_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            frozen["order_key"] = hashlib.sha256(f"{seed}:{row['task_id']}:{normalized}".encode("utf-8")).hexdigest()
            kept.append(frozen)
    kept.sort(key=lambda item: (item["order_key"], item["task_id"]))
    return kept, excluded


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_search_roots(work_root: Path) -> list[Path]:
    roots = [
        work_root / "vendor" / "LoRAFlow",
        work_root / "official_assets",
        work_root / "data",
        work_root / "train_data",
        work_root / "results",
        Path("/root/.cache/huggingface/datasets"),
        Path("/root/autodl-tmp/.cache/huggingface/datasets"),
    ]
    return [path for path in roots if path.exists()]
