#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


CHUNK = 8 << 20


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--payload-lock", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    lock = json.loads(Path(args.payload_lock).read_text(encoding="utf-8"))
    destination = Path(args.destination).resolve()
    marker = Path(args.marker).resolve()
    if digest_file(archive) != lock["archive_sha256"]:
        raise ValueError("Asset archive SHA-256 mismatch")
    expected = dict(lock["payload_files"])
    destination.mkdir(parents=True, exist_ok=True)
    observed: dict[str, str] = {}
    installed = 0
    reused = 0

    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = destination.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member type: {member.name}")
            if member.name not in expected:
                raise ValueError(f"Unexpected asset payload file: {member.name}")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read {member.name}")
            digest = hashlib.sha256()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file():
                    raise ValueError(f"Asset target is not a regular file: {target}")
                while chunk := source.read(CHUNK):
                    digest.update(chunk)
                observed_hash = digest.hexdigest()
                if digest_file(target) != observed_hash:
                    raise ValueError(f"Existing target differs; refusing to overwrite: {target}")
                reused += 1
            else:
                temporary = target.with_name(target.name + ".installing")
                if temporary.exists():
                    raise ValueError(f"Retained partial asset exists: {temporary}")
                with temporary.open("xb") as output:
                    while chunk := source.read(CHUNK):
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                observed_hash = digest.hexdigest()
                if observed_hash != expected[member.name]["sha256"]:
                    raise ValueError(f"Extracted asset hash mismatch: {member.name}")
                os.replace(temporary, target)
                installed += 1
            if observed_hash != expected[member.name]["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {member.name}")
            observed[member.name] = observed_hash

    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ValueError(f"Asset payload is incomplete: {missing}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "status": "PASS",
        "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": lock["archive_sha256"],
        "destination": str(destination),
        "files": len(observed),
        "new_files": installed,
        "reused_files": reused,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "files": len(observed),
        "new_files": installed,
        "reused_files": reused,
        "marker": str(marker),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
