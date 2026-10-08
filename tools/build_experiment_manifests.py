"""Generate integrity manifests for experiment subpackages in this GitHub layout."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PACKAGES = {
    "rq1_compilation_correctness": "PACKAGE_MANIFEST.sha256",
    "rq1_direct_load": "PACKAGE_MANIFEST.sha256",
    "rq2_mbppplus_quality": "PACKAGE_MANIFEST.sha256",
    "rq2_classeval_confirmation": "PACKAGE_MANIFEST.sha256",
    "rq3_system_efficiency": "CHECKSUMS.sha256",
    "rq4_workload_k_scaling/prs3_workloads": "PACKAGE_MANIFEST.sha256",
    "rq4_workload_k_scaling/m3_k_scaling": "PACKAGE_MANIFEST.sha256",
    "rq5_ablation_gate/phase2b_qgrid_ablation": "PACKAGE_MANIFEST.sha256",
    "rq5_ablation_gate/prs5_weight_audit": "PACKAGE_MANIFEST.sha256",
    "rq5_ablation_gate/m3g_real_gate": "PACKAGE_MANIFEST.sha256",
    "rq5_ablation_gate/m3r_routing_diagnostics": "PACKAGE_MANIFEST.sha256",
    "reviewer_required_v2": "PACKAGE_MANIFEST.sha256",
    "archive/prs4_reference_method_failed": "PACKAGE_MANIFEST.sha256",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(relative_root: str, manifest_name: str) -> int:
    package_root = EXPERIMENTS / relative_root
    manifest = package_root / manifest_name
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    lines = [f"{digest(path)}  {path.relative_to(package_root).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    for relative_root, manifest_name in PACKAGES.items():
        count = write_manifest(relative_root, manifest_name)
        print(f"WROTE {relative_root}/{manifest_name} entries={count}")


if __name__ == "__main__":
    main()
