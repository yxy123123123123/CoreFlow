#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-}"

SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
FORMAL_UPLOAD_ROOT="${FORMAL_UPLOAD_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_upload}"
FORMAL_WORK_ROOT="${FORMAL_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace}"
M2_ASSET_ROOT="${M2_ASSET_ROOT:-${FORMAL_WORK_ROOT}/m2_assets}"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_m3g_real_gate_entropy_workspace}"
PYTHON="${PYTHON:-${SOURCE_WORK_ROOT}/runtime_env/bin/python}"

REPORT_ROOT="${WORK_ROOT}/reports/m3g_real_gate_entropy_v1"
RESULT_ROOT="${WORK_ROOT}/results/m3g_real_gate_entropy_v1"
CONFIG="${ROOT}/config/m3g_real_gate_entropy_protocol.json"
OUTPUT="${REPORT_ROOT}/real_gate_entropy.json"

on_error() {
  local line="$1"
  echo "[ERROR] Stopped at line ${line} (exit 1); partial output was retained and no automatic retry was attempted." >&2
}
trap 'on_error $LINENO' ERR

verify_package() {
  "${PYTHON}" "${ROOT}/scripts/package_checksums.py" --root "${ROOT}"
}

check_paths() {
  test -x "${PYTHON}" || { echo "[ERROR] Missing python: ${PYTHON}" >&2; exit 1; }
  test -d "${SOURCE_WORK_ROOT}" || { echo "[ERROR] Missing SOURCE_WORK_ROOT: ${SOURCE_WORK_ROOT}" >&2; exit 1; }
  test -d "${FORMAL_UPLOAD_ROOT}" || { echo "[ERROR] Missing FORMAL_UPLOAD_ROOT: ${FORMAL_UPLOAD_ROOT}" >&2; exit 1; }
  test -d "${FORMAL_WORK_ROOT}" || { echo "[ERROR] Missing FORMAL_WORK_ROOT: ${FORMAL_WORK_ROOT}" >&2; exit 1; }
  test -d "${MODEL}" || { echo "[ERROR] Missing MODEL: ${MODEL}" >&2; exit 1; }
  test -d "${M2_ASSET_ROOT}/gates" || { echo "[ERROR] Missing M2 gate directory: ${M2_ASSET_ROOT}/gates" >&2; exit 1; }
  for seed in 41 42 43; do
    test -f "${M2_ASSET_ROOT}/gates/code_k5_code_seed${seed}.pt" || {
      echo "[ERROR] Missing gate: ${M2_ASSET_ROOT}/gates/code_k5_code_seed${seed}.pt" >&2
      exit 1
    }
  done
  test -f "${FORMAL_WORK_ROOT}/data/formal_v2_primary/mbppplus_formal_candidate250.jsonl" || {
    test -f "${FORMAL_UPLOAD_ROOT}/data/mbppplus_formal_candidate250.jsonl" || {
      echo "[ERROR] Missing MBPP+ formal candidate data in formal workspace/upload." >&2
      exit 1
    }
  }
  test -f "${FORMAL_UPLOAD_ROOT}/data/prompt_code_mbppplus.txt" || {
    echo "[ERROR] Missing prompt file: ${FORMAL_UPLOAD_ROOT}/data/prompt_code_mbppplus.txt" >&2
    exit 1
  }
}

status() {
  echo "=== M3G real gate entropy audit ==="
  if test -f "${OUTPUT}"; then
    "${PYTHON}" - <<'PY' "${OUTPUT}"
import json, sys
path = sys.argv[1]
obj = json.load(open(path, "r", encoding="utf-8"))
print("real_gate_entropy.json:", obj.get("status"), obj.get("decision"))
print("pooled:", obj.get("pooled_sparsity_fractions"))
for seed in obj.get("seeds", []):
    print(f"seed{seed.get('seed')}: rows={seed.get('rows')} elapsed_min={seed.get('elapsed_seconds', 0)/60:.1f} sparsity={seed.get('sparsity_fractions')}")
PY
  else
    echo "real_gate_entropy.json: MISSING"
  fi
  echo
  echo "=== paths ==="
  echo "SOURCE_WORK_ROOT=${SOURCE_WORK_ROOT}"
  echo "FORMAL_UPLOAD_ROOT=${FORMAL_UPLOAD_ROOT}"
  echo "FORMAL_WORK_ROOT=${FORMAL_WORK_ROOT}"
  echo "M2_ASSET_ROOT=${M2_ASSET_ROOT}"
  echo "WORK_ROOT=${WORK_ROOT}"
}

analyze() {
  verify_package
  check_paths
  mkdir -p "${REPORT_ROOT}" "${RESULT_ROOT}"
  "${PYTHON}" "${ROOT}/scripts/m3g_real_gate_entropy.py" \
    --config "${CONFIG}" \
    --source-work-root "${SOURCE_WORK_ROOT}" \
    --formal-upload-root "${FORMAL_UPLOAD_ROOT}" \
    --formal-work-root "${FORMAL_WORK_ROOT}" \
    --m2-asset-root "${M2_ASSET_ROOT}" \
    --model "${MODEL}" \
    --output "${OUTPUT}"
}

pack_results() {
  verify_package
  test -f "${OUTPUT}" || { echo "[ERROR] Run analyze first" >&2; exit 1; }
  local archive="/root/autodl-tmp/coreflow_m3g_real_gate_entropy_results.tar.gz"
  tar -C "${WORK_ROOT}" -czf "${archive}" reports
  sha256sum "${archive}" > "${archive}.sha256"
  echo "{\"status\":\"PACKED\",\"archive\":\"${archive}\",\"sha256\":\"${archive}.sha256\"}"
}

case "${CMD}" in
  analyze)
    analyze
    ;;
  pack)
    pack_results
    ;;
  status)
    status
    ;;
  all)
    analyze
    status
    pack_results
    ;;
  *)
    echo "Usage: bash scripts/run_m3g_real_gate_entropy.sh {analyze|status|pack|all}" >&2
    exit 2
    ;;
esac
