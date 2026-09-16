#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_prs4a_prs4p_pilot_v1_workspace}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
export SOURCE_WORK_ROOT WORK_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
mkdir -p "$WORK_ROOT/logs"

verify() {
  "$PYTHON" "$ROOT/scripts/package_checksums.py" --root "$ROOT"
}

prepare_if_needed() {
  if [[ ! -f "$WORK_ROOT/reports/prs4a_prs4p/PILOT_SEAL.json" ]]; then
    "$PYTHON" "$ROOT/scripts/preflight.py"
    "$PYTHON" "$ROOT/scripts/archive_a.py"
    "$PYTHON" "$ROOT/scripts/seal.py"
  else
    echo "[SKIP] workspace already sealed"
  fi
}

case "${1:-}" in
  verify)
    verify
    ;;
  preflight)
    verify
    "$PYTHON" "$ROOT/scripts/preflight.py"
    ;;
  archive-a)
    verify
    "$PYTHON" "$ROOT/scripts/archive_a.py"
    ;;
  seal)
    verify
    "$PYTHON" "$ROOT/scripts/seal.py"
    ;;
  pilot-b)
    verify
    "$PYTHON" "$ROOT/scripts/run_pilot_b.py"
    ;;
  analyze)
    verify
    "$PYTHON" "$ROOT/scripts/analyze.py"
    ;;
  pack)
    verify
    "$PYTHON" "$ROOT/scripts/pack_results.py"
    ;;
  status)
    "$PYTHON" "$ROOT/scripts/status.py"
    ;;
  all)
    verify
    prepare_if_needed
    "$PYTHON" "$ROOT/scripts/run_pilot_b.py"
    "$PYTHON" "$ROOT/scripts/analyze.py"
    "$PYTHON" "$ROOT/scripts/pack_results.py"
    ;;
  *)
    echo "Usage: bash scripts/run_prs4a_prs4p.sh {verify|preflight|archive-a|seal|pilot-b|analyze|pack|status|all}" >&2
    exit 2
    ;;
esac
