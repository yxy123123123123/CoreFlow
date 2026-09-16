#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
M2_ASSET_ROOT="${M2_ASSET_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets}"
PHASE2B_WORK_ROOT="${PHASE2B_WORK_ROOT:-/root/autodl-tmp/coreflow_applsci_phase2b_s6s7_v1_hotfix5_workspace}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_prs5_weight_audit_v1_workspace}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
export SOURCE_WORK_ROOT M2_ASSET_ROOT PHASE2B_WORK_ROOT WORK_ROOT
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
mkdir -p "$WORK_ROOT/logs"

verify() { "$PYTHON" "$ROOT/scripts/package_checksums.py" --root "$ROOT"; }

case "${1:-}" in
  verify) verify ;;
  preflight) verify; "$PYTHON" "$ROOT/tests/run_tests.py"; "$PYTHON" "$ROOT/scripts/preflight.py" ;;
  seal) verify; "$PYTHON" "$ROOT/scripts/seal.py" ;;
  audit) verify; "$PYTHON" "$ROOT/scripts/run_weight_audit.py" ;;
  analyze) verify; "$PYTHON" "$ROOT/scripts/analyze.py" ;;
  pack) verify; "$PYTHON" "$ROOT/scripts/pack_results.py" ;;
  status) "$PYTHON" "$ROOT/scripts/status.py" ;;
  all)
    verify
    if [[ ! -f "$WORK_ROOT/reports/prs5/PRS5_SEAL.json" ]]; then
      "$PYTHON" "$ROOT/tests/run_tests.py"
      "$PYTHON" "$ROOT/scripts/preflight.py"
      "$PYTHON" "$ROOT/scripts/seal.py"
    else
      echo "[SKIP] workspace already sealed"
    fi
    "$PYTHON" "$ROOT/scripts/run_weight_audit.py"
    "$PYTHON" "$ROOT/scripts/analyze.py"
    "$PYTHON" "$ROOT/scripts/pack_results.py"
    ;;
  *) echo "Usage: bash scripts/run_prs5.sh {verify|preflight|seal|audit|analyze|pack|status|all}" >&2; exit 2 ;;
esac
