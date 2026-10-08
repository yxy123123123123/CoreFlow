#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CORE_PYTHON:-${PYTHON:-/root/autodl-tmp/coreflow_m0/runtime_env/bin/python}}"
export PYTHONPATH="$ROOT/scripts:$ROOT/coreflow:${PYTHONPATH:-}"
usage(){ echo "Usage: bash scripts/run_reviewer_required.sh {preflight|prepare|seal|local-audit|q224|drift|comol-groups|service|analyze|status|pack|all-must|all}" >&2; exit 2; }
stage="${1:-}"; shift || true
case "$stage" in
  preflight) exec "$PYTHON_BIN" "$ROOT/scripts/preflight_required.py" "$@" ;;
  prepare) exec "$PYTHON_BIN" "$ROOT/scripts/prepare_required.py" "$@" ;;
  seal) exec "$PYTHON_BIN" "$ROOT/scripts/seal_required.py" "$@" ;;
  local-audit)
    "$PYTHON_BIN" "$ROOT/scripts/compile_interface_tests.py"
    "$PYTHON_BIN" "$ROOT/scripts/storage_audit.py"
    "$PYTHON_BIN" "$ROOT/scripts/lifecycle_audit.py"
    "$PYTHON_BIN" "$ROOT/scripts/theory_error_analysis.py"
    "$PYTHON_BIN" "$ROOT/scripts/statistics_required.py" ;;
  q224) exec "$PYTHON_BIN" "$ROOT/scripts/run_q224.py" --dataset both "$@" ;;
  drift) exec "$PYTHON_BIN" "$ROOT/scripts/run_drift.py" "$@" ;;
  comol-groups) exec "$PYTHON_BIN" "$ROOT/scripts/run_comol_groups.py" "$@" ;;
  service) exec "$PYTHON_BIN" "$ROOT/scripts/run_service.py" "$@" ;;
  analyze) exec "$PYTHON_BIN" "$ROOT/scripts/analyze_required.py" "$@" ;;
  status) exec "$PYTHON_BIN" "$ROOT/scripts/status_required.py" "$@" ;;
  pack) exec "$PYTHON_BIN" "$ROOT/scripts/pack_required.py" "$@" ;;
  all-must)
    "$PYTHON_BIN" "$ROOT/scripts/preflight_required.py"
    "$PYTHON_BIN" "$ROOT/scripts/prepare_required.py"
    "$PYTHON_BIN" "$ROOT/scripts/seal_required.py"
    "$PYTHON_BIN" "$ROOT/scripts/compile_interface_tests.py"
    "$PYTHON_BIN" "$ROOT/scripts/storage_audit.py"
    "$PYTHON_BIN" "$ROOT/scripts/lifecycle_audit.py"
    "$PYTHON_BIN" "$ROOT/scripts/theory_error_analysis.py"
    "$PYTHON_BIN" "$ROOT/scripts/run_q224.py" --dataset both
    "$PYTHON_BIN" "$ROOT/scripts/run_drift.py"
    "$PYTHON_BIN" "$ROOT/scripts/run_comol_groups.py" --groups "${ACTIVE_GROUPS:-2,3}"
    "$PYTHON_BIN" "$ROOT/scripts/analyze_required.py"
    "$PYTHON_BIN" "$ROOT/scripts/pack_required.py" ;;
  all)
    "$ROOT/scripts/run_reviewer_required.sh" all-must
    "$ROOT/scripts/run_reviewer_required.sh" service --hardware "${HARDWARE_LABEL:-current_gpu}"
    "$ROOT/scripts/run_reviewer_required.sh" analyze
    "$ROOT/scripts/run_reviewer_required.sh" pack ;;
  *) usage ;;
esac
