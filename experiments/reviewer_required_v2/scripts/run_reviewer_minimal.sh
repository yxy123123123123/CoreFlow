#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}/runtime_env/bin/python}"
export PYTHONPATH="$ROOT/scripts:$ROOT/coreflow:${PYTHONPATH:-}"

usage() {
  echo "Usage: bash scripts/run_reviewer_minimal.sh {preflight|prepare|seal|q224|drift|service|analyze|status|pack|all} [args...]" >&2
  exit 2
}

stage="${1:-}"
shift || true
case "$stage" in
  preflight) exec "$PYTHON_BIN" "$ROOT/scripts/preflight.py" "$@" ;;
  prepare) exec "$PYTHON_BIN" "$ROOT/scripts/prepare.py" "$@" ;;
  seal) exec "$PYTHON_BIN" "$ROOT/scripts/seal.py" "$@" ;;
  q224) exec "$PYTHON_BIN" "$ROOT/scripts/run_q224.py" "$@" ;;
  drift) exec "$PYTHON_BIN" "$ROOT/scripts/run_drift.py" "$@" ;;
  service) exec "$PYTHON_BIN" "$ROOT/scripts/run_service.py" "$@" ;;
  analyze) exec "$PYTHON_BIN" "$ROOT/scripts/analyze.py" "$@" ;;
  status) exec "$PYTHON_BIN" "$ROOT/scripts/status.py" "$@" ;;
  pack) exec "$PYTHON_BIN" "$ROOT/scripts/pack.py" "$@" ;;
  all)
    "$PYTHON_BIN" "$ROOT/scripts/preflight.py"
    "$PYTHON_BIN" "$ROOT/scripts/prepare.py"
    "$PYTHON_BIN" "$ROOT/scripts/seal.py"
    "$PYTHON_BIN" "$ROOT/scripts/run_q224.py" --dataset both
    "$PYTHON_BIN" "$ROOT/scripts/run_drift.py"
    "$PYTHON_BIN" "$ROOT/scripts/run_service.py" --hardware "${HARDWARE_LABEL:-current_gpu}"
    "$PYTHON_BIN" "$ROOT/scripts/analyze.py"
    "$PYTHON_BIN" "$ROOT/scripts/pack.py"
    ;;
  *) usage ;;
esac
