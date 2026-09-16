#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
FORMAL_V1_WORK_ROOT="${FORMAL_V1_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace}"
FORMAL_V2_WORK_ROOT="${FORMAL_V2_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace}"
M3_WORK_ROOT="${M3_WORK_ROOT:-/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_prs3_pilot_v1_workspace}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PY="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
export MODEL SOURCE_WORK_ROOT FORMAL_V1_WORK_ROOT FORMAL_V2_WORK_ROOT M3_WORK_ROOT WORK_ROOT CUDA_DEVICE
export PYTHONPATH="$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/transformers/src:$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/peft-group_lora/src:${PYTHONPATH:-}"
mkdir -p "$WORK_ROOT/logs"

trap 'code=$?; echo "[ERROR] Stopped at line $LINENO (exit $code); partial output was retained and no automatic retry was attempted." >&2; exit $code' ERR

verify() { "$PY" "$ROOT/scripts/package_checksums.py"; }
run_logged() { local label="$1"; shift; "$@" 2>&1 | tee "$WORK_ROOT/logs/${label}.log"; }

preflight() {
  verify
  run_logged 01_tests "$PY" "$ROOT/tests/test_contract.py"
  run_logged 02_preflight env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/preflight.py"
}
prepare() { verify; run_logged 03_prepare "$PY" "$ROOT/scripts/prepare.py"; }
seal() { verify; run_logged 04_seal "$PY" "$ROOT/scripts/seal.py"; }
prs3a() { verify; run_logged 05_prs3a env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/run_prs3a.py"; }
prs3b() { verify; run_logged 06_prs3b env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/run_prs3b.py"; }
analyze() { verify; run_logged 07_analyze "$PY" "$ROOT/scripts/analyze.py"; }
pack() { verify; run_logged 08_pack "$PY" "$ROOT/scripts/pack_results.py"; }
status() { "$PY" "$ROOT/scripts/status.py"; }

case "${1:-}" in
  preflight) preflight ;;
  prepare) prepare ;;
  seal) seal ;;
  all-prepare) preflight; prepare; seal ;;
  prs3a) prs3a ;;
  prs3b) prs3b ;;
  analyze) analyze ;;
  pack) pack ;;
  status) status ;;
  all) preflight; prepare; seal; prs3a; prs3b; analyze; pack ;;
  *) echo "Usage: bash scripts/run_prs3_pilot.sh {preflight|prepare|seal|all-prepare|prs3a|prs3b|analyze|pack|status|all}" >&2; exit 2 ;;
esac
