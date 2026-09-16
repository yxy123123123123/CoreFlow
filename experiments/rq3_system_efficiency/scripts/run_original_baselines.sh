#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
FORMAL_V1_WORK_ROOT="${FORMAL_V1_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace}"
FORMAL_V2_WORK_ROOT="${FORMAL_V2_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_applsci_original_baselines_v1_workspace}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PY="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
export MODEL SOURCE_WORK_ROOT FORMAL_V1_WORK_ROOT FORMAL_V2_WORK_ROOT WORK_ROOT CUDA_DEVICE
export PYTHONPATH="$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/transformers/src:$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/peft-group_lora/src:${PYTHONPATH:-}"
mkdir -p "$WORK_ROOT/logs"

trap 'code=$?; echo "[ERROR] Stopped at line $LINENO (exit $code); partial output was retained and no automatic retry was attempted." >&2; exit $code' ERR

verify() { "$PY" "$ROOT/scripts/package_checksums.py"; }
run_logged() { local label="$1"; shift; "$@" 2>&1 | tee "$WORK_ROOT/logs/${label}.log"; }

preflight() { verify; run_logged 01_tests "$PY" "$ROOT/tests/test_contract.py"; run_logged 02_preflight env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/preflight.py"; }
prepare() { verify; run_logged 03_prepare "$PY" "$ROOT/scripts/prepare.py"; run_logged 04_r0_assets "$PY" "$ROOT/scripts/audit_assets.py"; }
seal() { verify; run_logged 05_seal "$PY" "$ROOT/scripts/seal.py"; }
r1() {
  verify
  local seed port
  for seed in 41 42 43; do
    port=$((6700 + seed))
    run_logged "r1_seed${seed}" env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/run_r1_quality.py" --seed "$seed" --port "$port"
  done
  run_logged r1_audit "$PY" "$ROOT/scripts/audit_r1.py"
}
r2() { verify; run_logged r2_systems env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/run_r2_matrix.py"; }
r3() { verify; run_logged r3_profiler env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PY" "$ROOT/scripts/run_r3.py"; }
analyze() { verify; run_logged analyze "$PY" "$ROOT/scripts/analyze.py"; }
pack() { verify; run_logged pack "$PY" "$ROOT/scripts/pack_results.py"; }
status() { "$PY" "$ROOT/scripts/status.py"; }

case "${1:-}" in
  preflight) preflight ;;
  prepare) prepare ;;
  seal) seal ;;
  all-prepare) preflight; prepare; seal ;;
  r1) r1 ;;
  r2) r2 ;;
  r3) r3 ;;
  analyze) analyze ;;
  pack) pack ;;
  status) status ;;
  all) preflight; prepare; seal; r1; r2; r3; analyze; pack ;;
  *) echo "Usage: bash scripts/run_original_baselines.sh {preflight|prepare|seal|all-prepare|r1|r2|r3|analyze|pack|status|all}" >&2; exit 2 ;;
esac
