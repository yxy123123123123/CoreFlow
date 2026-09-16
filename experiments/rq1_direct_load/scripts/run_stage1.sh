#!/usr/bin/env bash
set -Eeuo pipefail
# No automatic retry. Completed stages are reused; partial outputs stop the run.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_submission_final_stage1_v1_workspace}"
PY="${PY:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
export SOURCE_WORK_ROOT WORK_ROOT MODEL="${MODEL:-/root/autodl-tmp/Model}"
export M2_ASSET_ROOT="${M2_ASSET_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets}"
export M2_ASSET_LOCK="${M2_ASSET_LOCK:-/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json}"
export FORMAL_V2_WORK_ROOT="${FORMAL_V2_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONPATH="$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/transformers/src:$SOURCE_WORK_ROOT/vendor/LoRAFlow/UltraEval/lora-fusion/peft-group_lora/src:${PYTHONPATH:-}"
mkdir -p "$WORK_ROOT/logs"
verify(){ "$PY" "$ROOT/scripts/package_checksums.py" --root "$ROOT"; }
run(){ local log="$1";shift;verify;"$@" 2>&1 | tee "$WORK_ROOT/logs/$log.log"; }
stage="${1:-}"
case "$stage" in
  preflight) run 01_preflight "$PY" "$ROOT/scripts/preflight.py" ;;
  prepare) run 02_prepare "$PY" "$ROOT/scripts/prepare.py" ;;
  seal) run 03_seal "$PY" "$ROOT/scripts/seal.py" ;;
  qualification) run 04_qualification "$PY" "$ROOT/scripts/run_qualification.py" ;;
  qualify) run 05_qualification_decision "$PY" "$ROOT/scripts/decide_qualification.py" ;;
  e1-isvd) run 06_e1_isvd "$PY" "$ROOT/scripts/run_e1_isvd.py" ;;
  e1d) run 07_e1d "$PY" "$ROOT/scripts/run_e1d.py" ;;
  decide) run 08_stage1_decision "$PY" "$ROOT/scripts/decide_stage1.py" ;;
  status) verify;"$PY" "$ROOT/scripts/status.py" ;;
  pack)
    verify
    test -f "$WORK_ROOT/reports/stage1/stage1_decision.json" || { echo '[ERROR] Run decide first' >&2;exit 1; }
    OUT=/root/autodl-tmp/coreflow_submission_final_stage1_v1_results.tar.gz
    tar -czf "$OUT" -C "$WORK_ROOT" reports/stage1 results/stage1 data/stage1 banks/stage1 logs
    sha256sum "$OUT" > "$OUT.sha256";echo "PACKED: $OUT" ;;
  all-prepare) bash "$0" preflight && bash "$0" prepare && bash "$0" seal ;;
  all) bash "$0" all-prepare && bash "$0" qualification && bash "$0" qualify && bash "$0" e1-isvd && bash "$0" e1d && bash "$0" decide && bash "$0" pack ;;
  *) echo 'Usage: bash scripts/run_stage1.sh {preflight|prepare|seal|qualification|qualify|e1-isvd|e1d|decide|status|pack|all-prepare|all}' >&2;exit 1 ;;
esac
