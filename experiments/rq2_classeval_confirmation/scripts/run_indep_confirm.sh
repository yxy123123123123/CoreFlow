#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_indep_confirm_v1_workspace}"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
M2_ASSET_ROOT="${M2_ASSET_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets}"
M2_ASSET_LOCK="${M2_ASSET_LOCK:-/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json}"
ISVD_ASSET_ROOT="${ISVD_ASSET_ROOT:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace/isvd_banks}"
ISVD_LOCK="${ISVD_LOCK:-/root/autodl-tmp/coreflow_formal_v2_primary_workspace/reports/formal_v2_primary/isvd_asset_lock.json}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
CONFIG="$ROOT/config/indep_confirm_classeval_protocol.json"
REPORT_ROOT="$WORK_ROOT/reports/indep_confirm_classeval"
RESULT_ROOT="$WORK_ROOT/results/indep_confirm_classeval"
DATA_ROOT="$WORK_ROOT/data/indep_confirm_classeval"
SEAL="$REPORT_ROOT/PROTOCOL_SEAL.json"
QUAL_DECISION="$REPORT_ROOT/qualification_decision.json"
PHASE="${1:-}"

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
trap 'die "Stopped at line ${BASH_LINENO[0]} (exit $?); partial output was retained and no automatic retry was attempted."' ERR

verify_package() {
  [[ -x "$PYTHON" ]] || die "Frozen runtime Python not found: $PYTHON"
  "$PYTHON" "$SCRIPT_DIR/package_checksums.py" --root "$ROOT"
}

require_file() {
  [[ -f "$1" ]] || die "$2"
}

require_pass() {
  require_file "$1" "$2"
  "$PYTHON" - "$1" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") not in {"PASS", "ANALYZED", "SEALED_BEFORE_FORMAL_MODEL_OUTPUTS"}:
    raise SystemExit(f"{sys.argv[1]} status is {payload.get('status')}")
PY
}

ensure_asset_link() {
  mkdir -p "$WORK_ROOT" "$REPORT_ROOT" "$RESULT_ROOT" "$WORK_ROOT/logs"
  if [[ ! -e "$WORK_ROOT/m2_assets" ]]; then
    ln -s "$M2_ASSET_ROOT" "$WORK_ROOT/m2_assets"
  fi
}

preflight() {
  ensure_asset_link
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" preflight \
    --work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --m2-asset-root "$M2_ASSET_ROOT" \
    --m2-asset-lock "$M2_ASSET_LOCK" \
    --isvd-asset-root "$ISVD_ASSET_ROOT" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/preflight.json"
}

prepare() {
  require_pass "$REPORT_ROOT/preflight.json" "Run preflight first"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" prepare \
    --work-root "$WORK_ROOT" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/data_audit.json"
}

check_isvd() {
  require_pass "$REPORT_ROOT/data_audit.json" "Run prepare first"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" check-isvd \
    --work-root "$WORK_ROOT" \
    --isvd-asset-root "$ISVD_ASSET_ROOT" \
    --isvd-lock "$ISVD_LOCK" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/isvd_asset_lock.json"
}

evaluator_seal() {
  require_pass "$REPORT_ROOT/isvd_asset_lock.json" "Run check-isvd first"
  "$PYTHON" "$SCRIPT_DIR/seal_evaluator_classeval.py" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/evaluator_seal.json"
}

seal() {
  require_pass "$REPORT_ROOT/preflight.json" "Run preflight first"
  require_pass "$REPORT_ROOT/data_audit.json" "Run prepare first"
  require_pass "$REPORT_ROOT/isvd_asset_lock.json" "Run check-isvd first"
  require_pass "$REPORT_ROOT/evaluator_seal.json" "Run evaluator-seal first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" seal \
    --work-root "$WORK_ROOT" \
    --config "$CONFIG" \
    --preflight "$REPORT_ROOT/preflight.json" \
    --data-audit "$REPORT_ROOT/data_audit.json" \
    --isvd-asset-lock "$REPORT_ROOT/isvd_asset_lock.json" \
    --evaluator-seal "$REPORT_ROOT/evaluator_seal.json" \
    --output "$SEAL"
}

qual() {
  require_pass "$SEAL" "Run seal first"
  "$PYTHON" "$SCRIPT_DIR/run_formal_parallel.py" \
    --formal-work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --python "$PYTHON" \
    --config "$CONFIG" \
    --asset-lock "$M2_ASSET_LOCK" \
    --isvd-asset-root "$ISVD_ASSET_ROOT" \
    --isvd-lock "$ISVD_LOCK" \
    --data-root "$DATA_ROOT" \
    --result-root "$RESULT_ROOT" \
    --phase qualification \
    --methods code_full_k5_seed41 \
    --output-group qualification \
    --dataset "$DATA_ROOT/classeval_qualification.jsonl" \
    --devices "${CUDA_DEVICES:-0,1}" \
    --base-port 6500
}

decide_qual() {
  require_file "$SEAL" "Run seal first"
  "$PYTHON" "$SCRIPT_DIR/decide_qualification.py" \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT" \
    --report-root "$REPORT_ROOT" \
    --output "$QUAL_DECISION"
}

formal() {
  require_pass "$SEAL" "Run seal first"
  require_pass "$QUAL_DECISION" "Run decide-qual first"
  "$PYTHON" "$SCRIPT_DIR/run_formal_parallel.py" \
    --formal-work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --python "$PYTHON" \
    --config "$CONFIG" \
    --asset-lock "$M2_ASSET_LOCK" \
    --isvd-asset-root "$ISVD_ASSET_ROOT" \
    --isvd-lock "$ISVD_LOCK" \
    --data-root "$DATA_ROOT" \
    --result-root "$RESULT_ROOT" \
    --phase formal_main \
    --output-group formal_main \
    --dataset "$DATA_ROOT/classeval_formal.jsonl" \
    --devices "${CUDA_DEVICES:-0,1}" \
    --base-port 6600
}

analyze() {
  require_pass "$SEAL" "Run seal first"
  "$PYTHON" "$SCRIPT_DIR/analyze_v2_primary.py" \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT" \
    --report-root "$REPORT_ROOT" \
    --output "$REPORT_ROOT/primary_decision.json" \
    --summary-md "$REPORT_ROOT/PRIMARY_SUMMARY.md"
}

pack() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run analyze first"
  local output="/root/autodl-tmp/coreflow_indep_confirm_v1_results.tar.gz"
  tar -czf "$output" \
    -C "$WORK_ROOT" \
    reports/indep_confirm_classeval results/indep_confirm_classeval data/indep_confirm_classeval logs \
    -C "$ROOT" \
    config provenance docs README.md PACKAGE_MANIFEST.sha256 AutoDL_独立确认实验_执行流程.md
  sha256sum "$output" > "$output.sha256"
  ls -lh "$output" "$output.sha256"
}

status() {
  "$PYTHON" "$SCRIPT_DIR/status_v2_primary.py" --work-root "$WORK_ROOT"
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi

case "$PHASE" in
  preflight) preflight ;;
  prepare) prepare ;;
  check-isvd) check_isvd ;;
  evaluator-seal) evaluator_seal ;;
  seal) seal ;;
  qual) qual ;;
  decide-qual) decide_qual ;;
  formal) formal ;;
  analyze) analyze ;;
  pack) pack ;;
  status) status ;;
  all)
    preflight
    prepare
    check_isvd
    evaluator_seal
    seal
    qual
    decide_qual
    formal
    analyze
    pack
    ;;
  *)
    die "Usage: bash scripts/run_indep_confirm.sh {preflight|prepare|check-isvd|evaluator-seal|seal|qual|decide-qual|formal|analyze|pack|status|all}"
    ;;
esac
