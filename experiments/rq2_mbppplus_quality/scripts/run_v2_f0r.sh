#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v2_f0r_mbpp_interface_workspace}"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
M2_ASSET_ROOT="${M2_ASSET_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace/m2_assets}"
M2_ASSET_LOCK="${M2_ASSET_LOCK:-/root/autodl-tmp/coreflow_formal_v1_workspace/reports/m2_assets/asset_lock.json}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
CONFIG="$ROOT/config/formal_v2_f0r_protocol.json"
REPORT_ROOT="$WORK_ROOT/reports/formal_v2_f0r"
RESULT_ROOT="$WORK_ROOT/results/formal_v2_f0r"
DATA_ROOT="$WORK_ROOT/data/formal_v2_f0r"
SEAL="$REPORT_ROOT/PROTOCOL_SEAL.json"
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

ensure_asset_link() {
  mkdir -p "$WORK_ROOT" "$REPORT_ROOT" "$RESULT_ROOT"
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
    --config "$CONFIG" \
    --output "$REPORT_ROOT/preflight.json"
}

prepare() {
  require_file "$REPORT_ROOT/preflight.json" "Run preflight first"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" prepare \
    --work-root "$WORK_ROOT" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/data_audit.json"
}

build_isvd() {
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" build-isvd \
    --work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --m2-asset-lock "$M2_ASSET_LOCK" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/isvd_asset_lock.json"
}

e0() {
  require_file "$REPORT_ROOT/isvd_asset_lock.json" "Run build-isvd first"
  "$PYTHON" "$SCRIPT_DIR/run_e0_correctness.py" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/e0_correctness.json"
}

seal() {
  require_file "$REPORT_ROOT/preflight.json" "Run preflight first"
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  require_file "$REPORT_ROOT/isvd_asset_lock.json" "Run build-isvd first"
  require_file "$REPORT_ROOT/e0_correctness.json" "Run e0 first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" seal \
    --work-root "$WORK_ROOT" \
    --config "$CONFIG" \
    --preflight "$REPORT_ROOT/preflight.json" \
    --data-audit "$REPORT_ROOT/data_audit.json" \
    --isvd-asset-lock "$REPORT_ROOT/isvd_asset_lock.json" \
    --e0-correctness "$REPORT_ROOT/e0_correctness.json" \
    --output "$SEAL"
}

run_one() {
  local method="$1"
  local port="$2"
  local dataset
  if [[ "$method" == code_* ]]; then
    dataset="$DATA_ROOT/mbppplus_dev128.jsonl"
  else
    dataset="$DATA_ROOT/cmath_dev128.jsonl"
  fi
  "$PYTHON" "$SCRIPT_DIR/run_formal_method.py" \
    --formal-work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --config "$CONFIG" \
    --asset-lock "$M2_ASSET_LOCK" \
    --frozen-data "$dataset" \
    --method "$method" \
    --output "$RESULT_ROOT/go_nogo_quality/$method" \
    --port "$port"
}

go_nogo() {
  require_file "$SEAL" "Run seal before dev model generation"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase go_nogo_quality)
  local port=6100
  for method in "${methods[@]}"; do
    run_one "$method" "$port"
    port=$((port + 1))
  done
}

go_nogo_2gpu() {
  require_file "$SEAL" "Run seal before dev model generation"
  "$PYTHON" "$SCRIPT_DIR/run_formal_parallel.py" \
    --formal-work-root "$WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --python "$PYTHON" \
    --config "$CONFIG" \
    --asset-lock "$M2_ASSET_LOCK" \
    --data-root "$DATA_ROOT" \
    --result-root "$RESULT_ROOT" \
    --phase go_nogo_quality \
    --output-group go_nogo_quality \
    --devices "${CUDA_DEVICES:-0,1}" \
    --base-port 6200
}

analyze() {
  require_file "$SEAL" "Run seal first"
  "$PYTHON" "$SCRIPT_DIR/v2_control.py" analyze \
    --work-root "$WORK_ROOT" \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT" \
    --output "$REPORT_ROOT/go_nogo_decision.json"
}

pack() {
  require_file "$REPORT_ROOT/go_nogo_decision.json" "Run analyze first"
  local output="/root/autodl-tmp/coreflow_formal_v2_f0r_mbpp_interface_results.tar.gz"
  tar -czf "$output" \
    -C "$WORK_ROOT" \
    reports/formal_v2_f0r results/formal_v2_f0r data/formal_v2_f0r isvd_banks \
    -C "$ROOT" \
    config provenance docs README.md PACKAGE_MANIFEST.sha256
  sha256sum "$output" > "$output.sha256"
  ls -lh "$output" "$output.sha256"
}

status() {
  "$PYTHON" "$SCRIPT_DIR/status_v2_f0r.py" --work-root "$WORK_ROOT"
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi

case "$PHASE" in
  preflight) preflight ;;
  prepare) prepare ;;
  build-isvd) build_isvd ;;
  e0) e0 ;;
  seal) seal ;;
  go-nogo) go_nogo ;;
  go-nogo-2gpu) go_nogo_2gpu ;;
  analyze) analyze ;;
  pack) pack ;;
  status) status ;;
  all)
    preflight
    prepare
    build_isvd
    e0
    seal
    go_nogo
    analyze
    pack
    ;;
  all-2gpu)
    preflight
    prepare
    build_isvd
    e0
    seal
    go_nogo_2gpu
    analyze
    pack
    ;;
  *)
    die "Usage: bash scripts/run_v2_f0r.sh {preflight|prepare|build-isvd|e0|seal|go-nogo|go-nogo-2gpu|analyze|pack|status|all|all-2gpu}"
    ;;
esac
