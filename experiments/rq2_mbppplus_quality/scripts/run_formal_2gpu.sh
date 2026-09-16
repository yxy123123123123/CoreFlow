#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
FORMAL_WORK_ROOT="${FORMAL_WORK_ROOT:-/root/autodl-tmp/coreflow_formal_v1_workspace}"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
CONFIG="$ROOT/config/formal_protocol.json"
REPORT_ROOT="$FORMAL_WORK_ROOT/reports/formal_v1"
RESULT_ROOT="$FORMAL_WORK_ROOT/results/formal_v1"
DATA_ROOT="$FORMAL_WORK_ROOT/data/formal_v1"
ASSET_LOCK="$FORMAL_WORK_ROOT/reports/m2_assets/asset_lock.json"
SEAL="$REPORT_ROOT/PROTOCOL_SEAL.json"
DEVICES="${CUDA_DEVICES:-0,1}"
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

serial() {
  bash "$SCRIPT_DIR/run_formal.sh" "$@"
}

parallel_methods() {
  "$PYTHON" "$SCRIPT_DIR/run_formal_parallel.py" \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --python "$PYTHON" \
    --config "$CONFIG" \
    --asset-lock "$ASSET_LOCK" \
    --data-root "$DATA_ROOT" \
    --result-root "$RESULT_ROOT" \
    --devices "$DEVICES" \
    "$@"
}

legacy_pair_2gpu() {
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed; F0 legacy pair must precede seal"
  parallel_methods \
    --phase legacy_pair \
    --output-group f0_legacy_pair \
    --dataset "$DATA_ROOT/legacy_m2_math_300.jsonl" \
    --base-port 5920
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" legacy \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT/f0_legacy_pair" \
    --output "$REPORT_ROOT/legacy_math_k5_seed42_pair.json" \
    --summary-md "$REPORT_ROOT/LEGACY_MATH_K5_SEED42_PAIR.md"
}

smoke_2gpu() {
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed; smoke must precede seal"
  parallel_methods \
    --methods code_full_k5_seed41,code_core_q185_k5_seed41 \
    --output-group f0_smoke \
    --dataset "$DATA_ROOT/code_dev_128.jsonl" \
    --base-port 5930 \
    --limit 8
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" smoke \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT/f0_smoke" \
    --output "$REPORT_ROOT/smoke_decision.json" \
    --summary-md "$REPORT_ROOT/SMOKE_SUMMARY.md"
}

math_main_2gpu() {
  require_file "$SEAL" "Run seal first"
  parallel_methods --phase math_main --output-group math_main --base-port 5950
}

code_main_2gpu() {
  require_file "$SEAL" "Run seal first"
  require_file "$REPORT_ROOT/code_floor_decision.json" "Run code-floor first"
  local role
  role="$("$PYTHON" -c "import json; print(json.load(open('$REPORT_ROOT/code_floor_decision.json', encoding='utf-8'))['decision'])")"
  if [[ "$role" == "CODE_STRESS_ONLY" ]]; then
    printf '[STOP-RULE] code formal split remains closed because floor decision is CODE_STRESS_ONLY.\n'
    return
  fi
  parallel_methods --phase code_main --output-group code_main --base-port 5960
}

primary_after_seal_2gpu() {
  serial code-floor
  math_main_2gpu
  code_main_2gpu
  serial primary-analyze
}

k_extension_2gpu() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  parallel_methods \
    --phase k_extension \
    --output-group k_extension \
    --base-port 5970 \
    --floor-decision "$REPORT_ROOT/code_floor_decision.json" \
    --skip-code-if-stress-only
}

baselines_2gpu() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  parallel_methods \
    --phase baselines \
    --output-group baselines \
    --base-port 5980 \
    --floor-decision "$REPORT_ROOT/code_floor_decision.json" \
    --skip-code-if-stress-only
}

q224_2gpu() {
  require_file "$REPORT_ROOT/q224_activation.json" "q224 is conditional; run activate-q224 only after discussing q185 failure"
  parallel_methods \
    --phase q224_conditional \
    --activation "$REPORT_ROOT/q224_activation.json" \
    --output-group q224 \
    --base-port 5990
}

all_primary_2gpu() {
  serial install-assets
  serial preflight
  serial prepare
  legacy_pair_2gpu
  smoke_2gpu
  serial seal
  primary_after_seal_2gpu
}

all_secondary_2gpu() {
  k_extension_2gpu
  baselines_2gpu
  serial microbenchmark
  serial analyze
  serial pack
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi
case "$PHASE" in
  legacy-pair-2gpu) legacy_pair_2gpu ;;
  smoke-2gpu) smoke_2gpu ;;
  math-main-2gpu) math_main_2gpu ;;
  code-main-2gpu) code_main_2gpu ;;
  primary-after-seal-2gpu) primary_after_seal_2gpu ;;
  k-extension-2gpu) k_extension_2gpu ;;
  baselines-2gpu) baselines_2gpu ;;
  q224-2gpu) q224_2gpu ;;
  all-primary-2gpu) all_primary_2gpu ;;
  all-secondary-2gpu) all_secondary_2gpu ;;
  status) serial status ;;
  *)
    die "Usage: bash scripts/run_formal_2gpu.sh {legacy-pair-2gpu|smoke-2gpu|math-main-2gpu|code-main-2gpu|primary-after-seal-2gpu|k-extension-2gpu|baselines-2gpu|q224-2gpu|all-primary-2gpu|all-secondary-2gpu|status}"
    ;;
esac
