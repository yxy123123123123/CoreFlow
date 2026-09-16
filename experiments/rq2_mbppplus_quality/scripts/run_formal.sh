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

install_assets() {
  mkdir -p "$REPORT_ROOT"
  "$PYTHON" "$SCRIPT_DIR/install_assets.py" \
    --archive "$ROOT/payload/coreflow_m2_assets_payload_20260726.tar.gz" \
    --payload-lock "$ROOT/provenance/ASSET_PAYLOAD_LOCK.json" \
    --destination "$FORMAL_WORK_ROOT" \
    --marker "$REPORT_ROOT/assets_installed.json"
}

preflight() {
  require_file "$REPORT_ROOT/assets_installed.json" "Run install-assets first"
  "$PYTHON" "$SCRIPT_DIR/formal_control.py" preflight \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/preflight.json"
}

prepare() {
  require_file "$REPORT_ROOT/preflight.json" "Run preflight first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed"
  "$PYTHON" "$ROOT/tests/run_tests.py" | tee "$REPORT_ROOT/unit_tests.json"
  "$PYTHON" "$SCRIPT_DIR/formal_control.py" prepare \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --config "$CONFIG" \
    --output "$REPORT_ROOT/data_audit.json"
}

run_one() {
  local output_group="$1"
  local dataset="$2"
  local method="$3"
  local port="$4"
  local limit="${5:-}"
  local command=(
    "$PYTHON" "$SCRIPT_DIR/run_formal_method.py"
    --formal-work-root "$FORMAL_WORK_ROOT"
    --source-work-root "$SOURCE_WORK_ROOT"
    --model "$MODEL"
    --config "$CONFIG"
    --asset-lock "$ASSET_LOCK"
    --frozen-data "$dataset"
    --method "$method"
    --output "$RESULT_ROOT/$output_group/$method"
    --port "$port"
  )
  if [[ -n "$limit" ]]; then command+=(--limit "$limit"); fi
  "${command[@]}"
}

legacy_pair() {
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed; F0 legacy pair must precede seal"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase legacy_pair)
  local port=5720
  for method in "${methods[@]}"; do
    run_one "f0_legacy_pair" "$DATA_ROOT/legacy_m2_math_300.jsonl" "$method" "$port"
    port=$((port + 1))
  done
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" legacy \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT/f0_legacy_pair" \
    --output "$REPORT_ROOT/legacy_math_k5_seed42_pair.json" \
    --summary-md "$REPORT_ROOT/LEGACY_MATH_K5_SEED42_PAIR.md"
}

smoke() {
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  [[ ! -f "$SEAL" ]] || die "Protocol already sealed; smoke must precede seal"
  run_one "f0_smoke" "$DATA_ROOT/code_dev_128.jsonl" "code_full_k5_seed41" 5730 8
  run_one "f0_smoke" "$DATA_ROOT/code_dev_128.jsonl" "code_core_q185_k5_seed41" 5731 8
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" smoke \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT/f0_smoke" \
    --output "$REPORT_ROOT/smoke_decision.json" \
    --summary-md "$REPORT_ROOT/SMOKE_SUMMARY.md"
}

seal() {
  require_file "$REPORT_ROOT/preflight.json" "Run preflight first"
  require_file "$REPORT_ROOT/data_audit.json" "Run prepare first"
  require_file "$REPORT_ROOT/legacy_math_k5_seed42_pair.json" "Run legacy-pair first"
  require_file "$REPORT_ROOT/smoke_decision.json" "Run smoke first"
  "$PYTHON" "$SCRIPT_DIR/formal_control.py" seal \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --config "$CONFIG" \
    --preflight "$REPORT_ROOT/preflight.json" \
    --data-audit "$REPORT_ROOT/data_audit.json" \
    --legacy-pair "$REPORT_ROOT/legacy_math_k5_seed42_pair.json" \
    --smoke "$REPORT_ROOT/smoke_decision.json" \
    --output "$SEAL"
}

code_floor() {
  require_file "$SEAL" "Run seal before any formal phase"
  run_one "code_floor" "$DATA_ROOT/code_dev_128.jsonl" "code_full_k5_seed41" 5740
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" floor \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT/code_floor" \
    --output "$REPORT_ROOT/code_floor_decision.json" \
    --summary-md "$REPORT_ROOT/CODE_FLOOR_DECISION.md"
}

math_main() {
  require_file "$SEAL" "Run seal first"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase math_main)
  local port=5750
  for method in "${methods[@]}"; do
    run_one "math_main" "$DATA_ROOT/math_formal_300.jsonl" "$method" "$port"
    port=$((port + 1))
  done
}

floor_role() {
  "$PYTHON" -c "import json; print(json.load(open('$REPORT_ROOT/code_floor_decision.json', encoding='utf-8'))['decision'])"
}

code_main() {
  require_file "$SEAL" "Run seal first"
  require_file "$REPORT_ROOT/code_floor_decision.json" "Run code-floor first"
  local role
  role="$(floor_role)"
  if [[ "$role" == "CODE_STRESS_ONLY" ]]; then
    printf '[STOP-RULE] code formal split remains closed because floor decision is CODE_STRESS_ONLY.\n'
    return
  fi
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase code_main)
  local port=5760
  for method in "${methods[@]}"; do
    run_one "code_main" "$DATA_ROOT/code_formal_300.jsonl" "$method" "$port"
    port=$((port + 1))
  done
}

primary_analyze() {
  require_file "$REPORT_ROOT/code_floor_decision.json" "Run code-floor first"
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" primary \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT" \
    --floor-decision "$REPORT_ROOT/code_floor_decision.json" \
    --output "$REPORT_ROOT/primary_decision.json" \
    --summary-md "$REPORT_ROOT/PRIMARY_SUMMARY.md"
}

k_extension() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  local role
  role="$(floor_role)"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase k_extension)
  local port=5770
  for method in "${methods[@]}"; do
    if [[ "$role" == "CODE_STRESS_ONLY" && "$method" == code_* ]]; then
      printf '[STOP-RULE] skip formal-code extension %s; code split remains closed.\n' "$method"
      continue
    fi
    if [[ "$method" == code_* ]]; then
      dataset="$DATA_ROOT/code_formal_300.jsonl"
    else
      dataset="$DATA_ROOT/math_formal_300.jsonl"
    fi
    run_one "k_extension" "$dataset" "$method" "$port"
    port=$((port + 1))
  done
}

baselines() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  local role
  role="$(floor_role)"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" --config "$CONFIG" --phase baselines)
  local port=5780
  for method in "${methods[@]}"; do
    if [[ "$role" == "CODE_STRESS_ONLY" && "$method" == code_* ]]; then
      printf '[STOP-RULE] skip formal-code baseline %s; code split remains closed.\n' "$method"
      continue
    fi
    if [[ "$method" == code_* ]]; then
      dataset="$DATA_ROOT/code_formal_300.jsonl"
    else
      dataset="$DATA_ROOT/math_formal_300.jsonl"
    fi
    run_one "baselines" "$dataset" "$method" "$port"
    port=$((port + 1))
  done
}

activate_q224() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  local tasks="${2:-}"
  local rationale="${3:-}"
  [[ -n "$tasks" && -n "$rationale" ]] || die "Usage: bash scripts/run_formal.sh activate-q224 math,code 'frozen rationale'"
  "$PYTHON" "$SCRIPT_DIR/activate_q224.py" \
    --primary-decision "$REPORT_ROOT/primary_decision.json" \
    --q224-result-root "$RESULT_ROOT/q224" \
    --tasks "$tasks" \
    --rationale "$rationale" \
    --output "$REPORT_ROOT/q224_activation.json"
}

q224() {
  require_file "$REPORT_ROOT/q224_activation.json" "q224 is conditional; discuss q185 failure, then run activate-q224 before any q224 output"
  mapfile -t methods < <("$PYTHON" "$SCRIPT_DIR/list_methods.py" \
    --config "$CONFIG" --phase q224_conditional --activation "$REPORT_ROOT/q224_activation.json")
  local port=5790
  for method in "${methods[@]}"; do
    if [[ "$method" == code_* ]]; then
      dataset="$DATA_ROOT/code_formal_300.jsonl"
    else
      dataset="$DATA_ROOT/math_formal_300.jsonl"
    fi
    run_one "q224" "$dataset" "$method" "$port"
    port=$((port + 1))
  done
}

microbenchmark() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  "$PYTHON" "$SCRIPT_DIR/run_microbenchmark.py" \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --model "$MODEL" \
    --config "$CONFIG" \
    --asset-lock "$ASSET_LOCK" \
    --output-root "$RESULT_ROOT/microbenchmark" \
    --output "$REPORT_ROOT/microbenchmark_decision.json"
}

analyze() {
  require_file "$REPORT_ROOT/primary_decision.json" "Run primary-analyze first"
  require_file "$REPORT_ROOT/microbenchmark_decision.json" "Run microbenchmark first"
  local extra=()
  if [[ -f "$REPORT_ROOT/q224_activation.json" ]]; then
    extra+=(--q224-activation "$REPORT_ROOT/q224_activation.json")
  fi
  "$PYTHON" "$SCRIPT_DIR/analyze_formal.py" secondary \
    --config "$CONFIG" \
    --result-root "$RESULT_ROOT" \
    --primary-decision "$REPORT_ROOT/primary_decision.json" \
    --microbenchmark "$REPORT_ROOT/microbenchmark_decision.json" \
    "${extra[@]}" \
    --output "$REPORT_ROOT/secondary_decision.json" \
    --summary-md "$REPORT_ROOT/SECONDARY_SUMMARY.md"
}

pack() {
  require_file "$REPORT_ROOT/primary_decision.json" "Primary analysis is incomplete"
  require_file "$REPORT_ROOT/secondary_decision.json" "Secondary analysis is incomplete"
  local output="/root/autodl-tmp/coreflow_formal_v1_results.tar.gz"
  tar -czf "$output" \
    -C "$FORMAL_WORK_ROOT" \
    reports/formal_v1 results/formal_v1 data/formal_v1 reports/m2_assets \
    -C "$ROOT" \
    config/formal_protocol.json provenance docs README.md PACKAGE_MANIFEST.sha256
  sha256sum "$output" > "$output.sha256"
  ls -lh "$output" "$output.sha256"
}

status() {
  "$PYTHON" "$SCRIPT_DIR/status_formal.py" \
    --formal-work-root "$FORMAL_WORK_ROOT" \
    --config "$CONFIG"
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi
case "$PHASE" in
  install-assets) install_assets ;;
  preflight) preflight ;;
  prepare) prepare ;;
  legacy-pair) legacy_pair ;;
  smoke) smoke ;;
  seal) seal ;;
  code-floor) code_floor ;;
  math-main) math_main ;;
  code-main) code_main ;;
  primary-analyze) primary_analyze ;;
  k-extension) k_extension ;;
  baselines) baselines ;;
  activate-q224) activate_q224 "$@" ;;
  q224) q224 ;;
  microbenchmark) microbenchmark ;;
  analyze) analyze ;;
  pack) pack ;;
  status) status ;;
  all-primary)
    install_assets
    preflight
    prepare
    legacy_pair
    smoke
    seal
    code_floor
    math_main
    code_main
    primary_analyze
    ;;
  all-secondary)
    k_extension
    baselines
    microbenchmark
    analyze
    pack
    ;;
  *)
    die "Usage: bash scripts/run_formal.sh {install-assets|preflight|prepare|legacy-pair|smoke|seal|code-floor|math-main|code-main|primary-analyze|k-extension|baselines|activate-q224|q224|microbenchmark|analyze|pack|status|all-primary|all-secondary}"
    ;;
esac
